#!/usr/bin/env python3
"""
TinyStories → GSM8K Experiment Chain
====================================
Automated experiment for DGE paper validation.

Phases:
1. Create model (d=384, L=12, H=6, ctx=1024)
2. Train TinyStories (train split)
3. Evaluate TinyStories baseline (validation split)
4. Expand model (+640 d_model → 1024)
5. Train GSM8K (train split, asymmetric replay)
6. Evaluate TinyStories retention (validation split)
7. Evaluate GSM8K (test split)
8. Save experiment log (JSON)

Usage:
    python run_tinystories_gsm8k_chain.py
"""

import os
import sys
import json
import time
import math
import torch
import torch.optim as optim
from datetime import datetime

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dge_model import DGESimpleTransformer
from dge_training import train_dataset, DEVICE
from dge_logger import DGELogger
from replay_buffer import ReplayBuffer
from data import load_tinystories, load_gsm8k
from version import __version__

# Windows console-safe print (handles Unicode emojis)
def safe_print(msg):
    """Print with fallback for Windows cp1252 encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Strip or replace emoji characters for Windows console
        print(msg.encode('ascii', 'replace').decode('ascii'))

# ============================================================================
# CONFIGURATION
# ============================================================================

def detect_gpu_config():
    """
    Detect GPU and VRAM, return optimal batch sizes.
    Returns tuple: (tinystories_batch, gsm8k_batch, gpu_name, vram_gb)
    """
    if not torch.cuda.is_available():
        print("⚠️ No CUDA GPU detected. Using CPU settings (very slow).")
        return 4, 2, "CPU", 0
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    
    print(f"🖥️ GPU Detected: {gpu_name}")
    print(f"💾 VRAM: {vram_gb:.1f} GB")
    
    # Dynamic batch sizes based on VRAM
    if vram_gb >= 40:  # A40, A100-40GB, etc.
        ts_batch, gsm8k_batch = 64, 32
        print(f"✅ High VRAM ({vram_gb:.0f}GB): Using large batches (64/32)")
    elif vram_gb >= 24:  # RTX 4090, A4000, etc.
        ts_batch, gsm8k_batch = 32, 8
        print(f"✅ Medium VRAM ({vram_gb:.0f}GB): Using medium batches (32/8)")
    elif vram_gb >= 16:  # RTX 4080, T4, etc.
        ts_batch, gsm8k_batch = 16, 4
        print(f"⚠️ Limited VRAM ({vram_gb:.0f}GB): Using small batches (16/4)")
    else:  # <16GB
        ts_batch, gsm8k_batch = 8, 2
        print(f"⚠️ Low VRAM ({vram_gb:.0f}GB): Using tiny batches (8/2)")
    
    return ts_batch, gsm8k_batch, gpu_name, vram_gb

# Detect GPU and set optimal batch sizes
TS_BATCH, GSM8K_BATCH, GPU_NAME, VRAM_GB = detect_gpu_config()

# Scale learning rates with batch size
TS_LR = 1e-4 * (TS_BATCH / 16)  # Base LR at batch 16
GSM8K_LR = 5e-5 * (GSM8K_BATCH / 8)  # Base LR at batch 8

CONFIG = {
    # Model Architecture (fixed for entire experiment)
    "vocab_size": 50257,      # GPT-2 tokenizer
    "n_layer": 12,            # Cannot expand - fixed at 12
    "max_seq_len": 1024,      # Cannot expand - fixed at 1024
    
    # TinyStories Phase
    "tinystories_d_model": 384,
    "tinystories_n_head": 6,  # head_dim = 64
    "tinystories_epochs": 1,
    "tinystories_batch_size": TS_BATCH,   # Dynamic based on GPU
    "tinystories_seq_len": 256,
    "tinystories_max_samples": None,  # None = all (~2M)
    "tinystories_lr": TS_LR,
    
    # Expansion
    "expansion_delta": 640,   # 384 -> 1024
    
    # GSM8K Phase (AFTER EXPANSION - model is 3x larger!)
    "gsm8k_d_model": 1024,    # After expansion
    "gsm8k_n_head": 16,       # head_dim = 64
    "gsm8k_epochs": 3,
    "gsm8k_batch_size": GSM8K_BATCH,  # Dynamic based on GPU
    "gsm8k_seq_len": 256,
    "gsm8k_max_samples": None,  # None = all (~7.5K)
    "gsm8k_lr": GSM8K_LR,
    "gsm8k_replay_ratio": 0.1,  # 10% replay from TinyStories
    
    # Paths & Checkpointing
    "output_dir": "models/tinystories_gsm8k_chain",
    "local_checkpoint_interval": 1000,  # Local restorepoint (fast crash recovery)
    "hf_upload_interval": 5000,         # Remote backup to HF (bandwidth efficient)
    "git_backup_interval": 5000,        # Push logs to Git every N steps
    
    # GPU Info (for logging)
    "gpu_name": GPU_NAME,
    "vram_gb": VRAM_GB,
}

# HuggingFace Hub configuration
HF_REPO = "darealSven/dge-tinystories-gsm8k"

# ============================================================================
# GIT BACKUP FOR LOGS
# ============================================================================

def git_backup_logs(output_dir, step, commit_message=None):
    """
    Push experiment logs to Git repository.
    Called at regular intervals to preserve training metrics.
    """
    import subprocess
    
    try:
        if commit_message is None:
            commit_message = f"Experiment logs at step {step}"
        
        # Add log files (exclude large model files)
        log_patterns = [
            f"{output_dir}/*.jsonl",
            f"{output_dir}/*.json",
            f"{output_dir}/experiment_log.json",
            f"{output_dir}/dge_*.jsonl",
        ]
        
        # Git add (only logs, not weights)
        for pattern in log_patterns:
            subprocess.run(
                ["git", "add", "-f", pattern], 
                capture_output=True, 
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        
        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if "nothing to commit" in result.stdout or result.returncode == 1:
            safe_print(f"[GIT] No log changes to commit at step {step}")
            return False
        
        # Push
        result = subprocess.run(
            ["git", "push", "origin", "master"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            safe_print(f"[GIT] Logs pushed to Git at step {step}")
            return True
        else:
            safe_print(f"[GIT] Push failed: {result.stderr[:100]}")
            return False
            
    except Exception as e:
        safe_print(f"[GIT] Backup failed: {e}")
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_perplexity(model, dataloader, max_batches=100):
    """Compute perplexity on a dataloader (limited batches for speed)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = DEVICE
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y, sparsity_lambda=0.0)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    model.train()
    return perplexity, avg_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_chunked_archive(source_dir, output_prefix, chunk_size_mb=400):
    """
    Zip a directory and split it into chunks of given size.
    Returns list of generated chunk files.
    
    Uses ZIP64 extensions for files >4GB and DEFLATED compression.
    """
    import shutil
    import zipfile
    
    # 1. Zip to temporary single file using ZIP64 to support >4GB
    temp_zip = output_prefix + ".temp.zip"
    
    # Use DEFLATED for compression, ZIP64 for large files
    try:
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    # Skip the temp zip itself and any existing chunks
                    if file.endswith('.temp.zip') or '.zip.' in file:
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)
    except Exception as e:
        safe_print(f"[ZIP] Compression failed: {e}, trying without compression...")
        # Fallback to STORED (no compression)
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file.endswith('.temp.zip') or '.zip.' in file:
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)
    
    # 2. Split into chunks
    chunk_size = chunk_size_mb * 1024 * 1024
    chunk_files = []
    
    real_temp_zip = temp_zip
    
    part_num = 0
    with open(real_temp_zip, 'rb') as src:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            part_filename = f"{output_prefix}.zip.{part_num:03d}"
            with open(part_filename, 'wb') as dst:
                dst.write(chunk)
            chunk_files.append(part_filename)
            part_num += 1
            
    # 3. Cleanup temp file
    if os.path.exists(real_temp_zip):
        os.remove(real_temp_zip)
        
    return chunk_files

def restore_chunked_archive(archive_prefix, output_dir):
    """
    Reassemble split zip parts and unzip to output_dir.
    """
    import glob
    import zipfile
    import shutil
    
    # 1. Find parts
    parts = sorted(glob.glob(f"{archive_prefix}.zip.*"))
    if not parts:
        raise FileNotFoundError(f"No archive parts found for {archive_prefix}")
        
    # 2. Reassemble
    temp_zip = archive_prefix + ".reconstructed.zip"
    with open(temp_zip, 'wb') as dst:
        for part in parts:
            with open(part, 'rb') as src:
                shutil.copyfileobj(src, dst)
                
    # 3. Unzip
    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    # 4. Cleanup
    if os.path.exists(temp_zip):
        os.remove(temp_zip)

def ensure_checkpoint_restored(ckpt_path):
    """
    Check if checkpoint is chunked. If so, restore it.
    Returns True if weights.pt exists (or was restored), False otherwise.
    """
    import glob
    weights_path = os.path.join(ckpt_path, "weights.pt")
    if os.path.exists(weights_path):
        return True
    
    # Check for archive
    archive_prefix = os.path.join(ckpt_path, "checkpoint_archive")
    # Check for .zip.000 or similar
    if glob.glob(archive_prefix + ".zip.*"):
        print(f"📦 Restoring chunked checkpoint: {ckpt_path}...")
        try:
            restore_chunked_archive(archive_prefix, ckpt_path)
            if os.path.exists(weights_path):
                print(f"   ✅ Restored successfully.")
                return True
        except Exception as e:
            print(f"❌ Restoration failed: {e}")
            return False
            
    return False

import threading
import queue

_upload_queue = queue.Queue()
_upload_thread = None
_upload_thread_running = False

def _upload_worker():
    """Background worker that uploads checkpoints to HuggingFace Hub."""
    global _upload_thread_running
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set - background uploads disabled")
        return
    
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=hf_token)
        
        # Create repo if it doesn't exist (first upload will create it)
        try:
            create_repo(repo_id=HF_REPO, token=hf_token, private=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Repo creation note: {e}")
        
    except ImportError:
        print("⚠️ huggingface_hub not installed - background uploads disabled")
        return
    
    _upload_thread_running = True
    
    while _upload_thread_running:
        try:
            # Wait for upload task (timeout allows checking running flag)
            task = _upload_queue.get(timeout=5)
            if task is None:  # Shutdown signal
                break
            
            folder_path, step = task
            folder_name = os.path.basename(folder_path)
            
            # Milestones go to subfolders to preserve them; resume_checkpoint goes to root
            if folder_name.startswith("milestone_"):
                path_in_repo = folder_name  # e.g., "milestone_tinystories/"
                print(f"☁️ [Background] Uploading milestone: {folder_name} to HF Hub...")
            else:
                path_in_repo = ""  # Root for resume_checkpoint (overwrites)
                print(f"☁️ [Background] Uploading {folder_path} to HF Hub...")
                
                # DELETE OLD CHECKPOINT ARCHIVES to prevent mixing old/new chunks
                try:
                    from huggingface_hub import list_repo_files, delete_file
                    existing_files = list_repo_files(HF_REPO, token=hf_token)
                    old_archives = [f for f in existing_files if "checkpoint_archive" in f and "/" not in f]
                    if old_archives:
                        print(f"☁️ [Background] Deleting {len(old_archives)} old archive chunks...")
                        for old_file in old_archives:
                            try:
                                api.delete_file(old_file, repo_id=HF_REPO, token=hf_token, 
                                              commit_message=f"Cleanup before step {step}")
                            except:
                                pass  # Best effort
                except Exception as del_err:
                    print(f"☁️ [Background] Could not cleanup old files: {del_err}")
            
            try:
                api.upload_folder(
                    folder_path=folder_path,
                    path_in_repo=path_in_repo,
                    repo_id=HF_REPO,
                    repo_type="model",
                    commit_message=f"Checkpoint at step {step}"
                )
                print(f"☁️ [Background] ✅ Uploaded: {folder_path}")
            except Exception as e:
                print(f"☁️ [Background] ❌ Upload failed: {e}")
            
            _upload_queue.task_done()
            
        except queue.Empty:
            continue  # Just check running flag
        except Exception as e:
            print(f"☁️ [Background] Worker error: {e}")


def start_upload_worker():
    """Start the background upload worker thread."""
    global _upload_thread
    if _upload_thread is None or not _upload_thread.is_alive():
        _upload_thread = threading.Thread(target=_upload_worker, daemon=True)
        _upload_thread.start()
        print("☁️ Background HF upload worker started")


def upload_to_hf_async(folder_path, step):
    """Queue a folder for background upload to HuggingFace Hub."""
    _upload_queue.put((folder_path, step))


def shutdown_upload_worker():
    """Gracefully shutdown the upload worker and wait for pending uploads."""
    global _upload_thread_running
    _upload_thread_running = False
    _upload_queue.put(None)  # Shutdown signal
    if _upload_thread and _upload_thread.is_alive():
        print("☁️ Waiting for pending uploads to complete...")
        _upload_thread.join(timeout=300)  # Wait up to 5 minutes


# Track previous checkpoint paths for deletion
_previous_checkpoints = {}


def save_checkpoint(model, optimizer, path, step, config, save_optimizer=True, is_rolling=True, upload_to_hf=True):
    """
    Save model checkpoint with config.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        path: Directory path to save to
        step: Current training step
        config: Configuration dictionary
        save_optimizer (bool): Whether to save optimizer state. False for Milestones (weights only).
        is_rolling (bool): Whether to delete previous checkpoint of same key. True for Resume, False for Milestones.
        upload_to_hf (bool): Whether to queue this checkpoint for HuggingFace upload. 
                             Set False for frequent local checkpoints to save bandwidth.
    """
    import shutil
    
    # Get checkpoint category (e.g., "tinystories_checkpoint", "milestone_tinystories")
    checkpoint_key = os.path.basename(path)
    prev_path = _previous_checkpoints.get(checkpoint_key)
    
    # === PREPARE RESUME STATE (if exists) ===
    # If rolling (resume checkpoint), we want to include the resume_state.json
    # This file lives in the parent directory (output_dir)
    parent_dir = os.path.dirname(path)
    resume_file_src = os.path.join(parent_dir, "resume_state.json")
    resume_file_dst = os.path.join(path, "resume_state.json")
    
    # === DISK SPACE CHECK ===
    try:
        total, used, free = shutil.disk_usage(parent_dir if parent_dir else ".")
        free_gb = free / (2**30)
        safe_print(f"[DISK] Free: {free_gb:.2f} GB")
        
        if free_gb < 2.0:
            safe_print(f"[WARN] Critical disk space! Only {free_gb:.2f} GB left. Checkpoint may fail.")
        elif free_gb < 5.0:
            safe_print(f"[WARN] Low disk space: {free_gb:.2f} GB")
    except Exception as e:
        safe_print(f"[DISK] Could not check space: {e}")
    
    # === SAVE FIRST (before any deletion) ===
    os.makedirs(path, exist_ok=True)
    
    try:
        # Save weights
        weights_path = os.path.join(path, "weights.pt")
        torch.save(model.state_dict(), weights_path)
        
        # Save optimizer state (ONLY if requested)
        if save_optimizer:
            optimizer_path = os.path.join(path, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save config
        checkpoint_config = {
            "step": step,
            "d_model": model.d_model,
            "n_layer": len(model.layers),
            "n_head": model.layers[0].n_head,
            "vocab_size": model.token_emb.num_embeddings,
            "max_seq_len": model.max_seq_len,
            "timestamp": datetime.now().isoformat(),
            **config
        }
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(checkpoint_config, f, indent=2)
            
        # Copy resume_state.json if it exists and we are rolling (or always? harmless if always)
        if os.path.exists(resume_file_src):
            shutil.copy(resume_file_src, resume_file_dst)
        
        # Verify files exist
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise IOError("Checkpoint files were not saved correctly")
        
        safe_print(f"[SAVE] Checkpoint saved: {path} (Optim: {save_optimizer}, Rolling: {is_rolling})")
        
        # === CHUNK & COMPRESS (Moon Landing V2) ===
        # Zip and split large files to avoid HF/Git limits
        safe_print(f"[ZIP] Compressing and chunking checkpoint...")
        
        # 1. Create Archive (excluding json files initially? No, make_archive zips everything)
        # We want to zip `weights.pt` and `optimizer.pt`
        # But `config.json` and `resume_state.json` should ideally remain visible?
        # create_chunked_archive zips the whole folder.
        
        chunks = create_chunked_archive(path, os.path.join(path, "checkpoint_archive"), chunk_size_mb=400)
        print(f"   -> Created {len(chunks)} chunks.")
        
        # 2. Delete original large files to save space
        if os.path.exists(weights_path):
            os.remove(weights_path)
        if save_optimizer and os.path.exists(optimizer_path):
            os.remove(optimizer_path)
        
        # Note: config.json and resume_state.json remain in the folder (and inside zip too).
        # This is fine. (Data redundancy is negligible for small files).
            
        save_successful = True
        
    except Exception as e:
        safe_print(f"[ERROR] Checkpoint save FAILED: {e}")
        save_successful = False
    
    # === DELETE PREVIOUS ONLY IF SAVE SUCCEEDED AND IS ROLLING ===
    if save_successful and is_rolling and prev_path and prev_path != path:
        # Double check we're not deleting a milestone by accident (based on path name convention if any, but flags should control)
        if os.path.exists(prev_path):
            try:
                shutil.rmtree(prev_path)
                safe_print(f"[DELETE] Deleted previous checkpoint: {prev_path}")
            except Exception as e:
                safe_print(f"[WARN] Could not delete previous checkpoint: {e}")
    
    # Track this checkpoint for next deletion ONLY IF ROLLING
    if save_successful:
        if is_rolling:
            _previous_checkpoints[checkpoint_key] = path
        
        # Queue for background upload to HuggingFace Hub (only if requested)
        if upload_to_hf:
            upload_to_hf_async(path, step)


def save_resume_state(output_dir, phase, step, extra_data=None):
    """Save resume state so experiment can continue after crash."""
    state = {
        "phase": phase,
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_data:
        state.update(extra_data)
    
    with open(os.path.join(output_dir, "resume_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def load_resume_state(output_dir):
    """Load resume state if exists. Returns None if no resume state."""
    resume_path = os.path.join(output_dir, "resume_state.json")
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            return json.load(f)
    return None




# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def check_disk_space(required_gb=20):
    """
    Verifies that there is enough free disk space.
    """
    import shutil
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (2**30)
        print(f"💾 Disk Space Check: {free_gb:.2f} GB Free")
        
        if free_gb < required_gb:
            print(f"⚠️ WARNING: Low disk space! This experiment requires ~{required_gb}GB.")
            print(f"   You only have {free_gb:.2f}GB.")
            print("   Proceeding at your own risk (checkpoints may fail)...")
            time.sleep(5) 
        else:
            print("✅ Storage Check Passed.")
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")

def run_experiment():
    """Run the full TinyStories → GSM8K experiment chain."""
    
    print("=" * 70)
    print("🔬 DGE EXPERIMENT: TinyStories → GSM8K Chain")
    __version__ = "0.8.0"
    print(f"   Version: {__version__}")
    print(f"   Device: {DEVICE}")
    
    # Pre-flight Check
    check_disk_space(20)
    
    print("=" * 70)
    
    experiment_log = {
        "version": __version__,
        "device": str(DEVICE),
        "config": CONFIG,
        "phases": {},
        "start_time": datetime.now().isoformat(),
    }
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = DGELogger(CONFIG["output_dir"])
    
    # Start background HF upload worker
    start_upload_worker()
    
    # ========================================================================
    # HF CHECKPOINT RECOVERY (Download from Hub if local not present)
    # ========================================================================
    resume_checkpoint_path = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
    if not os.path.exists(resume_checkpoint_path) or not os.path.exists(os.path.join(resume_checkpoint_path, "weights.pt")):
        print(f"   🔍 No local checkpoint found. Checking HuggingFace Hub...")
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            import shutil
            
            # Check if resume_checkpoint exists on HF
            hf_token = os.environ.get("HF_TOKEN")
            files = list_repo_files(HF_REPO, token=hf_token)
            
            # Look for weights.pt (indicates a valid checkpoint)
            if "weights.pt" in files or any("checkpoint_archive" in f for f in files):
                print(f"   ⬇️ Downloading checkpoint from HuggingFace: {HF_REPO}")
                os.makedirs(resume_checkpoint_path, exist_ok=True)
                
                # Download config.json first to determine phase
                if "config.json" in files:
                    local_path = hf_hub_download(HF_REPO, "config.json", token=hf_token)
                    shutil.copy(local_path, os.path.join(resume_checkpoint_path, "config.json"))
                    
                    # Read checkpoint config to determine phase/d_model
                    with open(os.path.join(resume_checkpoint_path, "config.json"), 'r') as f:
                        ckpt_config = json.load(f)
                    ckpt_phase = ckpt_config.get("phase", "unknown")
                    ckpt_d_model = ckpt_config.get("d_model", 384)
                    ckpt_step = ckpt_config.get("step", 0)
                    print(f"   📋 Checkpoint info: phase={ckpt_phase}, d_model={ckpt_d_model}, step={ckpt_step}")
                    
                    # Create resume_state.json based on checkpoint
                    if ckpt_phase == "gsm8k" or ckpt_d_model > CONFIG["tinystories_d_model"]:
                        resume_phase = 5  # GSM8K phase (expanded)
                    elif ckpt_phase == "tinystories":
                        resume_phase = 3  # After TinyStories
                    else:
                        resume_phase = 2  # During TinyStories
                    
                    # Save resume_state to output_dir
                    resume_state_data = {"phase": resume_phase, "step": ckpt_step}
                    with open(os.path.join(CONFIG["output_dir"], "resume_state.json"), 'w') as f:
                        json.dump(resume_state_data, f)
                    print(f"   📝 Created resume_state: phase={resume_phase}, step={ckpt_step}")
                
                # Download chunked archives if present
                archive_files = [f for f in files if "checkpoint_archive" in f]
                if archive_files:
                    print(f"   📦 Downloading {len(archive_files)} archive chunks...")
                    for f in archive_files:
                        local_path = hf_hub_download(HF_REPO, f, token=hf_token)
                        shutil.copy(local_path, os.path.join(resume_checkpoint_path, os.path.basename(f)))
                    # Will be restored by ensure_checkpoint_restored later
                else:
                    # Download individual files
                    for f in ["weights.pt", "optimizer.pt"]:
                        if f in files:
                            local_path = hf_hub_download(HF_REPO, f, token=hf_token)
                            shutil.copy(local_path, os.path.join(resume_checkpoint_path, f))
                
                print(f"   ✅ Downloaded checkpoint from HuggingFace Hub")
            else:
                print(f"   ℹ️ No checkpoint found on HuggingFace Hub. Starting fresh.")
        except Exception as e:
            print(f"   ⚠️ Could not download from HF: {e}")
    
    # ========================================================================
    # RESUME DETECTION
    # ========================================================================
    resume_state = load_resume_state(CONFIG["output_dir"])
    resume_from_phase = 0
    final_step = 0
    
    if resume_state:
        resume_from_phase = resume_state.get("phase", 0)
        final_step = resume_state.get("step", 0)
        print(f"\n🔄 RESUME DETECTED: Continuing from Phase {resume_from_phase}, Step {final_step}")
    
    # ========================================================================
    # PHASE 1: Create Model (or load from checkpoint)
    # ========================================================================
    print("\n" + "=" * 70)
    print("📦 PHASE 1: Creating/Loading Model")
    print("=" * 70)
    
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["tinystories_d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["tinystories_n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    )
    
    optimizer = None
    replay_buffer = ReplayBuffer(max_size=10000, task_name="tinystories")
    
    # Check for existing checkpoints to resume from
    if resume_from_phase >= 5:
        # Load from GSM8K checkpoint (expanded model) - Unified Resume Checkpoint
        gsm8k_ckpt = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
        if ensure_checkpoint_restored(gsm8k_ckpt):
            print(f"   Loading expanded model from {gsm8k_ckpt}...")
            # Expand model first
            model.expand_model(
                new_input_dim=CONFIG["gsm8k_d_model"],
                new_output_dim=CONFIG["vocab_size"],
                router_type='bigram', use_gradient_rescue=True,
                router_init_bias=0.0, gating_threshold=0.0
            )
            model.load_state_dict(torch.load(os.path.join(gsm8k_ckpt, "weights.pt")))
            model = model.to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["gsm8k_lr"], weight_decay=0.01)
            if os.path.exists(os.path.join(gsm8k_ckpt, "optimizer.pt")):
                optimizer.load_state_dict(torch.load(os.path.join(gsm8k_ckpt, "optimizer.pt")))
            print(f"   ✅ Resumed from GSM8K checkpoint at step {final_step}")
    elif resume_from_phase >= 2:
        # Load from TinyStories checkpoint - Unified Resume Checkpoint
        ts_ckpt = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
        if ensure_checkpoint_restored(ts_ckpt):
            print(f"   Loading model from {ts_ckpt}...")
            model.load_state_dict(torch.load(os.path.join(ts_ckpt, "weights.pt")))
            model = model.to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["tinystories_lr"], weight_decay=0.01)
            if os.path.exists(os.path.join(ts_ckpt, "optimizer.pt")):
                optimizer.load_state_dict(torch.load(os.path.join(ts_ckpt, "optimizer.pt")))
            print(f"   ✅ Resumed from TinyStories checkpoint at step {final_step}")
    
    # Load replay buffer if exists
    buffer_path = os.path.join(CONFIG["output_dir"], "replay_buffer_tinystories")
    if os.path.exists(buffer_path):
        try:
            replay_buffer = ReplayBuffer(max_size=10000, task_name="tinystories")
            replay_buffer.load(buffer_path)
            print(f"   ✅ Loaded replay buffer: {len(replay_buffer)} samples")
        except Exception as e:
            print(f"   ⚠️ Could not load replay buffer: {e}")
    
    # --- NEW: Check for TinyStories Restorepoint to skip Phase 2 ---
    skip_tinystories_training = False
    if resume_from_phase < 3 and optimizer is None:
        # Import shared utilities
        from hf_utils import check_for_tinystories_restorepoint
        restorepoint_path = check_for_tinystories_restorepoint(CONFIG["output_dir"], ensure_checkpoint_restored)
        if restorepoint_path:
            print(f"   🔄 Using TinyStories restorepoint: {restorepoint_path}")
            model.load_state_dict(torch.load(os.path.join(restorepoint_path, "weights.pt")))
            model = model.to(DEVICE)
            skip_tinystories_training = True
            resume_from_phase = 3  # Skip to Phase 4
            print("   ✅ Pre-trained TinyStories loaded. Skipping Phase 2.")
    
    if optimizer is None:
        model = model.to(DEVICE)
    
    param_count = count_parameters(model)
    print(f"   d_model: {model.d_model}")
    print(f"   n_layer: {CONFIG['n_layer']}")
    print(f"   n_head: {model.layers[0].n_head}")
    print(f"   Parameters: {param_count:,}")
    
    experiment_log["phases"]["1_create"] = {
        "params": param_count,
        "d_model": model.d_model,
    }
    
    logger.log_event("CREATED", {"params": param_count}, step=0)
    
    # Save Step-0 Checkpoint (MILESTONE: Init - Weights Only)
    save_checkpoint(model, optimizer, # Optimizer is None here potentially? No, Phase 1 sets to None.
                   os.path.join(CONFIG["output_dir"], "milestone_step0_init"),
                   0, {"phase": "init"},
                   save_optimizer=False, is_rolling=False)
    
    # ========================================================================
    # PHASE 2: Train TinyStories (skip if already completed or restorepoint loaded)
    # ========================================================================
    if not skip_tinystories_training and resume_from_phase < 3:
        print("\n" + "=" * 70)
        print("📖 PHASE 2: Training on TinyStories")
        print("=" * 70)
        
        # Load dataset
        tinystories_train = load_tinystories(
            split='train',
            max_samples=CONFIG["tinystories_max_samples"],
            seq_len=CONFIG["tinystories_seq_len"],
            batch_size=CONFIG["tinystories_batch_size"],
            tokenizer_name='gpt2',
            vocab_size=CONFIG["vocab_size"]
        )
        
        # Create optimizer if not loaded from checkpoint
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["tinystories_lr"], weight_decay=0.01)
        
        # Train
        phase2_start = time.time()
        
        def checkpoint_fn(step):
            # Upload to HF only at hf_upload_interval, but always save locally
            should_upload = (step % CONFIG["hf_upload_interval"] == 0)
            save_checkpoint(model, optimizer, 
                           os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                           step, {"phase": "tinystories"}, 
                           save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
            save_resume_state(CONFIG["output_dir"], 2, step)
            
            # Push logs to Git at git_backup_interval
            if step % CONFIG["git_backup_interval"] == 0:
                git_backup_logs(CONFIG["output_dir"], step, f"TinyStories training logs step {step}")
        
        final_step = train_dataset(
            model=model,
            dataloader=tinystories_train,
            epochs=CONFIG["tinystories_epochs"],
            optimizer=optimizer,
            logger=logger,
            start_step=final_step,
            checkpoint_fn=checkpoint_fn,
            checkpoint_interval=CONFIG["local_checkpoint_interval"],  # Local restorepoint every 1000
            replay_buffer=replay_buffer,
            replay_ratio=0.0,  # No replay on first task
            task_name="tinystories",
            auto_populate_buffer=True
        )
        
        phase2_time = time.time() - phase2_start
        
        # Save checkpoint
        # Save checkpoint (MILESTONE: Weights Only, Permanent)
        save_checkpoint(model, optimizer,
                       os.path.join(CONFIG["output_dir"], "milestone_tinystories"),
                       final_step, {"phase": "tinystories_complete"},
                       save_optimizer=False, is_rolling=False)
        
        # Save replay buffer
        replay_buffer.save(os.path.join(CONFIG["output_dir"], "replay_buffer_tinystories"))
        
        # Save resume state
        save_resume_state(CONFIG["output_dir"], 3, final_step)
        
        experiment_log["phases"]["2_tinystories"] = {
            "steps": final_step,
            "time_seconds": phase2_time,
            "buffer_size": len(replay_buffer),
        }
    else:
        print("\n⏭️ Skipping Phase 2 (already completed)")
        phase2_time = 0

    
    # ========================================================================
    # PHASE 3: Evaluate TinyStories Baseline
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 3: Evaluating TinyStories Baseline")
    print("=" * 70)
    
    tinystories_val = load_tinystories(
        split='validation',
        max_samples=5000,  # Limit for speed
        seq_len=CONFIG["tinystories_seq_len"],
        batch_size=CONFIG["tinystories_batch_size"],
        tokenizer_name='gpt2',
        vocab_size=CONFIG["vocab_size"],
        shuffle=False
    )
    
    ts_baseline_ppl, ts_baseline_loss = compute_perplexity(model, tinystories_val)
    print(f"   TinyStories Baseline Perplexity: {ts_baseline_ppl:.2f}")
    print(f"   TinyStories Baseline Loss: {ts_baseline_loss:.4f}")
    
    experiment_log["phases"]["3_eval_baseline"] = {
        "tinystories_ppl": ts_baseline_ppl,
        "tinystories_loss": ts_baseline_loss,
    }
    
    # ========================================================================
    # PHASE 4: Expand Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 PHASE 4: Expanding Model for GSM8K")
    print("=" * 70)
    
    old_d_model = model.d_model
    old_params = count_parameters(model)
    
    # Expand!
    model.expand_model(
        new_input_dim=CONFIG["gsm8k_d_model"],
        new_output_dim=CONFIG["vocab_size"],
        router_type='bigram',
        use_gradient_rescue=True,
        use_orthogonal_init=False,
        isolate_cross_terms=False,
        router_init_bias=0.0,  # Open gates for Directed Synergy
        gating_threshold=0.0
    )
    
    new_params = count_parameters(model)
    print(f"   d_model: {old_d_model} → {model.d_model}")
    print(f"   Parameters: {old_params:,} → {new_params:,} (+{new_params-old_params:,})")
    
    experiment_log["phases"]["4_expand"] = {
        "old_d_model": old_d_model,
        "new_d_model": model.d_model,
        "old_params": old_params,
        "new_params": new_params,
        "added_params": new_params - old_params,
    }
    
    # Recreate optimizer with new parameters
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["gsm8k_lr"], weight_decay=0.01)

    # Save checkpoint (MILESTONE: Expansion Init - Weights Only)
    save_checkpoint(model, optimizer,
                   os.path.join(CONFIG["output_dir"], "milestone_expanded_init"),
                   final_step, {"phase": "expansion_complete"},
                   save_optimizer=False, is_rolling=False)
    
    # ========================================================================
    # PHASE 5: Train GSM8K
    # ========================================================================
    print("\n" + "=" * 70)
    print("🧮 PHASE 5: Training on GSM8K (with Asymmetric Replay)")
    print("=" * 70)
    
    gsm8k_train = load_gsm8k(
        split='train',
        max_samples=CONFIG["gsm8k_max_samples"],
        seq_len=CONFIG["gsm8k_seq_len"],
        batch_size=CONFIG["gsm8k_batch_size"],
        tokenizer_name='gpt2',
        vocab_size=CONFIG["vocab_size"]
    )
    
    phase5_start = time.time()
    
    def checkpoint_fn_gsm8k(step):
        # Upload to HF only at hf_upload_interval, but always save locally
        should_upload = (step % CONFIG["hf_upload_interval"] == 0)
        save_checkpoint(model, optimizer,
                       os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                       step, {"phase": "gsm8k"},
                       save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
        save_resume_state(CONFIG["output_dir"], 5, step)
        
        # Push logs to Git at git_backup_interval
        if step % CONFIG["git_backup_interval"] == 0:
            git_backup_logs(CONFIG["output_dir"], step, f"GSM8K training logs step {step}")
    
    final_step = train_dataset(
        model=model,
        dataloader=gsm8k_train,
        epochs=CONFIG["gsm8k_epochs"],
        optimizer=optimizer,
        logger=logger,
        start_step=final_step,
        checkpoint_fn=checkpoint_fn_gsm8k,
        checkpoint_interval=CONFIG["local_checkpoint_interval"],  # Local restorepoint every 1000
        replay_buffer=replay_buffer,
        replay_ratio=CONFIG["gsm8k_replay_ratio"],
        task_name="gsm8k",
        auto_populate_buffer=False  # Don't add GSM8K to TinyStories buffer
    )
    
    phase5_time = time.time() - phase5_start
    
    # Save final checkpoint (MILESTONE: GSM8K Final - Weights Only)
    save_checkpoint(model, optimizer,
                   os.path.join(CONFIG["output_dir"], "milestone_gsm8k_final"),
                   final_step, {"phase": "gsm8k_complete"},
                   save_optimizer=False, is_rolling=False)
    
    experiment_log["phases"]["5_gsm8k"] = {
        "steps": final_step,
        "time_seconds": phase5_time,
        "replay_ratio": CONFIG["gsm8k_replay_ratio"],
    }
    
    # ========================================================================
    # PHASE 10: Cross-Verification (Moon Landing Requirement)
    # ========================================================================
    print("\n" + "=" * 70)
    print("🧪 PHASE 10: Cross-Verification (Moon Landing)")
    print("=" * 70)
    
    verification_results = {}
    
    # Datasets for verification
    datasets = {
        "TinyStories": tinystories_val,  # Assuming this is defined/loaded
        # We need to reload/ensure GSM8K test set is available correctly
        "GSM8K": None # Will load inside loop or check availability
    }
    
    # We need to verify TinyStories Val is available (it was loaded in Phase 3/6 logic?)
    # Phase 2 defined `tinystories_train`, Phase 3/6 logic usually defines val.
    # Let's verify and load if needed.
    
    # 1. Load Validation Data
    try:
        ts_val = load_tinystories(split='validation', max_samples=100, seq_len=CONFIG["tinystories_seq_len"], batch_size=CONFIG["tinystories_batch_size"], tokenizer_name='gpt2', vocab_size=CONFIG["vocab_size"])
        gsm8k_test = load_gsm8k(split='test', max_samples=100, seq_len=CONFIG["gsm8k_seq_len"], batch_size=CONFIG["gsm8k_batch_size"], tokenizer_name='gpt2', vocab_size=CONFIG["vocab_size"])
        datasets["TinyStories"] = ts_val
        datasets["GSM8K"] = gsm8k_test
    except Exception as e:
        print(f"⚠️ Could not load verification datasets: {e}")
    
    # Models to verify
    checkpoints_to_verify = [
        ("milestone_tinystories", CONFIG["tinystories_d_model"]), 
        ("milestone_gsm8k_final", CONFIG["gsm8k_d_model"])
    ]
    
    for ckpt_name, d_model_ckpt in checkpoints_to_verify:
        ckpt_path = os.path.join(CONFIG["output_dir"], ckpt_name)
        if not ensure_checkpoint_restored(ckpt_path):
            print(f"⚠️ Checkpoint {ckpt_name} not found (or failed restore), skipping verification.")
            continue
            
        print(f"\n🔍 Verifying Model: {ckpt_name}")
        verification_results[ckpt_name] = {}
        
        # Load Model
        try:
            # We need to re-init model with correct size
            # TinyStories model is small (d_model=384), GSM8K is expanded (d_model=1024)
            # We must be careful about model instantiation
            if "tinystories" in ckpt_name:
                 # Small model params
                 v_model = DGESimpleTransformer(
                    vocab_size=CONFIG["vocab_size"],
                    d_model=CONFIG["tinystories_d_model"],
                    n_layer=CONFIG["n_layer"],
                    n_head=CONFIG["tinystories_n_head"],
                    max_seq_len=CONFIG["max_seq_len"]
                 )
            else:
                 # Expanded model params (approximate logic, assuming GSM8K is expanded)
                 v_model = DGESimpleTransformer(
                     # ... this is tricky because expanded model has 1024 d_model
                     # and expansion logic.
                     # Better to instantiate small and EXPAND if needed?
                     vocab_size=CONFIG["vocab_size"],
                     d_model=CONFIG["tinystories_d_model"], # Start small
                     n_layer=CONFIG["n_layer"],
                     n_head=CONFIG["tinystories_n_head"],
                     max_seq_len=CONFIG["max_seq_len"]
                 )
                 # Expand it
                 v_model.expand_model(
                     new_input_dim=CONFIG["gsm8k_d_model"],
                     new_output_dim=CONFIG["vocab_size"],
                     router_type='bigram', use_gradient_rescue=True,
                     router_init_bias=0.0, gating_threshold=0.0
                 )
            
            v_model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
            v_model = v_model.to(DEVICE)
            v_model.eval()
            
            for ds_name, ds_loader in datasets.items():
                if ds_loader:
                    print(f"   Evaluating on {ds_name}...")
                    ppl, loss = compute_perplexity(v_model, ds_loader, max_batches=50)
                    print(f"   -> PPL: {ppl:.2f}, Loss: {loss:.4f}")
                    verification_results[ckpt_name][ds_name] = {"ppl": ppl, "loss": loss}
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            verification_results[ckpt_name]["error"] = str(e)

    experiment_log["verification"] = verification_results

    
    # ========================================================================
    # PHASE 6: Evaluate TinyStories Retention
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 6: Evaluating TinyStories Retention")
    print("=" * 70)
    
    ts_retention_ppl, ts_retention_loss = compute_perplexity(model, tinystories_val)
    retention_ratio = ts_baseline_ppl / ts_retention_ppl if ts_retention_ppl > 0 else 0
    
    print(f"   TinyStories Baseline: {ts_baseline_ppl:.2f}")
    print(f"   TinyStories After GSM8K: {ts_retention_ppl:.2f}")
    print(f"   Retention Ratio: {retention_ratio:.2%}")
    
    experiment_log["phases"]["6_retention"] = {
        "baseline_ppl": ts_baseline_ppl,
        "retention_ppl": ts_retention_ppl,
        "retention_ratio": retention_ratio,
    }
    
    # ========================================================================
    # PHASE 7: Evaluate GSM8K
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 7: Evaluating GSM8K Performance")
    print("=" * 70)
    
    gsm8k_test = load_gsm8k(
        split='test',
        max_samples=None,  # All test samples (~1.3K)
        seq_len=CONFIG["gsm8k_seq_len"],
        batch_size=CONFIG["gsm8k_batch_size"],
        tokenizer_name='gpt2',
        vocab_size=CONFIG["vocab_size"],
        shuffle=False
    )
    
    gsm8k_ppl, gsm8k_loss = compute_perplexity(model, gsm8k_test, max_batches=50)
    print(f"   GSM8K Perplexity: {gsm8k_ppl:.2f}")
    print(f"   GSM8K Loss: {gsm8k_loss:.4f}")
    
    experiment_log["phases"]["7_gsm8k_eval"] = {
        "ppl": gsm8k_ppl,
        "loss": gsm8k_loss,
    }
    
    # ========================================================================
    # PHASE 8: Save Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("💾 PHASE 8: Saving Experiment Results")
    print("=" * 70)
    
    experiment_log["end_time"] = datetime.now().isoformat()
    experiment_log["total_time_seconds"] = phase2_time + phase5_time
    
    # Summary metrics for paper
    experiment_log["summary"] = {
        "task_a_baseline_ppl": ts_baseline_ppl,
        "task_a_retention_ppl": ts_retention_ppl,
        "task_a_retention_ratio": retention_ratio,
        "task_b_ppl": gsm8k_ppl,
        "model_params_initial": old_params,
        "model_params_final": new_params,
        "expansion_added_params": new_params - old_params,
        "total_training_time_hours": (phase2_time + phase5_time) / 3600,
    }
    
    # Save JSON
    results_path = os.path.join(CONFIG["output_dir"], "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"   Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"   TinyStories Baseline PPL: {ts_baseline_ppl:.2f}")
    print(f"   TinyStories Retention PPL: {ts_retention_ppl:.2f}")
    print(f"   Retention Ratio: {retention_ratio:.2%}")
    print(f"   GSM8K PPL: {gsm8k_ppl:.2f}")
    print(f"   Total Time: {(phase2_time + phase5_time) / 3600:.2f} hours")
    print("=" * 70)
    
    # ========================================================================
    # PHASE 9: Upload to HuggingFace Hub
    # ========================================================================
    print("\n" + "=" * 70)
    print("☁️ PHASE 9: Uploading to HuggingFace Hub")
    print("=" * 70)
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = "darealSven/dge-tinystories-gsm8k"
    
    if hf_token:
        try:
            from huggingface_hub import HfApi, create_repo
            
            api = HfApi(token=hf_token)
            
            # Create repo if it doesn't exist (private by default)
            try:
                create_repo(repo_id=hf_repo, token=hf_token, private=True, exist_ok=True)
                print(f"   📦 Repository: {hf_repo}")
            except Exception as e:
                print(f"   ⚠️ Repo creation note: {e}")
            
            # Upload entire output directory
            print(f"   📤 Uploading {CONFIG['output_dir']}...")
            api.upload_folder(
                folder_path=CONFIG["output_dir"],
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"DGE Experiment Results - {datetime.now().isoformat()}"
            )
            
            print(f"   ✅ Uploaded to: https://huggingface.co/{hf_repo}")
            experiment_log["hf_upload"] = {"status": "success", "repo": hf_repo}
            
        except ImportError:
            print("   ⚠️ huggingface_hub not installed. Skipping upload.")
            print("      Install with: pip install huggingface_hub")
            experiment_log["hf_upload"] = {"status": "skipped", "reason": "huggingface_hub not installed"}
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            experiment_log["hf_upload"] = {"status": "failed", "error": str(e)}
    else:
        print("   ⚠️ HF_TOKEN not set. Skipping HuggingFace upload.")
        print("      Set with: export HF_TOKEN=hf_your_token")
        experiment_log["hf_upload"] = {"status": "skipped", "reason": "HF_TOKEN not set"}
    
    # Save final log with upload status
    with open(results_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 ALL PHASES COMPLETE!")
    print("=" * 70)
    print(f"\n📥 Download results with:")
    print(f"   huggingface-cli download {hf_repo} --local-dir ./results")
    
    return experiment_log


# ============================================================================
# RUNPOD AUTO-STOP (PAUSE)
# ============================================================================

def stop_runpod():
    """Stop (Pause) the RunPod pod to save compute costs but KEEP data. Uses 'stop'."""
    import subprocess
    
    pod_id = os.environ.get('RUNPOD_POD_ID')
    
    if pod_id:
        print("\n" + "=" * 70)
        print("⏸️ STOPPING (PAUSING) RUNPOD POD")
        print("=" * 70)
        print(f"   Pod ID: {pod_id}")
        print("   Action: STOP (Pauses billing, keeps data on disk)")
        
        try:
            # Try 'stop' command
            result = subprocess.run(
                ["runpodctl", "stop", "pod", pod_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print("   ✅ Pod stop initiated! Goodbye.")
            else:
                print(f"   ⚠️ Command returned: {result.stderr}")
                print("   ⚠️ Please manually STOP the pod to avoid charges.")
        except FileNotFoundError:
            print("   ⚠️ runpodctl not found. Pod will continue running.")
        except Exception as e:
            print(f"   ❌ Failed to stop: {e}")
            print("   ⚠️ Please manually STOP the pod to avoid charges.")
    else:
        print("\n💻 Running locally (no RUNPOD_POD_ID). Skipping pod stop.")


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Wait for pending HF uploads
        shutdown_upload_worker()
        print("\n🏁 Experiment Script Completed.")
        
        # STOP the pod automatically to save money (but keep data)
        stop_runpod()
