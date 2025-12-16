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

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model Architecture (fixed for entire experiment)
    "vocab_size": 50257,      # GPT-2 tokenizer
    "n_layer": 12,            # Cannot expand - fixed at 12
    "max_seq_len": 1024,      # Cannot expand - fixed at 1024
    
    # TinyStories Phase
    "tinystories_d_model": 384,
    "tinystories_n_head": 6,  # head_dim = 64
    "tinystories_epochs": 1,
    "tinystories_batch_size": 16,   # Reduced from 64 to avoid OOM
    "tinystories_seq_len": 256,     # Reduced from 512 to avoid OOM
    "tinystories_max_samples": None,  # None = all (~2M)
    "tinystories_lr": 1e-4,
    
    # Expansion
    "expansion_delta": 640,   # 384 -> 1024
    
    # GSM8K Phase
    "gsm8k_d_model": 1024,    # After expansion
    "gsm8k_n_head": 16,       # head_dim = 64
    "gsm8k_epochs": 3,
    "gsm8k_batch_size": 8,    # Reduced from 32 (expanded model is 915M params)
    "gsm8k_seq_len": 256,     # Reduced from 512 to avoid OOM
    "gsm8k_max_samples": None,  # None = all (~7.5K)
    "gsm8k_lr": 5e-5,
    "gsm8k_replay_ratio": 0.1,  # 10% replay from TinyStories
    
    # Paths
    "output_dir": "models/tinystories_gsm8k_chain",
    "checkpoint_interval": 5000,
}

# HuggingFace Hub configuration
HF_REPO = "darealSven/dge-tinystories-gsm8k"

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


# ============================================================================
# BACKGROUND HF UPLOAD
# ============================================================================

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
            print(f"☁️ [Background] Uploading {folder_path} to HF Hub...")
            
            try:
                api.upload_folder(
                    folder_path=folder_path,
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


def save_checkpoint(model, optimizer, path, step, config):
    """Save model checkpoint with config. Deletes previous checkpoint AFTER successful save."""
    import shutil
    
    # Get checkpoint category (e.g., "tinystories_checkpoint", "gsm8k_checkpoint")
    checkpoint_key = os.path.basename(path)
    prev_path = _previous_checkpoints.get(checkpoint_key)
    
    # === SAVE FIRST (before any deletion) ===
    os.makedirs(path, exist_ok=True)
    
    try:
        # Save weights
        weights_path = os.path.join(path, "weights.pt")
        torch.save(model.state_dict(), weights_path)
        
        # Save optimizer state
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
        
        # Verify files exist
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise IOError("Checkpoint files were not saved correctly")
        
        print(f"💾 Checkpoint saved: {path}")
        save_successful = True
        
    except Exception as e:
        print(f"❌ Checkpoint save FAILED: {e}")
        save_successful = False
    
    # === DELETE PREVIOUS ONLY IF SAVE SUCCEEDED ===
    if save_successful and prev_path and prev_path != path and "model_" not in checkpoint_key:
        if os.path.exists(prev_path):
            try:
                shutil.rmtree(prev_path)
                print(f"🗑️ Deleted previous checkpoint: {prev_path}")
            except Exception as e:
                print(f"⚠️ Could not delete previous checkpoint: {e}")
    
    # Track this checkpoint for next deletion
    if save_successful:
        _previous_checkpoints[checkpoint_key] = path
        
        # Queue for background upload to HuggingFace Hub
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

def run_experiment():
    """Run the full TinyStories → GSM8K experiment chain."""
    
    print("=" * 70)
    print("🔬 DGE EXPERIMENT: TinyStories → GSM8K Chain")
    print(f"   Version: {__version__}")
    print(f"   Device: {DEVICE}")
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
        # Load from GSM8K checkpoint (expanded model)
        gsm8k_ckpt = os.path.join(CONFIG["output_dir"], "gsm8k_checkpoint")
        if os.path.exists(os.path.join(gsm8k_ckpt, "weights.pt")):
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
        # Load from TinyStories checkpoint
        ts_ckpt = os.path.join(CONFIG["output_dir"], "tinystories_checkpoint")
        if os.path.exists(os.path.join(ts_ckpt, "weights.pt")):
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
            replay_buffer = ReplayBuffer.load(buffer_path)
            print(f"   ✅ Loaded replay buffer: {len(replay_buffer)} samples")
        except Exception as e:
            print(f"   ⚠️ Could not load replay buffer: {e}")
    
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
    
    # ========================================================================
    # PHASE 2: Train TinyStories (skip if already completed)
    # ========================================================================
    if resume_from_phase < 3:
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
            save_checkpoint(model, optimizer, 
                           os.path.join(CONFIG["output_dir"], "tinystories_checkpoint"),
                           step, {"phase": "tinystories"})
            save_resume_state(CONFIG["output_dir"], 2, step)
        
        final_step = train_dataset(
            model=model,
            dataloader=tinystories_train,
            epochs=CONFIG["tinystories_epochs"],
            optimizer=optimizer,
            logger=logger,
            start_step=final_step,
            checkpoint_fn=checkpoint_fn,
            checkpoint_interval=CONFIG["checkpoint_interval"],
            replay_buffer=replay_buffer,
            replay_ratio=0.0,  # No replay on first task
            task_name="tinystories",
            auto_populate_buffer=True
        )
        
        phase2_time = time.time() - phase2_start
        
        # Save checkpoint
        save_checkpoint(model, optimizer,
                       os.path.join(CONFIG["output_dir"], "model_tinystories"),
                       final_step, {"phase": "tinystories_complete"})
        
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
        save_checkpoint(model, optimizer,
                       os.path.join(CONFIG["output_dir"], "gsm8k_checkpoint"),
                       step, {"phase": "gsm8k"})
    
    final_step = train_dataset(
        model=model,
        dataloader=gsm8k_train,
        epochs=CONFIG["gsm8k_epochs"],
        optimizer=optimizer,
        logger=logger,
        start_step=final_step,
        checkpoint_fn=checkpoint_fn_gsm8k,
        checkpoint_interval=CONFIG["checkpoint_interval"],
        replay_buffer=replay_buffer,
        replay_ratio=CONFIG["gsm8k_replay_ratio"],
        task_name="gsm8k",
        auto_populate_buffer=False  # Don't add GSM8K to TinyStories buffer
    )
    
    phase5_time = time.time() - phase5_start
    
    # Save final checkpoint
    save_checkpoint(model, optimizer,
                   os.path.join(CONFIG["output_dir"], "model_gsm8k"),
                   final_step, {"phase": "gsm8k_complete"})
    
    # Save resume state
    save_resume_state(CONFIG["output_dir"], 6, final_step)
    
    experiment_log["phases"]["5_gsm8k"] = {
        "steps": final_step,
        "time_seconds": phase5_time,
        "replay_ratio": CONFIG["gsm8k_replay_ratio"],
    }
    
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
# RUNPOD AUTO-TERMINATE
# ============================================================================

def terminate_runpod():
    """Terminate the RunPod pod to avoid ongoing charges. Uses 'remove' for zero cost."""
    import subprocess
    
    pod_id = os.environ.get('RUNPOD_POD_ID')
    
    if pod_id:
        print("\n" + "=" * 70)
        print("🛑 TERMINATING RUNPOD POD")
        print("=" * 70)
        print(f"   Pod ID: {pod_id}")
        print("   Action: REMOVE (complete deletion, zero cost)")
        
        try:
            result = subprocess.run(
                ["runpodctl", "remove", "pod", pod_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print("   ✅ Pod termination initiated!")
            else:
                print(f"   ⚠️ Command returned: {result.stderr}")
        except FileNotFoundError:
            print("   ⚠️ runpodctl not found. Pod will continue running.")
        except Exception as e:
            print(f"   ❌ Failed to terminate: {e}")
    else:
        print("\n💻 Running locally (no RUNPOD_POD_ID). Skipping pod termination.")


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
        # Wait for pending HF uploads to complete
        shutdown_upload_worker()
        # ALWAYS terminate pod to avoid ongoing charges
        terminate_runpod()



