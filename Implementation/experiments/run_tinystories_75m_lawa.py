#!/usr/bin/env python3
"""
TinyStories 75M Training Script (LAWA Experimental)
====================================================
Train a 75M parameter model on TinyStories with LAWA checkpoint averaging.

LAWA (LAtest Weight Averaging):
- Based on "Early Weight Averaging meets High Learning Rates" (ICLR 2024)
- Maintains sliding window of last K checkpoints
- Averages weights for improved generalization

Architecture:
- d_model: 768
- n_layer: 12
- n_head: 12
- vocab_size: 50257 (GPT-2)
- ~75M parameters

Training Target:
- Loss < 2.0 (coherent generation)
- ~100K steps
- ~3B tokens

Estimated Cost: $50-70 on L40S spot
Estimated Time: ~60-80 hours
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import DGESimpleTransformer
from data.loader import load_tinystories
from utils.model_manager import ModelManager, Diary
from utils.checkpoint_averaging import CheckpointAverager, create_lawa
from cloud.gdrive_sync import GDriveSync
from hf.repo_manager import HFRepoManager

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Model Architecture (75M params)
    "vocab_size": 50257,
    "d_model": 768,
    "n_layer": 12,
    "n_head": 12,
    "max_seq_len": 1024,
    
    # Training
    
    # Training
    "batch_size": 16, # Reduced from 32 for safety on 48GB cards (OOM protection)
    "gradient_accumulation_steps": 2, # x16 = 32 effective batch size
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "max_steps": 100000,
    "eval_interval": 1000,
    "save_interval": 1000,  # Save every 1000 steps (~10 min) for spot pod safety!
    "log_interval": 100,
    
    # Data
    "dataset": "roneneldan/TinyStories",
    "split": "train",
    
    # Targets
    "target_loss": 2.0,  # Stop early if reached
    
    # Naming - V 0.2.0: Per-model HF repo
    "model_name": "tinystories-75m",  # Used for HF repo: darealSven/dge-tinystories-75m
    "family_name": "tinystories_75m",
    "stage_name": "full_training_lawa",  # Separate stage for LAWA experiment
    
    # LAWA Configuration
    "use_lawa": True,              # Enable LAWA checkpoint averaging
    "lawa_window": 5,              # Number of checkpoints to average
    "lawa_start_step": 2000,       # Start averaging after warmup
    "lawa_save_averaged": True,    # Save averaged weights alongside raw
    
    # Google Drive Configuration
    "use_gdrive": True,            # GDrive sync ENABLED
    "gdrive_remote": "gdrive",     # rclone remote name
    "gdrive_path": "DGE/checkpoints",  # Path in GDrive
}

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def setup_ddp():
    """
    Initialize DistributedDataParallel if launched via torchrun.
    
    Returns:
        (rank, local_rank, world_size, is_ddp)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched via torchrun
        dist.init_process_group("nccl")
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        # Single GPU fallback
        return 0, 0, 1, False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║      TinyStories 75M - LAWA Experimental  (V 0.19.0-exp)         ║
║                                                                  ║
║   Architecture: 768d × 12L × 12H = ~75M params                   ║
║   LAWA: Latest Weight Averaging (window=5)                       ║
║   Target: Loss < 2.0 (coherent generation)                       ║
╚══════════════════════════════════════════════════════════════════╝
""", flush=True)
    
    # Setup DDP (if multi-GPU via torchrun)
    rank, local_rank, world_size, is_ddp = setup_ddp()
    
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0:
            print(f"📡 DDP Mode: {world_size} GPUs, Rank {rank}", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📡 Device: {device}")
    
    if device.type == "cpu":
        print("⚠️  WARNING: Training on CPU will be extremely slow!")
        print("   Please run on a GPU instance.")
    
    # Initialize managers (only on rank 0 for logging)
    mgr = ModelManager()
    hf_mgr = HFRepoManager(CONFIG["model_name"])  # V 0.2.0: Per-model repo
    
    # Create model directory
    family_path = os.path.join("models", CONFIG["family_name"])
    stage_path = os.path.join(family_path, CONFIG["stage_name"])
    os.makedirs(stage_path, exist_ok=True)
    
    # Initialize diary (only rank 0 writes)
    diary = Diary(stage_path)
    if rank == 0:
        diary.log("TRAINING_START", "TinyStories 75M training initiated", CONFIG)
    
    # ========================================================================
    # BUILD MODEL
    # ========================================================================
    if rank == 0:
        print("\n🔨 Building 75M model...")
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"   Total params: {total_params / 1e6:.1f}M")
        print(f"   Trainable: {trainable_params / 1e6:.1f}M")
    
    model = model.to(device)
    
    # Multi-GPU support with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"   🚀 Using {world_size} GPUs with DistributedDataParallel!")
    
    # ========================================================================
    # INITIALIZE LAWA AND GDRIVE (if enabled)
    # ========================================================================
    lawa_mgr = None
    gdrive_sync = None
    
    if CONFIG["use_lawa"] and rank == 0:
        lawa_mgr = create_lawa(window_size=CONFIG["lawa_window"])
        print(f"\n📊 LAWA enabled: window={CONFIG['lawa_window']}, start_step={CONFIG['lawa_start_step']}")
        diary.log("LAWA_INIT", "LAWA checkpoint averaging initialized", {
            "window_size": CONFIG["lawa_window"],
            "start_step": CONFIG["lawa_start_step"]
        })
    
    if CONFIG["use_gdrive"] and rank == 0:
        gdrive_sync = GDriveSync(
            remote_name=CONFIG["gdrive_remote"],
            remote_path=CONFIG["gdrive_path"],
            model_name=CONFIG["model_name"]
        )
        if gdrive_sync.is_available():
            print(f"☁️ Google Drive sync enabled: {gdrive_sync.full_remote}")
        else:
            print("⚠️ Google Drive not available (rclone not configured)")
            gdrive_sync = None
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (if available)
    # ========================================================================
    resume_step = 0
    checkpoint_path = os.path.join(stage_path, "weights.pt")
    
    # Try to download checkpoint from HuggingFace
    if rank == 0:
        print("\n🔍 Checking for existing checkpoint on HuggingFace...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        
        # Download weights
        hf_weights = hf_hub_download(
            repo_id=hf_mgr.repo_id,
            filename=f"{CONFIG['family_name']}/{CONFIG['stage_name']}/weights.pt",
            token=os.environ.get("HF_TOKEN")
        )
        shutil.copy(hf_weights, checkpoint_path)
        if rank == 0:
            print(f"   ✅ Downloaded checkpoint from HuggingFace", flush=True)
        
        # Download diary files (for resume step extraction)
        try:
            hf_diary_md = hf_hub_download(
                repo_id=hf_mgr.repo_id,
                filename=f"{CONFIG['family_name']}/{CONFIG['stage_name']}/diary.md",
                token=os.environ.get("HF_TOKEN")
            )
            shutil.copy(hf_diary_md, os.path.join(stage_path, "diary.md"))
            if rank == 0:
                print(f"   ✅ Downloaded diary.md from HuggingFace", flush=True)
        except Exception:
            pass  # diary.md might not exist
        
        try:
            hf_diary_jsonl = hf_hub_download(
                repo_id=hf_mgr.repo_id,
                filename=f"{CONFIG['family_name']}/{CONFIG['stage_name']}/diary.jsonl",
                token=os.environ.get("HF_TOKEN")
            )
            shutil.copy(hf_diary_jsonl, os.path.join(stage_path, "diary.jsonl"))
            if rank == 0:
                print(f"   ✅ Downloaded diary.jsonl from HuggingFace", flush=True)
        except Exception:
            pass  # diary.jsonl might not exist
            
    except Exception as e:
        if rank == 0:
            print(f"   ℹ️ No HF checkpoint found: {e}", flush=True)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"   📂 Loading checkpoint: {checkpoint_path}", flush=True)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            # Handle DataParallel wrapper
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print(f"   ✅ Checkpoint loaded successfully", flush=True)
            
            # Try to extract step from diary
            diary_path = os.path.join(stage_path, "diary.md")
            if os.path.exists(diary_path):
                with open(diary_path, 'r') as f:
                    content = f.read()
                    # Find last CHECKPOINT entry
                    import re
                    matches = re.findall(r'Saved at step (\d+)', content)
                    if matches:
                        resume_step = int(matches[-1])
                        print(f"   🔄 Resuming from step {resume_step}", flush=True)
        except Exception as e:
            print(f"   ⚠️ Could not load checkpoint: {e}", flush=True)
    else:
        print(f"   🆕 Starting fresh training (no checkpoint found)", flush=True)
    
    # Save config
    config_path = os.path.join(stage_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    if rank == 0:
        print("\n📚 Loading TinyStories dataset...")
    
    # Load the underlying dataset first
    from data.loader import load_tinystories, TextDataset
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Load raw dataset
    dataset = load_dataset("roneneldan/TinyStories", split=CONFIG["split"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract texts and create torch dataset
    texts = [item['text'] for item in dataset]
    torch_dataset = TextDataset(texts, tokenizer, CONFIG["max_seq_len"], CONFIG["vocab_size"])
    
    # Create sampler for DDP
    if is_ddp:
        train_sampler = DistributedSampler(torch_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        torch_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"✅ Loaded {len(torch_dataset)} samples, {len(train_loader)} batches (per GPU)")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n🚀 Starting training...", flush=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / max(1, CONFIG["warmup_steps"])
        return 1.0  # Constant after warmup
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_time = time.time()
    global_step = resume_step  # Resume from checkpoint step
    running_loss = 0.0
    best_loss = float('inf')
    
    model.train()
    
    try:
        while global_step < CONFIG["max_steps"]:
            for batch in train_loader:
                if global_step >= CONFIG["max_steps"]:
                    break
                
                # Handle batch format
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    # TinyStories loader returns full sequence, we need to shift for labels
                    # If "labels" key exists, use it (e.g., from HF datasets with labels)
                    if "labels" in batch:
                        labels = batch["labels"].to(device)
                    else: 
                        # Create shifted labels if not provided
                        labels = input_ids.clone()
                else:
                    raise ValueError(f"Unknown batch type: {type(batch)}")
                
                # Prepare inputs and targets for the model
                # inputs are tokens 0 to N-1
                # targets are tokens 1 to N
                inputs = input_ids[:, :-1].contiguous()
                targets = labels[:, 1:].contiguous() # Use the shifted labels
                
                # Forward
                optimizer.zero_grad()
                logits, loss = model(inputs, targets)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                global_step += 1
                
                # Logging (only rank 0)
                if global_step % CONFIG["log_interval"] == 0:
                    avg_loss = running_loss / CONFIG["log_interval"]
                    elapsed = time.time() - start_time
                    tokens_seen = global_step * CONFIG["batch_size"] * CONFIG["max_seq_len"] * world_size
                    tokens_per_sec = tokens_seen / elapsed
                    
                    # ETA calculation
                    remaining_steps = CONFIG["max_steps"] - global_step
                    eta_seconds = remaining_steps / (global_step / elapsed) if global_step > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    if rank == 0:
                        print(f"Step {global_step:6d} | Loss: {avg_loss:.4f} | "
                              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                              f"Tokens: {tokens_seen/1e9:.2f}B | "
                              f"Speed: {tokens_per_sec/1000:.1f}K tok/s | "
                              f"ETA: {eta_hours:.1f}h", flush=True)
                    
                    running_loss = 0.0
                    
                    # Check target loss
                    if avg_loss < CONFIG["target_loss"] and rank == 0:
                        print(f"\n🎯 Target loss {CONFIG['target_loss']} reached!")
                
                # Evaluation (only rank 0)
                if global_step % CONFIG["eval_interval"] == 0 and rank == 0:
                    diary.log("EVAL", f"Step {global_step}", {
                        "step": global_step,
                        "loss": avg_loss,
                        "tokens_seen": tokens_seen,
                        "elapsed_hours": elapsed / 3600
                    })
                
                # Save checkpoint (only rank 0)
                if global_step % CONFIG["save_interval"] == 0 and rank == 0:
                    # For DDP, save the unwrapped model
                    model_to_save = model.module if hasattr(model, 'module') else model
                    raw_state_dict = model_to_save.state_dict()
                    
                    # Save raw checkpoint
                    checkpoint_path = os.path.join(stage_path, "weights.pt")
                    torch.save(raw_state_dict, checkpoint_path)
                    
                    # LAWA: Update buffer and optionally save averaged weights
                    if lawa_mgr is not None:
                        lawa_mgr.update(raw_state_dict)
                        
                        if global_step >= CONFIG["lawa_start_step"] and lawa_mgr.is_ready():
                            averaged_state = lawa_mgr.get_averaged()
                            
                            if averaged_state and CONFIG["lawa_save_averaged"]:
                                lawa_path = os.path.join(stage_path, "weights_lawa.pt")
                                torch.save(averaged_state, lawa_path)
                                print(f"   📊 LAWA: Saved averaged weights (buffer={lawa_mgr.get_buffer_size()})", flush=True)
                                
                                diary.log("LAWA_CHECKPOINT", f"LAWA average at step {global_step}", {
                                    "step": global_step,
                                    "buffer_size": lawa_mgr.get_buffer_size(),
                                    "loss": avg_loss
                                })
                    
                    diary.log("CHECKPOINT", f"Saved at step {global_step}", {
                        "step": global_step,
                        "loss": avg_loss
                    })
                    
                    # Upload to HF
                    try:
                        hf_mgr.upload_directory(
                            stage_path,
                            f"{CONFIG['family_name']}/{CONFIG['stage_name']}"
                        )
                        print(f"   ☁️ Uploaded to HuggingFace", flush=True)
                    except Exception as e:
                        print(f"   ⚠️ HF upload failed: {e}")
                    
                    # Sync to Google Drive (if enabled)
                    if gdrive_sync is not None:
                        try:
                            gdrive_sync.upload_directory(stage_path)
                            print(f"   ☁️ Synced to Google Drive", flush=True)
                        except Exception as e:
                            print(f"   ⚠️ GDrive sync failed: {e}")
                    
                    # Track best
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = os.path.join(stage_path, "best_weights.pt")
                        torch.save(raw_state_dict, best_path)
                        print(f"   🏆 New best loss: {best_loss:.4f}")
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\n\n⚠️ Training interrupted!")
    
    # ========================================================================
    # FINAL SAVE (only rank 0)
    # ========================================================================
    if rank == 0:
        print("\n💾 Saving final model...")
        
        model_to_save = model.module if hasattr(model, 'module') else model
        final_path = os.path.join(stage_path, "weights.pt")
        torch.save(model_to_save.state_dict(), final_path)
        
        elapsed_total = time.time() - start_time
        final_tokens = global_step * CONFIG["batch_size"] * CONFIG["max_seq_len"] * world_size
        
        diary.log("TRAINING_COMPLETE", "Training finished", {
            "final_step": global_step,
            "final_loss": avg_loss if 'avg_loss' in dir() else None,
            "best_loss": best_loss,
            "tokens_seen": final_tokens,
            "elapsed_hours": elapsed_total / 3600
        })
        
        # Final HF upload
        print("\n☁️ Uploading to HuggingFace...")
        try:
            hf_mgr.upload_directory(
                stage_path,
                f"{CONFIG['family_name']}/{CONFIG['stage_name']}"
            )
            print(f"✅ Uploaded to {CONFIG['hf_repo']}/{CONFIG['family_name']}/{CONFIG['stage_name']}")
        except Exception as e:
            print(f"❌ HF upload failed: {e}")
        
        # ========================================================================
        # SUMMARY
        # ========================================================================
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE                             ║
╠══════════════════════════════════════════════════════════════════╣
║   Steps: {global_step:,}
║   Tokens: {final_tokens/1e9:.2f}B
║   Time: {elapsed_total/3600:.1f} hours
║   Best Loss: {best_loss:.4f}
║   
║   Model saved: {final_path}
║   Diary: {stage_path}/diary.md
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Test generation
        print("\n🧪 Test Generation:")
        model.eval()
        prompt = "Once upon a time"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = input_ids
            for _ in range(50):
                logits, _ = model(generated[:, -CONFIG["max_seq_len"]:])
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"   Prompt: '{prompt}'")
        print(f"   Output: '{output}'")
        
        print("\n✅ Done!")
    
    # DDP cleanup
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
