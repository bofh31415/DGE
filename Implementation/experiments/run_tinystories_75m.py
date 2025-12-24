#!/usr/bin/env python3
"""
TinyStories 75M Training Script
================================
Train a 75M parameter model on TinyStories to fluency.

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
from datetime import datetime
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import DGESimpleTransformer
from data.loader import load_tinystories
from utils.model_manager import ModelManager, Diary
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
    "stage_name": "full_training",
}

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           TinyStories 75M - Full Training Run  (V 0.17.1)        ║
║                                                                  ║
║   Architecture: 768d × 12L × 12H = ~75M params                   ║
║   Target: Loss < 2.0 (coherent generation)                       ║
║   Estimated Time: 60-80 hours                                    ║
╚══════════════════════════════════════════════════════════════════╝
""", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Device: {device}")
    
    if device.type == "cpu":
        print("⚠️  WARNING: Training on CPU will be extremely slow!")
        print("   Please run on a GPU instance.")
    
    # Initialize managers
    mgr = ModelManager()
    hf_mgr = HFRepoManager(CONFIG["model_name"])  # V 0.2.0: Per-model repo
    
    # Create model directory
    family_path = os.path.join("models", CONFIG["family_name"])
    stage_path = os.path.join(family_path, CONFIG["stage_name"])
    os.makedirs(stage_path, exist_ok=True)
    
    # Initialize diary
    diary = Diary(stage_path)
    diary.log("TRAINING_START", "TinyStories 75M training initiated", CONFIG)
    
    # ========================================================================
    # BUILD MODEL
    # ========================================================================
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
    print(f"   Total params: {total_params / 1e6:.1f}M")
    print(f"   Trainable: {trainable_params / 1e6:.1f}M")
    
    model = model.to(device)
    
    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"   🚀 Using {num_gpus} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (if available)
    # ========================================================================
    resume_step = 0
    checkpoint_path = os.path.join(stage_path, "weights.pt")
    
    # Try to download checkpoint from HuggingFace
    print("\n🔍 Checking for existing checkpoint on HuggingFace...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_weights = hf_hub_download(
            repo_id=hf_mgr.repo_id,
            filename=f"{CONFIG['family_name']}/{CONFIG['stage_name']}/weights.pt",
            token=os.environ.get("HF_TOKEN")
        )
        # Copy to local path
        import shutil
        shutil.copy(hf_weights, checkpoint_path)
        print(f"   ✅ Downloaded checkpoint from HuggingFace", flush=True)
    except Exception as e:
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
    print("\n📚 Loading TinyStories dataset...")
    train_loader = load_tinystories(
        split=CONFIG["split"],
        max_samples=None,  # Use full dataset
        seq_len=CONFIG["max_seq_len"],
        batch_size=CONFIG["batch_size"],
        tokenizer_name='gpt2',
        vocab_size=CONFIG["vocab_size"],
        shuffle=True
    )
    
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
                
                # Logging
                if global_step % CONFIG["log_interval"] == 0:
                    avg_loss = running_loss / CONFIG["log_interval"]
                    elapsed = time.time() - start_time
                    tokens_seen = global_step * CONFIG["batch_size"] * CONFIG["max_seq_len"]
                    tokens_per_sec = tokens_seen / elapsed
                    
                    # ETA calculation
                    remaining_steps = CONFIG["max_steps"] - global_step
                    eta_seconds = remaining_steps / (global_step / elapsed) if global_step > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    print(f"Step {global_step:6d} | Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Tokens: {tokens_seen/1e9:.2f}B | "
                          f"Speed: {tokens_per_sec/1000:.1f}K tok/s | "
                          f"ETA: {eta_hours:.1f}h", flush=True)
                    
                    running_loss = 0.0
                    
                    # Check target loss
                    if avg_loss < CONFIG["target_loss"]:
                        print(f"\n🎯 Target loss {CONFIG['target_loss']} reached!")
                
                # Evaluation
                if global_step % CONFIG["eval_interval"] == 0:
                    diary.log("EVAL", f"Step {global_step}", {
                        "step": global_step,
                        "loss": avg_loss,
                        "tokens_seen": tokens_seen,
                        "elapsed_hours": elapsed / 3600
                    })
                
                # Save checkpoint
                if global_step % CONFIG["save_interval"] == 0:
                    checkpoint_path = os.path.join(stage_path, "weights.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    
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
                    
                    # Track best
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = os.path.join(stage_path, "best_weights.pt")
                        torch.save(model.state_dict(), best_path)
                        print(f"   🏆 New best loss: {best_loss:.4f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted!")
    
    # ========================================================================
    # FINAL SAVE
    # ========================================================================
    print("\n💾 Saving final model...")
    
    final_path = os.path.join(stage_path, "weights.pt")
    torch.save(model.state_dict(), final_path)
    
    elapsed_total = time.time() - start_time
    final_tokens = global_step * CONFIG["batch_size"] * CONFIG["max_seq_len"]
    
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

if __name__ == "__main__":
    main()
