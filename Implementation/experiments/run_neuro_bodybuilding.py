"""
Neuro-Bodybuilding Experiment (V 0.2.0)
=======================================
Tests sparsity tuning for DGE models using genetic-algorithm inspired
fitness-based masking during training.

Features:
- Dynamic GPU detection
- Checkpointing and crash recovery
- Phase-based resume
- DGELogger integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import shutil
import time
from datetime import datetime
from dotenv import load_dotenv

import data.loader as data
import core.model as dge_model
import neuro_bodybuilding
from utils.logger import DGELogger

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

def detect_gpu_config():
    """Detect GPU and return optimal batch size."""
    if not torch.cuda.is_available():
        return 8, "CPU", 0
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 40:
        batch = 64
    elif vram_gb >= 24:
        batch = 32
    elif vram_gb >= 16:
        batch = 16
    else:
        batch = 8
    print(f"🖥️ GPU: {gpu_name} ({vram_gb:.1f}GB) → batch_size={batch}")
    return batch, gpu_name, vram_gb

BATCH_SIZE, GPU_NAME, VRAM_GB = detect_gpu_config()

CONFIG = {
    # Model
    "vocab_size": 50257,
    "d_model": 384,
    "n_layer": 6,
    "n_head": 6,
    "seq_len": 128,
    
    # Training
    "batch_size": BATCH_SIZE,
    "lr": 1e-4,
    "epochs": 5,
    "max_steps": 2000,
    "checkpoint_interval": 500,
    
    # Output - Now uses unified repo structure
    "output_dir": "models/neuro_bodybuilding",
    "gpu_name": GPU_NAME,
    "vram_gb": VRAM_GB,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use unified HF repo manager
    # Use unified HF repo manager
from hf.repo_manager import HFRepoManager, wait_for_uploads
from hf.utils import download_foundation_model # Import
HF_MANAGER = HFRepoManager("neuro_bodybuilding")
HF_REPO = "darealSven/dge"

# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, trainer, path, step, phase):
    """Save checkpoint with all training state."""
    os.makedirs(path, exist_ok=True)
    
    # Disk space check
    try:
        total, used, free = shutil.disk_usage(os.path.dirname(path) or ".")
        free_gb = free / (2**30)
        print(f"[DISK] Free: {free_gb:.2f} GB")
        if free_gb < 2.0:
            print(f"[WARN] Low disk space!")
    except:
        pass
    
    torch.save(model.state_dict(), os.path.join(path, "weights.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    
    # Save trainer history
    trainer_state = {
        "history": trainer.history,
    }
    torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))
    
    config = {
        "step": step,
        "phase": phase,
        "d_model": model.d_model,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[SAVE] Checkpoint: {path} (step={step})")

def save_resume_state(output_dir, phase, step):
    """Save resume state for crash recovery."""
    state = {"phase": phase, "step": step, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(output_dir, "resume_state.json"), "w") as f:
        json.dump(state, f, indent=2)

def load_resume_state(output_dir):
    """Load resume state if exists."""
    path = os.path.join(output_dir, "resume_state.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Main experiment with crash recovery."""
    print("\n" + "=" * 60)
    print("🏋️ NEURO-BODYBUILDING EXPERIMENT V0.2.0")
    print("=" * 60)
    print(f"   Device: {DEVICE}")
    print("=" * 60)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = DGELogger(CONFIG["output_dir"])
    
    # ========================================================================
    # RESUME DETECTION
    # ========================================================================
    resume_state = load_resume_state(CONFIG["output_dir"])
    start_step = 0
    
    if resume_state:
        start_step = resume_state.get("step", 0)
        print(f"\n🔄 RESUME: Step {start_step}")
    
    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    print("\n📦 Creating/Loading Model...")
    model = dge_model.DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        max_seq_len=CONFIG["seq_len"]
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if resuming
    ckpt_path = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
    if start_step > 0 and os.path.exists(os.path.join(ckpt_path, "weights.pt")):
        print(f"   Loading checkpoint from {ckpt_path}...")
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
        if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        print(f"   ✅ Resumed from step {start_step}")
    else:
        # Foundation Model Check (German)
        foundation_dir = os.path.join(CONFIG["output_dir"], "foundation_base")
        if download_foundation_model("german_v1", foundation_dir):
             print(f"   🏛️ Using Foundation Model: german_v1")
             try:
                 model.load_state_dict(torch.load(os.path.join(foundation_dir, "weights.pt")))
                 # We reset optimizer for the new experiment usually?
                 # Yes, Neuro Bodybuilding starts fresh tuning.
                 print("   ✅ Foundation weights loaded.")
             except Exception as e:
                 print(f"   ⚠️ Failed to load foundation weights: {e}")
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("\n📖 Loading Dataset...")
    try:
        dataloader = data.load_local_dataset(
            'german_psycho_train',
            seq_len=CONFIG["seq_len"],
            batch_size=CONFIG["batch_size"],
            tokenizer_name='gpt2'
        )
        print("   ✅ Loaded 'german_psycho_train'")
    except Exception as e:
        print(f"   ⚠️ Dataset not found ({e}). Using TinyStories fallback.")
        dataloader = data.load_tinystories(
            split='train',
            max_samples=5000,
            seq_len=CONFIG["seq_len"],
            batch_size=CONFIG["batch_size"]
        )
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    trainer = neuro_bodybuilding.NeuroTrainer(
        model, dataloader, criterion, optimizer, DEVICE, logger=logger
    )
    
    print("\n" + "=" * 60)
    print("🏋️ Starting Workout...")
    print("=" * 60)
    
    step = start_step
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        for batch_idx, batch in enumerate(dataloader):
            if step >= CONFIG["max_steps"]:
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch['input_ids'].to(DEVICE)
                y = batch.get('labels', x).to(DEVICE)
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Train step
            loss, fitness = trainer.train_step(x, y)
            
            if step % 50 == 0:
                print(f"   Epoch {epoch} | Step {step} | Loss: {loss:.4f} | Fitness: {fitness:.4f}")
                logger.log_step({
                    "step": step,
                    "epoch": epoch,
                    "loss": loss,
                    "fitness": fitness,
                })
            
            if step % CONFIG["checkpoint_interval"] == 0 and step > 0:
                save_checkpoint(model, optimizer, trainer, ckpt_path, step, 1)
                save_resume_state(CONFIG["output_dir"], 1, step)
            
            step += 1
        
        if step >= CONFIG["max_steps"]:
            break
    
    # ========================================================================
    # SAVE FINAL
    # ========================================================================
    end_time = time.time()
    
    # Save final milestone
    save_checkpoint(model, optimizer, trainer,
                   os.path.join(CONFIG["output_dir"], "milestone_final"),
                   step, 2)
    
    # Save results
    results = {
        "config": {k: v for k, v in CONFIG.items() if not callable(v)},
        "final_step": step,
        "total_time": end_time - start_time,
        "history": trainer.history[-100:],  # Last 100 entries
    }
    
    with open(os.path.join(CONFIG["output_dir"], "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Workout Complete!")
    print(f"   Time: {end_time - start_time:.2f}s")
    print(f"   Final Step: {step}")
    print(f"   Results saved to {CONFIG['output_dir']}")
    
    return results


if __name__ == "__main__":
    run_experiment()
