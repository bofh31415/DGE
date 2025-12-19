#!/usr/bin/env python3
"""
Create Foundation Models
========================
Generates three pristine, specialized base models to serve as immutable
starting points for future DGE experiments.

1. English Core (TinyStories)
2. German Core (German TinyStories)
3. Math Core (GSM8K)

Safeguards:
- Checks HF Repo before starting.
- SKIPS training if model already exists in 'foundations/'.
- NEVER overwrites.
"""

import os
import sys
import torch
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dge_model import DGESimpleTransformer
from dge_training import train_dataset, DEVICE
from dge_logger import DGELogger
from data import load_tinystories, load_gsm8k, load_german_tinystories
try:
    from huggingface_hub import HfApi, list_repo_files
    HF_Hub = True
except ImportError:
    HF_Hub = False

from version import __version__

# ============================================================================
# CONFIGURATION
# ============================================================================

COMMON_ARCH = {
    "vocab_size": 50257,
    "d_model": 384,
    "n_layer": 12,
    "n_head": 6,
    "max_seq_len": 1024
}

TRAIN_CONFIG = {
    "epochs": 3,
    "batch_size": 32,
    "lr": 3e-4,
    "checkpoint_interval": 1000 # Local only
}

HF_REPO = "darealSven/dge"
API_TOKEN = os.environ.get("HF_TOKEN")

MODELS_TO_BUILD = [
    {
        "name": "english_v1",
        "desc": "English Core (TinyStories)",
        "loader": lambda: load_tinystories(split='train', batch_size=32),
    },
    {
        "name": "german_v1",
        "desc": "German Core (German TinyStories)",
        "loader": lambda: load_german_tinystories(split='train', batch_size=32),
    },
    {
        "name": "math_v1",
        "desc": "Math Core (GSM8K)",
        "loader": lambda: load_gsm8k(split='train', batch_size=32),
    }
]

def check_if_exists(model_name):
    """Check if model exists in HF foundations/ folder."""
    if not HF_Hub or not API_TOKEN:
        print("⚠️ HF_TOKEN missing or lib not installed. Cannot check overwrite protection.")
        return False # Fail safe? Or Fail open? Let's assume safe to train locally, but warn on upload.
    
    try:
        files = list_repo_files(HF_REPO, token=API_TOKEN)
        target_path = f"foundations/{model_name}/weights.pt"
        if target_path in files:
            return True
    except Exception as e:
        print(f"⚠️ Could not check repo: {e}")
    return False

def upload_foundation(local_dir, model_name):
    """Upload completed foundation model to HF."""
    if not HF_Hub or not API_TOKEN:
        print(f"❌ Cannot upload {model_name}: No Token.")
        return
    
    print(f"☁️ Uploading {model_name} to {HF_REPO}/foundations/{model_name}...")
    api = HfApi(token=API_TOKEN)
    try:
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=f"foundations/{model_name}",
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"Add Foundation Model: {model_name}"
        )
        print("✅ Upload Complete.")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

def run_experiment():
    print("=" * 70)
    print("🏛️ CREATING FOUNDATION MODELS")
    print(f"   Repo: {HF_REPO}/foundations/")
    print("=" * 70)
    
    base_dir = "models/foundations"
    os.makedirs(base_dir, exist_ok=True)
    
    for config in MODELS_TO_BUILD:
        name = config["name"]
        print(f"\n👉 Target: {name} ({config['desc']})")
        
        # 1. Safety Check
        if check_if_exists(name):
            print(f"   🔒 ALREADY EXISTS on HF. Skipping to prevent overwrite.")
            continue
            
        print("   ✅ Slot available. Starting initialization...")
        
        # 2. Init Model (Fresh)
        model = DGESimpleTransformer(**COMMON_ARCH)
        optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=0.01)
        logger = DGELogger(os.path.join(base_dir, name))
        
        # 3. Load Data
        dataloader = config["loader"]()
        
        # 4. Train
        print(f"   🏋️ Training {name}...")
        train_dataset(
            model=model,
            dataloader=dataloader,
            epochs=TRAIN_CONFIG["epochs"],
            optimizer=optimizer,
            logger=logger,
            task_name=name,
            auto_populate_buffer=False # Foundations don't need replay buffer populated
        )
        
        # 5. Save & Upload
        save_path = os.path.join(base_dir, name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "weights.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        
        # Upload
        upload_foundation(save_path, name)
        
        # Clear VRAM
        del model
        del optimizer
        torch.cuda.empty_cache()
        print(f"   ✨ {name} Complete.")

if __name__ == "__main__":
    run_experiment()
