#!/usr/bin/env python3
"""
Rosetta Stone Experiment
========================
Hypothesis: DGE can combine "English Math Logic" (Core) and "German Language" (Expansion)
to solve "German Math Word Problems" (Synergy Task), which it never saw during training.

Phases:
1. Load TinyStories Base (English Fluency)
2. Train GSM8K (English Math) -> Imprint Logic into Core
3. Expand Model (+Capacity)
4. Train German Psycho -> Learn German in Expansion
5. Evaluate on MGSM (German Math) -> Verify Synergy

Usage:
    python run_rosetta_stone_experiment.py
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
from data import load_tinystories, load_gsm8k, load_local_dataset, load_mgsm
from version import __version__
from run_tinystories_gsm8k_chain import (
    start_upload_worker, shutdown_upload_worker, 
    upload_to_hf_async, save_checkpoint,
    save_resume_state, load_resume_state,
    check_disk_space, ensure_checkpoint_restored,
    count_parameters, compute_perplexity,
    set_experiment_folder
)
from hf_utils import check_for_tinystories_restorepoint, generate_model_card

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model Architecture
    "vocab_size": 50257,
    "n_layer": 12,
    "max_seq_len": 1024,
    
    # Phase 1: TinyStories Base (Core)
    "core_d_model": 384,
    "core_n_head": 6,
    
    # Phase 2: GSM8K (English Math - Logic Imprinting)
    "gsm8k_epochs": 3,
    "gsm8k_batch_size": 16, # Smaller batch for reasoning density
    "gsm8k_seq_len": 256,
    "gsm8k_lr": 1e-4,
    
    # Expansion
    "expansion_delta": 640,
    
    # Phase 4: German Psycho (Language Acquisition)
    "german_d_model": 1024, # 384 + 640
    "german_n_head": 16,
    "german_epochs": 3,
    "german_batch_size": 32,
    "german_seq_len": 256,
    "german_lr": 1e-4,
    "german_replay_ratio": 0.1, # Keep English/Math alive
    
    # Paths
    "output_dir": "models/rosetta_stone",
    "local_checkpoint_interval": 1000,
    "hf_upload_interval": 5000,
}

# Use unified HF repo manager
from hf_repo_manager import HFRepoManager
HF_MANAGER = HFRepoManager("rosetta_stone")
HF_REPO = "darealSven/dge"

def run_experiment():
    print("=" * 70)
    print("🗿 ROSETTA STONE EXPERIMENT: English Logic + German Language = Synergy")
    print(f"   Version: {__version__}")
    print(f"   Device: {DEVICE}")
    
    check_disk_space(20)
    print("=" * 70)
    
    experiment_log = {
        "version": __version__,
        "config": CONFIG,
        "phases": {},
        "start_time": datetime.now().isoformat(),
    }
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = DGELogger(CONFIG["output_dir"])
    
    # Configure Uploads
    set_experiment_folder("rosetta_stone")
    start_upload_worker()
    
    # --- Resume Detection ---
    resume_state = load_resume_state(CONFIG["output_dir"])
    resume_from_phase = 0
    final_step = 0
    
    if resume_state:
        resume_from_phase = resume_state.get("phase", 0)
        final_step = resume_state.get("step", 0)
        print(f"\n🔄 RESUME DETECTED: Phase {resume_from_phase}, Step {final_step}")

    # ========================================================================
    # PHASE 1: Initialize / Load Base
    # ========================================================================
    print("\n📦 PHASE 1: Model Initialization")
    
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["core_d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["core_n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    )
    
    optimizer = None
    replay_buffer = ReplayBuffer(max_size=10000, task_name="core_mix")
    
    # Check for Resume Checkpoint
    ckpt_path = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
    if ensure_checkpoint_restored(ckpt_path):
        print(f"   Loading resume checkpoint from {ckpt_path}...")
        
        # If expected to be expanded (Phase 4+), expand before load
        if resume_from_phase >= 3:
            print("   Applying DGE Expansion before loading...")
            model.expand_model(
                new_input_dim=CONFIG["german_d_model"],
                new_output_dim=CONFIG["vocab_size"],
                router_type='bigram', use_gradient_rescue=True,
                router_init_bias=0.0
            )
            
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
        model = model.to(DEVICE)
        
        # Restore optimizer
        lr = CONFIG["gsm8k_lr"] if resume_from_phase < 3 else CONFIG["german_lr"]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        print("   ✅ Restoration Complete.")
        
    else:
        # If no resume, try to load TinyStories Base
        restorepoint_path = check_for_tinystories_restorepoint(CONFIG["output_dir"], ensure_checkpoint_restored)
        if restorepoint_path:
            print(f"   🔄 Using TinyStories restored base from: {restorepoint_path}")
            model.load_state_dict(torch.load(os.path.join(restorepoint_path, "weights.pt")))
            model = model.to(DEVICE)
        else:
            print("   ⚠️ No base model found! Starting from scratch (Suboptimal for this experiment).")
            model = model.to(DEVICE)

    # ========================================================================
    # PHASE 2: Train GSM8K (English Math Logic)
    # ========================================================================
    if resume_from_phase < 2:
        print("\n🧮 PHASE 2: Imprinting English Math Logic (GSM8K)")
        
        gsm8k_loader = load_gsm8k(
            split='train',
            seq_len=CONFIG["gsm8k_seq_len"],
            batch_size=CONFIG["gsm8k_batch_size"],
            tokenizer_name='gpt2'
        )
        
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["gsm8k_lr"], weight_decay=0.01)
            
        def checkpoint_fn_gsm8k(step):
            should_upload = (step % CONFIG["hf_upload_interval"] == 0)
            save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                           step, {"phase": "gsm8k"}, save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
            save_resume_state(CONFIG["output_dir"], 2, step)
            
        final_step = train_dataset(
            model=model, dataloader=gsm8k_loader,
            epochs=CONFIG["gsm8k_epochs"], optimizer=optimizer, logger=logger,
            start_step=final_step, checkpoint_fn=checkpoint_fn_gsm8k,
            checkpoint_interval=CONFIG["local_checkpoint_interval"],
            replay_buffer=replay_buffer, replay_ratio=0.0,
            auto_populate_buffer=True
        )
        
        save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "milestone_logic_expert"),
                       final_step, {"phase": "gsm8k_complete"}, save_optimizer=False, is_rolling=False)
    else:
        print("⏭️ Skipping Phase 2 (GSM8K Logic Imprinted)")

    # ========================================================================
    # PHASE 3: Expansion
    # ========================================================================
    if resume_from_phase < 3:
        print("\n🔧 PHASE 3: Exanding Model Capacity")
        
        model.expand_model(
            new_input_dim=CONFIG["german_d_model"],
            new_output_dim=CONFIG["vocab_size"],
            router_type='bigram',
            use_gradient_rescue=True,
            router_init_bias=0.0
        )
        model = model.to(DEVICE)
        
        # Reset Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["german_lr"], weight_decay=0.01)
        save_resume_state(CONFIG["output_dir"], 3, final_step)
    else:
        print("⏭️ Skipping Phase 3 (Expanded)")

    # ========================================================================
    # PHASE 4: Train German TinyStories (Language)
    # ========================================================================
    if resume_from_phase < 4:
        print("\n🇩🇪 PHASE 4: Learning German (German TinyStories)")
        print("\n   UPGRADE: Using SkySyrup/tinystories_german (2.8M samples)")
        
        german_loader = load_german_tinystories(
            split='train',
            max_samples=None, # Use all available if possible, or limit? Default None is fine.
            seq_len=CONFIG["german_seq_len"],
            batch_size=CONFIG["german_batch_size"],
            tokenizer_name='gpt2'
        )
        
        def checkpoint_fn_german(step):
            should_upload = (step % CONFIG["hf_upload_interval"] == 0)
            save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                           step, {"phase": "german"}, save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
            save_resume_state(CONFIG["output_dir"], 4, step)
            
        final_step = train_dataset(
            model=model, dataloader=german_loader,
            epochs=CONFIG["german_epochs"], optimizer=optimizer, logger=logger,
            start_step=final_step, checkpoint_fn=checkpoint_fn_german,
            checkpoint_interval=CONFIG["local_checkpoint_interval"],
            replay_buffer=replay_buffer, 
            replay_ratio=CONFIG["german_replay_ratio"], # Replay GSM8K/TS
            task_name="german_tinystories"
        )
        
        save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "milestone_rosetta_final"),
                       final_step, {"phase": "rosetta_complete"}, save_optimizer=False, is_rolling=False)

    # ========================================================================
    # PHASE 5: Synergy Test (MGSM - German Math)
    # ========================================================================
    print("\n" + "=" * 70)
    print("🧪 PHASE 5: The Rosetta Synergy Test (German Math)")
    print("=" * 70)
    
    results = {}
    
    # Load MGSM (German)
    try:
        mgsm_loader = load_mgsm(split='test', lang='de', batch_size=16)
    except Exception as e:
        print(f"⚠️ Could not load MGSM: {e}")
        mgsm_loader = None
        
    if mgsm_loader:
        # 1. Evaluate Final Model (Rosetta)
        print("\n👉 Evaluating Rosetta Model (English Logic + German Language)...")
        ppl, loss = compute_perplexity(model, mgsm_loader, max_batches=20)
        print(f"   Results: PPL={ppl:.2f}, Loss={loss:.4f}")
        results["rosetta_final"] = {"ppl": ppl, "loss": loss}
        
    # Save Results
    with open(os.path.join(CONFIG["output_dir"], "synergy_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Rosetta Stone Experiment Complete!")
    shutdown_upload_worker()

if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
        shutdown_upload_worker()
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        shutdown_upload_worker()
