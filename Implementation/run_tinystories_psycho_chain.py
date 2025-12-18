#!/usr/bin/env python3
"""
TinyStories → German Psycho Experiment Chain
============================================
Automated experiment for DGE paper validation (Section 4.2).

Hypothesis: DGE will partition languages, keeping English in the frozen core
and learning German logic/persona in the expansion layers.

Phases:
1. Create/Load model (d=384, L=12, H=6, ctx=1024)
2. Train TinyStories (Baseline English)
3. Evaluate TinyStories baseline
4. Expand model (+640 d_model → 1024)
5. Train German Psycho (German + System Prompts)
6. Evaluate TinyStories retention (Validation)
7. Save experiment log

Usage:
    python run_tinystories_psycho_chain.py
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
from data import load_tinystories, load_local_dataset
from version import __version__
from run_tinystories_gsm8k_chain import (
    start_upload_worker, shutdown_upload_worker, 
    upload_to_hf_async, save_checkpoint,
    save_resume_state, load_resume_state,
    check_disk_space, ensure_checkpoint_restored,
    count_parameters, compute_perplexity
)
from hf_utils import check_for_tinystories_restorepoint, generate_model_card

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model Architecture (fixed for entire experiment)
    "vocab_size": 50257,      # GPT-2 tokenizer
    "n_layer": 12,            # Base Depth
    "max_seq_len": 1024,      # Context Window
    
    # TinyStories Phase (English Core)
    "tinystories_d_model": 384,
    "tinystories_n_head": 6,  
    "tinystories_epochs": 1,
    "tinystories_batch_size": 64,
    "tinystories_seq_len": 256,
    "tinystories_max_samples": None, # All data
    "tinystories_lr": 2e-4,
    
    # Expansion (The "Growth Spurt")
    "expansion_delta": 640,   # 384 -> 1024
    
    # German Psycho Phase (Expansion Task)
    "psycho_d_model": 1024,    # Target size
    "psycho_n_head": 16,       
    "psycho_epochs": 3,        # Train longer for complex persona
    "psycho_batch_size": 32,
    "psycho_seq_len": 256,
    "psycho_lr": 1e-4,
    "psycho_replay_ratio": 0.1, # 10% English Replay to maintain core
    
    # Paths & Checkpointing
    "output_dir": "models/tinystories_psycho_chain",
    "local_checkpoint_interval": 1000,
    "hf_upload_interval": 5000,
}

# Use unified HF repo manager
from hf_repo_manager import HFRepoManager, wait_for_uploads
HF_MANAGER = HFRepoManager("german_psycho")
HF_REPO = "darealSven/dge-models"

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 70)
    print("🧠 DGE EXPERIMENT: TinyStories → German Psycho")
    print(f"   Version: {__version__}")
    print(f"   Device: {DEVICE}")
    
    # Pre-flight Safety Check
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
    
    # Start background upload
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
    # PHASE 1: Create/Load Model (Check for Restorepoints)
    # ========================================================================
    print("\n📦 PHASE 1: Model Initialization")
    
    # Initialize Base (Small)
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["tinystories_d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["tinystories_n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    )
    
    optimizer = None
    replay_buffer = ReplayBuffer(max_size=10000, task_name="tinystories")
    skip_tinystories_training = False
    
    # --- Priority 1: Check for Resume Checkpoint (Local) ---
    ckpt_path = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
    if ensure_checkpoint_restored(ckpt_path):
        print(f"   Loading resume checkpoint from {ckpt_path}...")
        
        # If expected to be expanded (Phase 5+), expand before load
        if resume_from_phase >= 5:
            print("   Applying DGE Expansion before loading...")
            model.expand_model(
                new_input_dim=CONFIG["psycho_d_model"],
                new_output_dim=CONFIG["vocab_size"],
                router_type='bigram', use_gradient_rescue=True,
                router_init_bias=0.0
            )
            
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
        model = model.to(DEVICE)
        
        # Restore optimizer
        lr = CONFIG["tinystories_lr"] if resume_from_phase < 5 else CONFIG["psycho_lr"]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        print("   ✅ Restoration Complete.")
        
        if resume_from_phase >= 3:
            skip_tinystories_training = True
    
    # --- Priority 2: Check for TinyStories Restorepoint (Local/HF Hub) ---
    elif resume_from_phase < 3:
        restorepoint_path = check_for_tinystories_restorepoint(CONFIG["output_dir"], ensure_checkpoint_restored)
        if restorepoint_path:
            print(f"   🔄 Using TinyStories restorepoint from: {restorepoint_path}")
            model.load_state_dict(torch.load(os.path.join(restorepoint_path, "weights.pt")))
            model = model.to(DEVICE)
            skip_tinystories_training = True
            resume_from_phase = 3  # Skip to Phase 4
            print("   ✅ TinyStories Pre-trained Model Loaded. Skipping Phase 2.")
        else:
            model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)

    # ========================================================================
    # PHASE 2: Train TinyStories (Baseline) - SKIPPED if restorepoint exists
    # ========================================================================
    if not skip_tinystories_training and resume_from_phase < 3:
        print("\n📖 PHASE 2: Training TinyStories (English Core)")
        
        ts_train = load_tinystories(
            split='train',
            max_samples=CONFIG["tinystories_max_samples"],
            seq_len=CONFIG["tinystories_seq_len"],
            batch_size=CONFIG["tinystories_batch_size"],
            tokenizer_name='gpt2', vocab_size=CONFIG["vocab_size"]
        )
        
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["tinystories_lr"], weight_decay=0.01)
            
        def checkpoint_fn(step):
            should_upload = (step % CONFIG["hf_upload_interval"] == 0)
            save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                           step, {"phase": "tinystories"}, save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
            save_resume_state(CONFIG["output_dir"], 2, step)
            
        final_step = train_dataset(
            model=model, dataloader=ts_train,
            epochs=CONFIG["tinystories_epochs"], optimizer=optimizer, logger=logger,
            start_step=final_step, checkpoint_fn=checkpoint_fn,
            checkpoint_interval=CONFIG["local_checkpoint_interval"],
            replay_buffer=replay_buffer, replay_ratio=0.0,
            auto_populate_buffer=True # Save English samples for later
        )
        
        # Save Milestone
        save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "milestone_tinystories"),
                       final_step, {"phase": "tinystories_complete"}, save_optimizer=False, is_rolling=False)
    else:
        print("⏭️ Skipping Phase 2 (Using Restorepoint)")

    # ========================================================================
    # PHASE 4: Expand for German Psycho
    # ========================================================================
    if resume_from_phase < 5:
        print("\n🔧 PHASE 4: Expanding Model (New Neurons for German)")
        
        old_d = model.d_model
        model.expand_model(
            new_input_dim=CONFIG["psycho_d_model"],
            new_output_dim=CONFIG["vocab_size"],
            router_type='bigram',
            use_gradient_rescue=True,
            isolate_cross_terms=False,
            router_init_bias=0.0 # Open gates = Plasticity
        )
        model = model.to(DEVICE)
        print(f"   Expansion: {old_d} -> {model.d_model}")
        
        # Reset Optimizer for expanded parameters
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["psycho_lr"], weight_decay=0.01)
        
        # Save Expansion Init
        save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "milestone_expanded_init"),
                       final_step, {"phase": "expansion_complete"}, save_optimizer=False, is_rolling=False)
    else:
        print("⏭️ Skipping Phase 4 (Expanded)")

    # ========================================================================
    # PHASE 5: Train German Psycho (Target Task)
    # ========================================================================
    if resume_from_phase < 6:
        print("\n🇩🇪 PHASE 5: Training German Psycho (with English Replay)")
        
        # Load Standardized Dataset
        psycho_loader = load_local_dataset(
            local_name='german_psycho_train',
            seq_len=CONFIG["psycho_seq_len"],
            batch_size=CONFIG["psycho_batch_size"],
            tokenizer_name='gpt2'
        )
        
        def checkpoint_fn_psycho(step):
            should_upload = (step % CONFIG["hf_upload_interval"] == 0)
            save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "resume_checkpoint"),
                           step, {"phase": "psycho"}, save_optimizer=True, is_rolling=True, upload_to_hf=should_upload)
            save_resume_state(CONFIG["output_dir"], 5, step)
            
        final_step = train_dataset(
            model=model, dataloader=psycho_loader,
            epochs=CONFIG["psycho_epochs"], optimizer=optimizer, logger=logger,
            start_step=final_step, checkpoint_fn=checkpoint_fn_psycho,
            checkpoint_interval=CONFIG["local_checkpoint_interval"],
            replay_buffer=replay_buffer, 
            replay_ratio=CONFIG["psycho_replay_ratio"], # Maintain English
            task_name="german_psycho"
        )
        
        # Save Final
        save_checkpoint(model, optimizer, os.path.join(CONFIG["output_dir"], "milestone_psycho_final"),
                       final_step, {"phase": "psycho_complete"}, save_optimizer=False, is_rolling=False)

    # ========================================================================
    # PHASE 10: Cross-Verification (Baseline & Final)
    # ========================================================================
    print("\n" + "=" * 70)
    print("🧪 PHASE 10: Comprehensive Validation (Baselines & Retention)")
    print("=" * 70)
    
    verification_results = {}
    
    # 1. Load Test/Validation Sets
    print("   Loading validation datasets...")
    datasets = {}
    try:
        datasets["TinyStories (Base)"] = load_tinystories(
            split='validation', max_samples=200, 
            seq_len=CONFIG["tinystories_seq_len"], batch_size=CONFIG["tinystories_batch_size"], 
            tokenizer_name='gpt2', vocab_size=CONFIG["vocab_size"]
        )
        datasets["German Psycho (Target)"] = load_local_dataset(
            local_name='german_psycho_test', # Using TEST split
            seq_len=CONFIG["psycho_seq_len"], batch_size=CONFIG["psycho_batch_size"], 
            tokenizer_name='gpt2'
        )
    except Exception as e:
        print(f"⚠️ Validation Dataset Load Failed: {e}")
    
    # 2. Define Checkpoints to Verify
    checkpoints_to_audit = [
        ("milestone_tinystories", {
            "d_model": CONFIG["tinystories_d_model"],
            "n_head": CONFIG["tinystories_n_head"],
            "is_expanded": False
        }),
        ("milestone_psycho_final", {
            "d_model": CONFIG["psycho_d_model"],
            "n_head": CONFIG["psycho_n_head"],
            "is_expanded": True
        })
    ]
    
    # 3. Validation Loop
    for ckpt_name, params in checkpoints_to_audit:
        ckpt_path = os.path.join(CONFIG["output_dir"], ckpt_name)
        if not ensure_checkpoint_restored(ckpt_path):
            print(f"⏩ Skipping {ckpt_name} (Not found)")
            continue
            
        print(f"\n🔍 Auditing Model: {ckpt_name}")
        verification_results[ckpt_name] = {}
        
        try:
            # Reconstruct Model Architecture
            v_model = DGESimpleTransformer(
                vocab_size=CONFIG["vocab_size"],
                d_model=CONFIG["tinystories_d_model"], # Always start base
                n_layer=CONFIG["n_layer"],
                n_head=CONFIG["tinystories_n_head"],
                max_seq_len=CONFIG["max_seq_len"]
            )
            
            # Apply Expansion if needed
            if params["is_expanded"]:
                v_model.expand_model(
                    new_input_dim=CONFIG["psycho_d_model"],
                    new_output_dim=CONFIG["vocab_size"],
                    router_type='bigram', use_gradient_rescue=True,
                    router_init_bias=0.0
                )
            
            # Load Weights
            v_model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
            v_model = v_model.to(DEVICE)
            
            # Run Probes
            for ds_name, ds_loader in datasets.items():
                print(f"   👉 Probing with {ds_name}...")
                ppl, loss = compute_perplexity(v_model, ds_loader, max_batches=50)
                print(f"      PPL: {ppl:.2f} | Loss: {loss:.4f}")
                verification_results[ckpt_name][ds_name] = {"ppl": ppl, "loss": loss}
                
        except Exception as e:
            print(f"   ❌ Audit failed for {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()

    experiment_log["verification"] = verification_results
    
    # Save Final Log
    with open(os.path.join(CONFIG["output_dir"], "experiment_results.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)

    print("\n✅ Experiment Complete!")
    shutdown_upload_worker()

if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted.")
        shutdown_upload_worker()
    except Exception as e:
        print(f"\n\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        shutdown_upload_worker()
