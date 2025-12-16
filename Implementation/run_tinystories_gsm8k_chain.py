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
    "tinystories_batch_size": 64,
    "tinystories_seq_len": 512,
    "tinystories_max_samples": None,  # None = all (~2M)
    "tinystories_lr": 1e-4,
    
    # Expansion
    "expansion_delta": 640,   # 384 -> 1024
    
    # GSM8K Phase
    "gsm8k_d_model": 1024,    # After expansion
    "gsm8k_n_head": 16,       # head_dim = 64
    "gsm8k_epochs": 3,
    "gsm8k_batch_size": 32,
    "gsm8k_seq_len": 512,
    "gsm8k_max_samples": None,  # None = all (~7.5K)
    "gsm8k_lr": 5e-5,
    "gsm8k_replay_ratio": 0.1,  # 10% replay from TinyStories
    
    # Paths
    "output_dir": "models/tinystories_gsm8k_chain",
    "checkpoint_interval": 500,
}

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


def save_checkpoint(model, optimizer, path, step, config):
    """Save model checkpoint with config."""
    os.makedirs(path, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), os.path.join(path, "weights.pt"))
    
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
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(checkpoint_config, f, indent=2)
    
    print(f"💾 Checkpoint saved: {path}")


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
    
    # ========================================================================
    # PHASE 1: Create Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("📦 PHASE 1: Creating Initial Model")
    print("=" * 70)
    
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["tinystories_d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["tinystories_n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    )
    model = model.to(DEVICE)
    
    param_count = count_parameters(model)
    print(f"   d_model: {CONFIG['tinystories_d_model']}")
    print(f"   n_layer: {CONFIG['n_layer']}")
    print(f"   n_head: {CONFIG['tinystories_n_head']}")
    print(f"   Parameters: {param_count:,}")
    
    experiment_log["phases"]["1_create"] = {
        "params": param_count,
        "d_model": CONFIG["tinystories_d_model"],
    }
    
    logger.log_event("CREATED", {"params": param_count}, step=0)
    
    # ========================================================================
    # PHASE 2: Train TinyStories
    # ========================================================================
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
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["tinystories_lr"], weight_decay=0.01)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=10000, task_name="tinystories")
    
    # Train
    phase2_start = time.time()
    
    def checkpoint_fn(step):
        save_checkpoint(model, optimizer, 
                       os.path.join(CONFIG["output_dir"], "tinystories_checkpoint"),
                       step, {"phase": "tinystories"})
    
    final_step = train_dataset(
        model=model,
        dataloader=tinystories_train,
        epochs=CONFIG["tinystories_epochs"],
        optimizer=optimizer,
        logger=logger,
        start_step=0,
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
    
    experiment_log["phases"]["2_tinystories"] = {
        "steps": final_step,
        "time_seconds": phase2_time,
        "buffer_size": len(replay_buffer),
    }
    
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
    
    # ========================================================================
    # PHASE 10: Git Push Logs (Backup)
    # ========================================================================
    print("\n" + "=" * 70)
    print("📝 PHASE 10: Git Push Logs (Backup)")
    print("=" * 70)
    
    try:
        import subprocess
        
        # Only push small files (JSON, configs)
        subprocess.run(["git", "add", 
                       f"{CONFIG['output_dir']}/experiment_results.json",
                       f"{CONFIG['output_dir']}/*/config.json"],
                      capture_output=True, cwd=os.path.dirname(__file__))
        
        result = subprocess.run(
            ["git", "commit", "-m", f"Experiment results: {datetime.now().isoformat()}"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            push_result = subprocess.run(
                ["git", "push", "origin", "master"],
                capture_output=True, text=True, cwd=os.path.dirname(__file__)
            )
            if push_result.returncode == 0:
                print("   ✅ Logs pushed to GitHub")
                experiment_log["git_push"] = {"status": "success"}
            else:
                print(f"   ⚠️ Git push failed: {push_result.stderr}")
                experiment_log["git_push"] = {"status": "failed", "error": push_result.stderr}
        else:
            print(f"   ℹ️ Nothing to commit or commit failed")
            experiment_log["git_push"] = {"status": "no_changes"}
            
    except Exception as e:
        print(f"   ⚠️ Git operations failed: {e}")
        experiment_log["git_push"] = {"status": "error", "error": str(e)}
    
    # Save final log with upload status
    with open(results_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 ALL PHASES COMPLETE!")
    print("=" * 70)
    
    return experiment_log


if __name__ == "__main__":
    run_experiment()

