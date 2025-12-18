"""
Dynamic Vocabulary Expansion Experiment (V 0.2.0)
=================================================
Tests the hypothesis that LLMs can learn new tokens during training
using DGE-inspired gated vocabulary expansion.

This experiment:
1. Pre-trains on TinyStories with base vocabulary
2. Adds new compound tokens (e.g., "Influencer") with closed gates
3. Continues training to see if gates open for useful tokens

Features:
- Checkpointing and crash recovery
- HuggingFace Hub upload
- Phase-based resume
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
import shutil
from datetime import datetime
from transformers import AutoTokenizer

from dge_model import DGESimpleTransformer
from dge_logger import DGELogger
from vocab_expansion import GatedVocabExpander
from data import load_tinystories

# ============================================================================
# CONFIGURATION
# ============================================================================

def detect_gpu_config():
    """Detect GPU and return optimal batch size."""
    if not torch.cuda.is_available():
        return 4, "CPU", 0
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
    "vocab_size": 50257,  # GPT-2
    "d_model": 256,
    "n_layer": 4,
    "n_head": 4,
    "max_seq_len": 256,
    
    # Training
    "batch_size": BATCH_SIZE,
    "lr": 1e-4,
    "warmup_steps": 500,
    "phase1_steps": 2000,  # Pre-train on base vocab
    "phase2_steps": 2000,  # Train with expanded vocab
    "checkpoint_interval": 500,  # Restore every N steps
    
    # Vocab Expansion
    "tokens_to_add": [
        ("Influencer", ["In", "flu", "enc", "er"]),
        ("TikTok", ["T", "ik", "T", "ok"]),
        ("Instagram", ["Inst", "agram"]),
    ],
    "gate_init": -10.0,
    "gate_boost": 10.0,
    
    # Output & HF - Now uses unified repo structure
    "output_dir": "models/vocab_expansion_experiment",
    "gpu_name": GPU_NAME,
    "vram_gb": VRAM_GB,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use unified HF repo manager
from hf_repo_manager import HFRepoManager, wait_for_uploads
HF_MANAGER = HFRepoManager("vocab_expansion")
HF_REPO = "darealSven/dge-models"

# ============================================================================
# CHECKPOINTING (Reuse from GSM8K chain)
# ============================================================================

def save_checkpoint(model, optimizer, path, step, phase, expander=None):
    """Save checkpoint with config."""
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
    
    config = {
        "step": step,
        "phase": phase,
        "d_model": model.d_model,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if expander:
        expander.save_state(os.path.join(path, "expander_state.json"))
    
    print(f"[SAVE] Checkpoint: {path} (step={step}, phase={phase})")

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
    """Main experiment function with crash recovery."""
    print("\n" + "=" * 70)
    print("🔤 DYNAMIC VOCABULARY EXPANSION EXPERIMENT V0.2.0")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   Tokens to add: {[t[0] for t in CONFIG['tokens_to_add']]}")
    print("=" * 70)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = DGELogger(CONFIG["output_dir"])
    
    # ========================================================================
    # RESUME DETECTION
    # ========================================================================
    resume_state = load_resume_state(CONFIG["output_dir"])
    resume_phase = 0
    start_step = 0
    
    if resume_state:
        resume_phase = resume_state.get("phase", 0)
        start_step = resume_state.get("step", 0)
        print(f"\n🔄 RESUME: Phase {resume_phase}, Step {start_step}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("\n📦 Creating/Loading Model...")
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    ).to(DEVICE)
    
    expander = GatedVocabExpander(
        model=model,
        tokenizer=tokenizer,
        gate_init=CONFIG["gate_init"],
        gate_boost=CONFIG["gate_boost"],
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    
    # Load checkpoint if resuming
    ckpt_path = os.path.join(CONFIG["output_dir"], "resume_checkpoint")
    if resume_phase > 0 and os.path.exists(os.path.join(ckpt_path, "weights.pt")):
        print(f"   Loading checkpoint from {ckpt_path}...")
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "weights.pt")))
        if os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        if os.path.exists(os.path.join(ckpt_path, "expander_state.json")):
            expander.load_state(os.path.join(ckpt_path, "expander_state.json"))
        print(f"   ✅ Resumed from step {start_step}")
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # PHASE 1: Pre-train (skip if already done)
    # ========================================================================
    step = start_step
    
    if resume_phase < 2:
        print("\n" + "=" * 70)
        print("📖 Phase 1: Pre-training on TinyStories")
        print("=" * 70)
        
        try:
            dataloader = load_tinystories(
                split='train',
                max_samples=10000,
                seq_len=CONFIG["max_seq_len"],
                batch_size=CONFIG["batch_size"],
                tokenizer_name='gpt2',
                vocab_size=CONFIG["vocab_size"]
            )
        except Exception as e:
            print(f"⚠️ Could not load TinyStories: {e}")
            print("   Using synthetic data...")
            dataloader = _create_synthetic_data(tokenizer)
        
        model.train()
        for epoch in range(3):
            for batch in dataloader:
                if step >= CONFIG["phase1_steps"]:
                    break
                
                x = batch['input_ids'].to(DEVICE)
                y = batch.get('labels', x).to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    y[:, 1:].reshape(-1),
                    ignore_index=-100
                )
                loss.backward()
                optimizer.step()
                
                if step % 100 == 0:
                    print(f"   Step {step}: Loss = {loss.item():.4f}")
                    logger.log_step({"step": step, "phase": "pretrain", "loss": loss.item()})
                
                if step % CONFIG["checkpoint_interval"] == 0 and step > 0:
                    save_checkpoint(model, optimizer, ckpt_path, step, 1, expander)
                    save_resume_state(CONFIG["output_dir"], 1, step)
                
                step += 1
            
            if step >= CONFIG["phase1_steps"]:
                break
        
        # Save Phase 1 milestone
        save_checkpoint(model, optimizer, 
                       os.path.join(CONFIG["output_dir"], "milestone_pretrain"),
                       step, 1, expander)
        save_resume_state(CONFIG["output_dir"], 2, step)
        print(f"   ✅ Phase 1 complete at step {step}")
    
    # ========================================================================
    # PHASE 2: Add tokens
    # ========================================================================
    if resume_phase < 3:
        print("\n" + "=" * 70)
        print("🔤 Phase 2: Adding new tokens with gating")
        print("=" * 70)
        
        for token, constituents in CONFIG["tokens_to_add"]:
            new_id = expander.add_token(token, constituent_tokens=constituents)
            print(f"   Added: '{token}' (ID: {new_id})")
        
        # Transfer to model
        model.token_emb = expander.model.token_emb
        model.lm_head = expander.model.lm_head
        if hasattr(expander.model, '_vocab_gates'):
            model._vocab_gates = expander.model._vocab_gates
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"] * 0.5, weight_decay=0.01)
        save_resume_state(CONFIG["output_dir"], 3, step)
    
    # ========================================================================
    # PHASE 3: Train with expanded vocab
    # ========================================================================
    print("\n" + "=" * 70)
    print("📈 Phase 3: Training with expanded vocabulary")
    print("=" * 70)
    
    gate_history = []
    
    try:
        dataloader = load_tinystories(
            split='train', max_samples=10000, seq_len=CONFIG["max_seq_len"],
            batch_size=CONFIG["batch_size"], tokenizer_name='gpt2',
            vocab_size=CONFIG["vocab_size"]
        )
    except:
        dataloader = _create_synthetic_data(tokenizer)
    
    for batch in dataloader:
        if step >= CONFIG["phase1_steps"] + CONFIG["phase2_steps"]:
            break
        
        x = batch['input_ids'].to(DEVICE)
        y = batch.get('labels', x).to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Apply gating
        if hasattr(model, '_vocab_gates'):
            gates = torch.sigmoid(model._vocab_gates)
            # Only gate up to logits vocab size
            if gates.size(0) <= logits.size(-1):
                logits[:, :, :gates.size(0)] = logits[:, :, :gates.size(0)] * gates.unsqueeze(0).unsqueeze(0)
            
            if step % 100 == 0:
                new_token_gates = {
                    token: gates[tid].item() if tid < gates.size(0) else 0
                    for token, tid in expander.added_tokens.items()
                }
                gate_history.append((step, new_token_gates))
                print(f"   Step {step}: Gates = {new_token_gates}")
        
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            logger.log_step({"step": step, "phase": "expansion", "loss": loss.item()})
        
        if step % CONFIG["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, ckpt_path, step, 3, expander)
            save_resume_state(CONFIG["output_dir"], 3, step)
        
        step += 1
    
    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 Results: Gate Evolution")
    print("=" * 70)
    
    for step_num, gates in gate_history[-5:]:  # Last 5 entries
        print(f"   Step {step_num}:")
        for token, gate_val in gates.items():
            status = "🟢 OPEN" if gate_val > 0.5 else "🔴 CLOSED"
            print(f"      {token}: {gate_val:.4f} {status}")
    
    # Save milestone
    save_checkpoint(model, optimizer,
                   os.path.join(CONFIG["output_dir"], "milestone_final"),
                   step, 4, expander)
    
    results = {
        "config": {k: v for k, v in CONFIG.items() if not callable(v)},
        "gate_history": gate_history,
        "added_tokens": expander.added_tokens,
        "final_step": step,
    }
    
    with open(os.path.join(CONFIG["output_dir"], "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n🏁 Experiment complete!")
    return results


def _create_synthetic_data(tokenizer):
    """Create synthetic data if TinyStories not available."""
    from torch.utils.data import DataLoader, Dataset
    
    texts = [
        "Once upon a time there was a little girl.",
        "The boy loved to play with his toys.",
        "A magical fairy flew through the forest.",
    ] * 1000
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.encodings = tokenizer(
                texts, truncation=True, padding='max_length',
                max_length=max_len, return_tensors='pt'
            )
        
        def __len__(self):
            return self.encodings['input_ids'].shape[0]
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'labels': self.encodings['input_ids'][idx],
            }
    
    dataset = SimpleDataset(texts, tokenizer, 256)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)


if __name__ == "__main__":
    run_experiment()
