"""
Dynamic Vocabulary Expansion Experiment (V 0.1.0)
=================================================
Tests the hypothesis that LLMs can learn new tokens during training
using DGE-inspired gated vocabulary expansion.

This experiment:
1. Pre-trains on TinyStories with base vocabulary
2. Adds new compound tokens (e.g., "Influencer") with closed gates
3. Continues training to see if gates open for useful tokens
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime
from transformers import AutoTokenizer

from dge_model import DGESimpleTransformer
from dge_logger import DGELogger
from vocab_expansion import GatedVocabExpander
from data import load_tinystories

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model
    "vocab_size": 50257,  # GPT-2
    "d_model": 256,
    "n_layer": 4,
    "n_head": 4,
    "max_seq_len": 256,
    
    # Training
    "batch_size": 32,
    "lr": 1e-4,
    "warmup_steps": 500,
    "phase1_steps": 2000,  # Pre-train on base vocab
    "phase2_steps": 2000,  # Train with expanded vocab
    
    # Vocab Expansion
    "tokens_to_add": [
        ("Influencer", ["In", "flu", "enc", "er"]),
        ("TikTok", ["T", "ik", "T", "ok"]),
        ("Instagram", ["Inst", "agram"]),
    ],
    "gate_init": -10.0,  # New tokens start closed
    "gate_boost": 10.0,  # Gradient multiplier
    
    # Output
    "output_dir": "models/vocab_expansion_experiment",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment():
    """Main experiment function."""
    print("\n" + "=" * 70)
    print("🔤 DYNAMIC VOCABULARY EXPANSION EXPERIMENT")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   Tokens to add: {[t[0] for t in CONFIG['tokens_to_add']]}")
    print("=" * 70)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = DGELogger(CONFIG["output_dir"])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("\n📦 Phase 1: Creating base model...")
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    ).to(DEVICE)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize vocab expander
    expander = GatedVocabExpander(
        model=model,
        tokenizer=tokenizer,
        gate_init=CONFIG["gate_init"],
        gate_boost=CONFIG["gate_boost"],
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    
    # ========================================================================
    # PHASE 1: Pre-train on TinyStories (base vocab)
    # ========================================================================
    print("\n" + "=" * 70)
    print("📖 Phase 1: Pre-training on TinyStories (base vocab)")
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
        print("   Using synthetic data instead...")
        dataloader = _create_synthetic_data()
    
    model.train()
    step = 0
    for epoch in range(3):  # Multiple passes
        for batch in dataloader:
            if step >= CONFIG["phase1_steps"]:
                break
            
            x = batch['input_ids'].to(DEVICE)
            y = batch['labels'].to(DEVICE) if 'labels' in batch else x
            
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
                logger.log_step({
                    "step": step,
                    "phase": "pretrain",
                    "loss": loss.item(),
                })
            
            step += 1
        
        if step >= CONFIG["phase1_steps"]:
            break
    
    print(f"   ✅ Phase 1 complete at step {step}")
    
    # ========================================================================
    # PHASE 2: Add new tokens with gating
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔤 Phase 2: Adding new tokens with gated expansion")
    print("=" * 70)
    
    for token, constituents in CONFIG["tokens_to_add"]:
        new_id = expander.add_token(token, constituent_tokens=constituents)
        print(f"   Added: '{token}' (ID: {new_id})")
    
    # Resize model embeddings for new tokens
    model.token_emb = expander.model.token_emb
    model.lm_head = expander.model.lm_head
    if hasattr(expander.model, '_vocab_gates'):
        model._vocab_gates = expander.model._vocab_gates
    
    # New optimizer for expanded model
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"] * 0.5, weight_decay=0.01)
    
    # ========================================================================
    # PHASE 3: Continue training with expanded vocab
    # ========================================================================
    print("\n" + "=" * 70)
    print("📈 Phase 3: Training with expanded vocabulary")
    print("=" * 70)
    
    gate_history = []
    
    for batch in dataloader:
        if step >= CONFIG["phase1_steps"] + CONFIG["phase2_steps"]:
            break
        
        x = batch['input_ids'].to(DEVICE)
        y = batch['labels'].to(DEVICE) if 'labels' in batch else x
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Apply gating to new token logits
        if hasattr(model, '_vocab_gates'):
            gates = torch.sigmoid(model._vocab_gates)
            logits = logits * gates.unsqueeze(0).unsqueeze(0)
            
            # Track gate values for new tokens
            if step % 100 == 0:
                new_token_gates = {
                    token: gates[tid].item() 
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
            logger.log_step({
                "step": step,
                "phase": "expansion",
                "loss": loss.item(),
            })
        
        step += 1
    
    print(f"   ✅ Phase 3 complete at step {step}")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 Results: Gate Evolution")
    print("=" * 70)
    
    for step_num, gates in gate_history:
        print(f"   Step {step_num}:")
        for token, gate_val in gates.items():
            status = "🟢 OPEN" if gate_val > 0.5 else "🔴 CLOSED"
            print(f"      {token}: {gate_val:.4f} {status}")
    
    # Save results
    results = {
        "config": CONFIG,
        "gate_history": gate_history,
        "added_tokens": expander.added_tokens,
        "final_step": step,
    }
    
    with open(os.path.join(CONFIG["output_dir"], "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    expander.save_state(os.path.join(CONFIG["output_dir"], "expander_state.json"))
    
    print("\n🏁 Experiment complete!")
    return results


def _create_synthetic_data():
    """Create synthetic data if TinyStories not available."""
    # Simple synthetic dataset
    texts = [
        "Once upon a time there was a little girl.",
        "The boy loved to play with his toys.",
        "A magical fairy flew through the forest.",
    ] * 1000
    
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.encodings = tokenizer(
                texts, 
                truncation=True, 
                padding='max_length', 
                max_length=max_len,
                return_tensors='pt'
            )
        
        def __len__(self):
            return self.encodings['input_ids'].shape[0]
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'labels': self.encodings['input_ids'][idx],
            }
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SimpleDataset(texts, tokenizer, 256)
    return DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    run_experiment()
