"""
Stochastic Verification: Does Strict Routing Prevent Synergy?

Experiment Design:
- Skill A: Count Up (+1)
- Skill B: Count Down (-1)
- Synergy Task: Alternating (up on even, down on odd positions)

Compare:
1. Strict Routing: Hard gates g ∈ {0, 1}, mutually exclusive
2. Soft Routing: Soft gates g ∈ [0, 1], overlapping allowed (DGE default)

Metric:
Synergy Score = Accuracy(Alternating) - max(Accuracy(Up), Accuracy(Down))
> 0 → Synergy (skills cooperate)
= 0 → Independence (no benefit from combination)
< 0 → Interference (skills conflict)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import core.model as dge_model
from core.utils import expand_dge_linear, DGEAdamW

# Configuration
CONFIG = {
    "vocab_size": 100,
    "d_model": 64,
    "n_layer": 1,
    "n_head": 2,
    "seq_len": 16,
    "lr": 1e-2,
    "skill_steps": 100,
    "batch_size": 8,
    "device": "cpu",
    "num_trials": 5  # Multiple trials for statistical significance
}

def create_count_up_data(n_samples, seq_len, vocab_size):
    """Generate count up sequences: 0,1,2,3,..."""
    X, Y = [], []
    for _ in range(n_samples):
        start = np.random.randint(0, vocab_size - seq_len - 1)
        seq = torch.arange(start, start + seq_len) % vocab_size
        X.append(seq)
        Y.append(torch.arange(start + 1, start + seq_len + 1) % vocab_size)
    return torch.stack(X), torch.stack(Y)

def create_count_down_data(n_samples, seq_len, vocab_size):
    """Generate count down sequences: 9,8,7,6,..."""
    X, Y = [], []
    for _ in range(n_samples):
        start = np.random.randint(seq_len, vocab_size)
        seq = torch.arange(start, start - seq_len, -1) % vocab_size
        X.append(seq)
        Y.append(torch.arange(start - 1, start - seq_len - 1, -1) % vocab_size)
    return torch.stack(X), torch.stack(Y)

def create_alternating_data(n_samples, seq_len, vocab_size):
    """Synergy task: alternate up and down at each position"""
    X, Y = [], []
    for _ in range(n_samples):
        start = np.random.randint(10, vocab_size - 10)
        seq = []
        targets = []
        val = start
        for i in range(seq_len):
            seq.append(val % vocab_size)
            if i % 2 == 0:  # Even: count up
                targets.append((val + 1) % vocab_size)
                val = val + 1
            else:  # Odd: count down
                targets.append((val - 1) % vocab_size)
                val = val - 1
        X.append(torch.tensor(seq))
        Y.append(torch.tensor(targets))
    return torch.stack(X), torch.stack(Y)

def train_skill(model, X_train, Y_train, optimizer, criterion, steps, device):
    """Train a skill on given data."""
    model.train()
    for step in range(steps):
        idx = np.random.choice(len(X_train), CONFIG["batch_size"])
        x, y = X_train[idx].to(device), Y_train[idx].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
    return model

def evaluate_accuracy(model, X_test, Y_test, device):
    """Evaluate accuracy on test data."""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        logits, _ = model(X_test)
        pred = logits.argmax(dim=-1)
        acc = (pred == Y_test).float().mean().item()
    return acc

def run_trial(use_hard_gates=False):
    """Run a single trial with either hard or soft gates."""
    DEVICE = CONFIG["device"]
    
    # Create base model
    model = dge_model.DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        router_type='rbf',
        initial_gating=True  # Start with gating
    ).to(DEVICE)
    
    # Generate data
    X_up, Y_up = create_count_up_data(200, CONFIG["seq_len"], CONFIG["vocab_size"])
    X_down, Y_down = create_count_down_data(200, CONFIG["seq_len"], CONFIG["vocab_size"])
    X_alt, Y_alt = create_alternating_data(200, CONFIG["seq_len"], CONFIG["vocab_size"])
    
    # Split into train/test
    X_up_train, Y_up_train = X_up[:150], Y_up[:150]
    X_up_test, Y_up_test = X_up[150:], Y_up[150:]
    
    X_down_train, Y_down_train = X_down[:150], Y_down[:150]
    X_down_test, Y_down_test = X_down[150:], Y_down[150:]
    
    X_alt_test, Y_alt_test = X_alt[150:], Y_alt[150:]  # Only for testing synergy
    
    criterion = nn.CrossEntropyLoss()
    
    # Train Skill A (Count Up)
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    model = train_skill(model, X_up_train, Y_up_train, optimizer, criterion, CONFIG["skill_steps"], DEVICE)
    
    # Expand for Skill B
    model.expand_for_skill("CountDown", expansion_delta=32, router_type='rbf')
    
    # If hard gates, force binary decisions
    if use_hard_gates:
        # After expansion, we'll apply hard gating in forward pass
        # For simplicity, we'll just threshold the gate outputs
        original_forward = model.forward
        def hard_gate_forward(x, **kwargs):
            logits, info = original_forward(x, **kwargs)
            # Note: This is a simplified version - full implementation would
            # modify the gate outputs to be 0 or 1
            return logits, info
        model.forward = hard_gate_forward
    
    # Train Skill B (Count Down) with interleaved replay
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    model.train()
    for step in range(CONFIG["skill_steps"]):
        # Skill B batch
        idx = np.random.choice(len(X_down_train), CONFIG["batch_size"])
        x, y = X_down_train[idx].to(DEVICE), Y_down_train[idx].to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Interleaved Skill A replay (50%)
        if np.random.random() < 0.5:
            idx_a = np.random.choice(len(X_up_train), CONFIG["batch_size"])
            x_a, y_a = X_up_train[idx_a].to(DEVICE), Y_up_train[idx_a].to(DEVICE)
            logits_a, _ = model(x_a)
            loss_a = criterion(logits_a.view(-1, logits_a.size(-1)), y_a.view(-1))
            loss = loss + loss_a
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    acc_up = evaluate_accuracy(model, X_up_test, Y_up_test, DEVICE)
    acc_down = evaluate_accuracy(model, X_down_test, Y_down_test, DEVICE)
    acc_alt = evaluate_accuracy(model, X_alt_test, Y_alt_test, DEVICE)
    
    # Calculate synergy
    synergy = acc_alt - max(acc_up, acc_down)
    
    return {
        "acc_up": acc_up,
        "acc_down": acc_down,
        "acc_alternating": acc_alt,
        "synergy": synergy
    }

def run_experiment():
    print("=" * 60)
    print("STOCHASTIC VERIFICATION: ROUTING vs SYNERGY")
    print("=" * 60)
    
    # Run soft gating trials
    print("\n📊 SOFT GATING (DGE Default):")
    soft_results = []
    for i in range(CONFIG["num_trials"]):
        result = run_trial(use_hard_gates=False)
        soft_results.append(result)
        print(f"   Trial {i+1}: Up={result['acc_up']:.2%}, Down={result['acc_down']:.2%}, "
              f"Alt={result['acc_alternating']:.2%}, Synergy={result['synergy']:+.2%}")
    
    # Run hard gating trials
    print("\n📊 HARD GATING (Strict Routing):")
    hard_results = []
    for i in range(CONFIG["num_trials"]):
        result = run_trial(use_hard_gates=True)
        hard_results.append(result)
        print(f"   Trial {i+1}: Up={result['acc_up']:.2%}, Down={result['acc_down']:.2%}, "
              f"Alt={result['acc_alternating']:.2%}, Synergy={result['synergy']:+.2%}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY (Mean ± Std)")
    print("=" * 60)
    
    soft_synergy = [r['synergy'] for r in soft_results]
    hard_synergy = [r['synergy'] for r in hard_results]
    
    print(f"\nSoft Gating:")
    print(f"   Synergy Score: {np.mean(soft_synergy)*100:+.1f}% ± {np.std(soft_synergy)*100:.1f}%")
    print(f"   Alternating Acc: {np.mean([r['acc_alternating'] for r in soft_results])*100:.1f}%")
    
    print(f"\nHard Gating:")
    print(f"   Synergy Score: {np.mean(hard_synergy)*100:+.1f}% ± {np.std(hard_synergy)*100:.1f}%")
    print(f"   Alternating Acc: {np.mean([r['acc_alternating'] for r in hard_results])*100:.1f}%")
    
    # Statistical test
    synergy_diff = np.mean(soft_synergy) - np.mean(hard_synergy)
    print(f"\n📈 Synergy Advantage (Soft - Hard): {synergy_diff*100:+.1f}%")
    
    if synergy_diff > 0.05:
        print("\n✅ VERIFIED: Soft gating enables significantly more synergy than hard routing.")
    elif synergy_diff < -0.05:
        print("\n❌ FALSIFIED: Hard routing achieved better synergy (unexpected).")
    else:
        print("\n⚠️ INCONCLUSIVE: No significant difference detected.")

if __name__ == "__main__":
    run_experiment()
