#!/usr/bin/env python3
"""
Router0 Demo - V 0.9.7
================================
Demonstrates Skill-Based Expansion with Router0 Architecture.

Flow:
1. Create base model (router0 = IDK)
2. Expand for "Count Up" (Skill A) -> Train
3. Expand for "Count Down" (Skill B) -> Train Interleaved (A+B)
4. Verify 100% Retention + 100% Plasticity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dge_model import DGESimpleTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_count_up(batch_size, seq_len=4):
    """1 2 3 → 4"""
    starts = torch.randint(1, 10 - seq_len, (batch_size,))
    inputs = torch.stack([starts + i for i in range(seq_len - 1)], dim=1)
    targets = (starts + seq_len - 1).unsqueeze(1)
    return inputs.to(DEVICE), targets.to(DEVICE)

def generate_count_down(batch_size, seq_len=4):
    """5 4 3 → 2"""
    starts = torch.randint(seq_len, 10, (batch_size,))
    inputs = torch.stack([starts - i for i in range(seq_len - 1)], dim=1)
    targets = (starts - seq_len + 1).unsqueeze(1)
    return inputs.to(DEVICE), targets.to(DEVICE)

# =============================================================================
# TRAINING
# =============================================================================

def train_skill(model, generate_fn, num_steps=300, lr=0.001):
    """Train on a skill."""
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(num_steps):
        x, y = generate_fn(32)
        optimizer.zero_grad()
        
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        
        last_logits = logits[:, -1, :]
        targets_flat = y.squeeze(-1)
        
        loss = criterion(last_logits, targets_flat)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            preds = last_logits.argmax(-1)
            acc = (preds == targets_flat).float().mean().item()
            print(f"   Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")
    
    # Final accuracy
    with torch.no_grad():
        x, y = generate_fn(100)
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        preds = logits[:, -1, :].argmax(-1)
        acc = (preds == y.squeeze(-1)).float().mean().item()
    return acc

def evaluate(model, generate_fn, name):
    """Evaluate accuracy on a task."""
    model.eval()
    with torch.no_grad():
        x, y = generate_fn(100)
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        preds = logits[:, -1, :].argmax(-1)
        acc = (preds == y.squeeze(-1)).float().mean().item()
    model.train()
    print(f"   {name}: {acc:.2%}")
    return acc

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_demo():
    print("=" * 70)
    print("🧪 Router0 Always IDK Demo - V 0.9.7")
    print("   Goal: 100% Retention + 100% Plasticity via Interleaved Training")
    print("=" * 70)
    
    # 1. Create base model (router0 = IDK only)
    print("\n📦 Step 1: Creating base model (router0 = IDK only)")
    model = DGESimpleTransformer(
        vocab_size=20,
        d_model=32,
        n_layer=1, # Keep small for clarity
        n_head=2,
        initial_gating=True,
        router_type='bigram' # V26: Bigram for context
    ).to(DEVICE)
    
    print(f"   Base Param Count: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Expand for Skill A (Count Up)
    print("\n🎯 Step 2: Expanding for 'Count Up' skill")
    skill_a = model.expand_for_skill("Count Up", expansion_delta=16, router_type='bigram')
    print(f"   After expansion: d_model={model.d_model}, params={sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train Count Up
    print("\n🏋️ Step 3: Training 'Count Up'")
    acc_a = train_skill(model, generate_count_up, num_steps=300)
    print(f"   ✅ Count Up trained: {acc_a:.2%} accuracy")
    
    # 4. Freeze Skill A? NO. Interleaved training handles it.
    print("\n🔓 Step 4: Skipping freeze (Interleaved Training will maintain retention)")
    
    # 5. Expand for Skill B (Count Down)
    print("\n🎯 Step 5: Expanding for 'Count Down' skill")
    skill_b = model.expand_for_skill("Count Down", expansion_delta=16, router_type='bigram')
    print(f"   After expansion: d_model={model.d_model}, params={sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Interleaved Training
    print("\n🏋️ Step 6: Interleaved Training (Count Down + Count Up Replay)")
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(300):
        # Alternate between tasks
        if step % 2 == 0:
            x, y = generate_count_down(32)  # New skill
            task = "Down"
        else:
            x, y = generate_count_up(32)    # Replay old skill
            task = "Up"
        
        optimizer.zero_grad()
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits[:, -1, :], y.squeeze(-1))
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            # Check both tasks
            with torch.no_grad():
                x_up, y_up = generate_count_up(100)
                out_up = model(x_up)
                acc_up = (out_up[0][:, -1, :].argmax(-1) == y_up.squeeze(-1)).float().mean().item()
                
                x_down, y_down = generate_count_down(100)
                out_down = model(x_down)
                acc_down = (out_down[0][:, -1, :].argmax(-1) == y_down.squeeze(-1)).float().mean().item()
            print(f"   Step {step+1}/300 | Up: {acc_up:.2%} | Down: {acc_down:.2%}")
    
    print(f"   ✅ Interleaved Training Complete")
    
    # 7. CRITICAL TEST
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    retention = evaluate(model, generate_count_up, "Count Up Retention")
    plasticity = evaluate(model, generate_count_down, "Count Down Accuracy")
    
    print("\n📋 Skill Registry:")
    skills = model.get_skill_info()
    for sid, info in skills.items():
        status = "FROZEN" if info['frozen'] else "ACTIVE"
        print(f"   [{sid}] {info['name']}: {info['old_d_model']}→{info['new_d_model']} {status}")
        
    if retention > 0.99 and plasticity > 0.99:
        print("\n✅ SUCCESS! Both skills retained!")
    else:
        print("\n⚠️ Some retention/plasticity issues - router tuning needed")

if __name__ == "__main__":
    run_demo()
