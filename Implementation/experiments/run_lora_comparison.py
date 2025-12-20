"""
LoRA vs DGE Comparison Experiment (Phase 4)

This script runs the same continual learning protocol with both:
1. DGE (Directed Synergy) - Dynamic gating with asymmetric topology
2. LoRA (Low-Rank Adaptation) - Static low-rank matrices

Win Condition for DGE:
- DGE should show lower "Forgetting" (better Stability) than LoRA
- Even if LoRA learns slightly faster, DGE should preserve old task better

Argument:
"LoRA adds parameters but doesn't gate them dynamically.
 DGE adds parameters AND gates them to prevent interference."
"""

import os
import torch
import math
import copy
from datetime import datetime
from core.model import DGESimpleTransformer
from core.training import TaskType, generate_batch, evaluate_task
from core.utils import DGEAdamW
from utils.logger import DGELogger
from lora import LoRALinear, count_trainable_params


def compute_ppl(model, task_type, vocab_size, samples=10):
    """Compute perplexity on tasks."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(samples):
            x, y = generate_batch(task_type, vocab_size, 32, 32)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / samples
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    model.train()
    return ppl


def train_steps(model, task_type, optimizer, vocab_size, steps, log_every=50):
    """Train for specified steps and return loss history."""
    model.train()
    history = []
    
    for i in range(steps):
        x, y = generate_batch(task_type, vocab_size, 32, 32)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        if i % log_every == 0 or i == steps - 1:
            history.append((i, loss.item()))
            
    return history


def wrap_lora(model, rank=8, alpha=16.0):
    """
    Wrap DGE model linear layers with LoRA.
    This freezes base weights and adds low-rank adapters.
    """
    import torch.nn as nn
    
    # Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Find and wrap linear layers (except embedding)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get parent
            parts = name.split('.')
            if len(parts) == 1:
                parent = model
                child_name = parts[0]
            else:
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                
            # Replace with LoRA
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            
    return model


def run_lora_comparison():
    """
    Main comparison: DGE vs LoRA on continual learning.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"lora_vs_dge_{timestamp}"
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    exp_dir = os.path.join(models_dir, exp_name)
    
    logger = DGELogger(exp_dir)
    
    # Configuration
    vocab_size = 500
    d_model = 64
    n_layer = 2
    n_head = 4
    lora_rank = 8
    
    print("=" * 70)
    print("🔬 COMPARISON EXPERIMENT: DGE vs LoRA")
    print("=" * 70)
    print(f"Config: vocab={vocab_size}, d_model={d_model}, LoRA rank={lora_rank}")
    print()
    
    # ===== CREATE SHARED BASE MODEL =====
    print("📦 Creating shared base model...")
    base_model = DGESimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head
    )
    
    # Train Task A on base
    print("\n📚 Training Task A on base model...")
    base_optimizer = DGEAdamW(base_model.parameters(), lr=1e-3, weight_decay=0.0)
    train_steps(base_model, TaskType.COUNT_UP, base_optimizer, vocab_size, steps=500)
    
    ppl_a_base = compute_ppl(base_model, TaskType.COUNT_UP, vocab_size)
    acc_a_base = evaluate_task(base_model, TaskType.COUNT_UP, vocab_size)
    print(f"   Base Task A: PPL={ppl_a_base:.2f}, Acc={acc_a_base:.1f}%")
    
    # ===== DGE PATH =====
    print("\n" + "=" * 70)
    print("🧠 DGE PATH: Directed Synergy Expansion")
    print("=" * 70)
    
    dge_model = copy.deepcopy(base_model)
    
    # Expand with DGE
    dge_model.expand_model(
        new_input_dim=d_model + 32,
        new_output_dim=vocab_size,
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=False,
        router_init_bias=0.0
    )
    
    trainable_dge, total_dge, pct_dge = count_trainable_params(dge_model)
    print(f"   DGE Params: {trainable_dge:,} trainable / {total_dge:,} total ({pct_dge:.1f}%)")
    
    # Verify identity
    ppl_a_dge_post = compute_ppl(dge_model, TaskType.COUNT_UP, vocab_size)
    print(f"   Post-expand Task A: PPL={ppl_a_dge_post:.2f}")
    
    # Train Task B
    dge_optimizer = DGEAdamW(dge_model.parameters(), lr=1e-3, weight_decay=0.0)
    print("\n   Training Task B...")
    dge_history = train_steps(dge_model, TaskType.COUNT_DOWN, dge_optimizer, vocab_size, steps=500)
    
    ppl_a_dge_final = compute_ppl(dge_model, TaskType.COUNT_UP, vocab_size)
    ppl_b_dge_final = compute_ppl(dge_model, TaskType.COUNT_DOWN, vocab_size)
    acc_a_dge = evaluate_task(dge_model, TaskType.COUNT_UP, vocab_size)
    acc_b_dge = evaluate_task(dge_model, TaskType.COUNT_DOWN, vocab_size)
    
    dge_forgetting = ppl_a_dge_final - ppl_a_base
    
    print(f"\n   DGE RESULTS:")
    print(f"   Task A: PPL={ppl_a_dge_final:.2f}, Acc={acc_a_dge:.1f}%, Forgetting={dge_forgetting:+.2f}")
    print(f"   Task B: PPL={ppl_b_dge_final:.2f}, Acc={acc_b_dge:.1f}%")
    
    # ===== LORA PATH =====
    print("\n" + "=" * 70)
    print("🔧 LORA PATH: Low-Rank Adaptation")
    print("=" * 70)
    
    lora_model = copy.deepcopy(base_model)
    lora_model = wrap_lora(lora_model, rank=lora_rank, alpha=16.0)
    
    trainable_lora, total_lora, pct_lora = count_trainable_params(lora_model)
    print(f"   LoRA Params: {trainable_lora:,} trainable / {total_lora:,} total ({pct_lora:.1f}%)")
    
    # Verify identity (LoRA starts at zero, so should be identity)
    ppl_a_lora_post = compute_ppl(lora_model, TaskType.COUNT_UP, vocab_size)
    print(f"   Post-wrap Task A: PPL={ppl_a_lora_post:.2f}")
    
    # Train Task B
    lora_optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=1e-3
    )
    print("\n   Training Task B...")
    lora_history = train_steps(lora_model, TaskType.COUNT_DOWN, lora_optimizer, vocab_size, steps=500)
    
    ppl_a_lora_final = compute_ppl(lora_model, TaskType.COUNT_UP, vocab_size)
    ppl_b_lora_final = compute_ppl(lora_model, TaskType.COUNT_DOWN, vocab_size)
    acc_a_lora = evaluate_task(lora_model, TaskType.COUNT_UP, vocab_size)
    acc_b_lora = evaluate_task(lora_model, TaskType.COUNT_DOWN, vocab_size)
    
    lora_forgetting = ppl_a_lora_final - ppl_a_base
    
    print(f"\n   LORA RESULTS:")
    print(f"   Task A: PPL={ppl_a_lora_final:.2f}, Acc={acc_a_lora:.1f}%, Forgetting={lora_forgetting:+.2f}")
    print(f"   Task B: PPL={ppl_b_lora_final:.2f}, Acc={acc_b_lora:.1f}%")
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'DGE':>15} {'LoRA':>15} {'Winner':>15}")
    print("-" * 70)
    
    # Stability (lower forgetting = better)
    stability_winner = "DGE" if dge_forgetting < lora_forgetting else "LoRA"
    print(f"{'Forgetting (Task A PPL Δ)':<25} {dge_forgetting:>+15.2f} {lora_forgetting:>+15.2f} {stability_winner:>15}")
    
    # Plasticity (lower Task B PPL = better)
    plasticity_winner = "DGE" if ppl_b_dge_final < ppl_b_lora_final else "LoRA"
    print(f"{'Task B PPL':<25} {ppl_b_dge_final:>15.2f} {ppl_b_lora_final:>15.2f} {plasticity_winner:>15}")
    
    # Task A Accuracy
    acc_a_winner = "DGE" if acc_a_dge > acc_a_lora else "LoRA"
    print(f"{'Task A Accuracy':<25} {acc_a_dge:>14.1f}% {acc_a_lora:>14.1f}% {acc_a_winner:>15}")
    
    # Task B Accuracy
    acc_b_winner = "DGE" if acc_b_dge > acc_b_lora else "LoRA"
    print(f"{'Task B Accuracy':<25} {acc_b_dge:>14.1f}% {acc_b_lora:>14.1f}% {acc_b_winner:>15}")
    
    # Parameter Efficiency
    param_winner = "DGE" if pct_dge < pct_lora else "LoRA"
    print(f"{'Trainable % of Params':<25} {pct_dge:>14.1f}% {pct_lora:>14.1f}% {param_winner:>15}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    
    dge_wins = sum([
        dge_forgetting < lora_forgetting,
        ppl_b_dge_final < ppl_b_lora_final,
        acc_a_dge > acc_a_lora,
        acc_b_dge > acc_b_lora
    ])
    
    if dge_wins >= 3:
        print("🏆 WINNER: DGE (Directed Synergy)")
        print("   Dynamic gating prevents interference better than static LoRA.")
    elif dge_wins == 2:
        print("🤝 TIE: Both methods perform similarly")
    else:
        print("🏆 WINNER: LoRA")
        print("   Static low-rank adaptation outperformed DGE on this benchmark.")
    
    print("=" * 70)
    
    # Log results
    logger.log_event("COMPARISON_COMPLETE", {
        "dge_forgetting": dge_forgetting,
        "lora_forgetting": lora_forgetting,
        "dge_task_b_ppl": ppl_b_dge_final,
        "lora_task_b_ppl": ppl_b_lora_final,
        "dge_task_a_acc": acc_a_dge,
        "lora_task_a_acc": acc_a_lora,
        "dge_wins": dge_wins,
        "winner": "DGE" if dge_wins >= 3 else ("TIE" if dge_wins == 2 else "LoRA")
    }, step=0)
    
    print(f"\n📁 Logs saved to: {exp_dir}")
    
    return {
        "dge_forgetting": dge_forgetting,
        "lora_forgetting": lora_forgetting,
        "winner": "DGE" if dge_wins >= 3 else ("TIE" if dge_wins == 2 else "LoRA")
    }


if __name__ == "__main__":
    run_lora_comparison()
