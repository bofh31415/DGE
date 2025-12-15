"""
Diagnostic Experiment: Dead Sidecar Investigation

Goal: Identify why Plasticity fails despite "correct" Directed Synergy settings.

Diagnostics to collect:
1. Gate Activations: Are gates actually open? (mean_open > 0.5)
2. Gradient Norms: Active_Grad_Norm, Gate_Grad_Norm, Frozen_Grad_Norm
3. Weight Magnitudes: How large are new weights vs old?
4. Signal Flow: Forward activation magnitudes per quadrant
5. LM Head: Is the head reading from new dimensions?
"""

import os
import torch
import torch.nn as nn
import math
from datetime import datetime
from dge_model import DGESimpleTransformer
from dge_training import TaskType, generate_batch, evaluate_task
from dge_utils import DGEAdamW, MoEGatedLinear, HybridGate


def collect_gate_stats(model):
    """Collect statistics about gate activations across all layers."""
    stats = {
        'gate_biases': [],
        'gate_open_probs': [],
        'router_weight_norms': []
    }
    
    for name, module in model.named_modules():
        if isinstance(module, HybridGate):
            if module.router is not None:
                # Get router output layer
                router = module.router
                if isinstance(router, nn.Sequential):
                    router = router[-1]
                    
                if hasattr(router, 'bias') and router.bias is not None:
                    bias = router.bias.detach()
                    stats['gate_biases'].extend(bias.tolist())
                    # Probability of being open = sigmoid(bias)
                    open_probs = torch.sigmoid(bias)
                    stats['gate_open_probs'].extend(open_probs.tolist())
                    
                if hasattr(router, 'weight') and router.weight is not None:
                    stats['router_weight_norms'].append(router.weight.norm().item())
                    
    return stats


def collect_gradient_stats(model):
    """Collect gradient statistics per quadrant."""
    stats = {
        'frozen_grad_norm': 0.0,
        'active_grad_norm': 0.0,
        'gate_grad_norm': 0.0,
        'q_tr_grad': 0.0,  # Core: Old→Old
        'q_br_grad': 0.0,  # Synergy: Old→New
        'q_bl_grad': 0.0,  # Capacity: New→New
        'q_tl_grad': 0.0,  # Firewall: New→Old
        'embedding_new_grad': 0.0,
        'lm_head_new_grad': 0.0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, MoEGatedLinear):
            if module.weight.grad is None:
                continue
                
            grad = module.weight.grad
            mask = module.backward_mask
            old_out = int((mask[:, 0] == 0).sum().item())  # Count frozen rows
            old_in = int((mask[0, :] == 0).sum().item())   # Count frozen cols
            
            # Per-quadrant gradients
            if old_out > 0 and old_in > 0:
                q_tr = grad[:old_out, :old_in]
                stats['q_tr_grad'] += q_tr.norm().item() ** 2
                
            if old_out > 0 and grad.shape[1] > old_in:
                q_tl = grad[:old_out, old_in:]
                stats['q_tl_grad'] += q_tl.norm().item() ** 2
                
            if grad.shape[0] > old_out and old_in > 0:
                q_br = grad[old_out:, :old_in]
                stats['q_br_grad'] += q_br.norm().item() ** 2
                
            if grad.shape[0] > old_out and grad.shape[1] > old_in:
                q_bl = grad[old_out:, old_in:]
                stats['q_bl_grad'] += q_bl.norm().item() ** 2
                
            # Aggregate frozen/active
            frozen_grad = grad * (1 - mask)
            active_grad = grad * mask
            stats['frozen_grad_norm'] += frozen_grad.norm().item() ** 2
            stats['active_grad_norm'] += active_grad.norm().item() ** 2
            
            # Gate gradients
            for gate in [module.gate_row, module.gate_col]:
                if gate is not None and hasattr(gate, 'router') and gate.router is not None:
                    router = gate.router
                    if isinstance(router, nn.Sequential):
                        for layer in router:
                            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                                stats['gate_grad_norm'] += layer.weight.grad.norm().item() ** 2
                    elif hasattr(router, 'weight') and router.weight.grad is not None:
                        stats['gate_grad_norm'] += router.weight.grad.norm().item() ** 2
    
    # Check embedding
    if hasattr(model, 'token_emb') and model.token_emb.weight.grad is not None:
        emb_grad = model.token_emb.weight.grad
        old_d = 64  # Original d_model
        if emb_grad.shape[1] > old_d:
            new_part = emb_grad[:, old_d:]
            stats['embedding_new_grad'] = new_part.norm().item()
            
    # Check lm_head
    if hasattr(model, 'lm_head') and model.lm_head.weight.grad is not None:
        head_grad = model.lm_head.weight.grad
        old_in = 64
        if head_grad.shape[1] > old_in:
            new_cols = head_grad[:, old_in:]
            stats['lm_head_new_grad'] = new_cols.norm().item()
    
    # Convert to norms
    stats['frozen_grad_norm'] = stats['frozen_grad_norm'] ** 0.5
    stats['active_grad_norm'] = stats['active_grad_norm'] ** 0.5
    stats['gate_grad_norm'] = stats['gate_grad_norm'] ** 0.5
    stats['q_tr_grad'] = stats['q_tr_grad'] ** 0.5
    stats['q_br_grad'] = stats['q_br_grad'] ** 0.5
    stats['q_bl_grad'] = stats['q_bl_grad'] ** 0.5
    stats['q_tl_grad'] = stats['q_tl_grad'] ** 0.5
    
    return stats


def collect_weight_stats(model):
    """Collect weight magnitude statistics."""
    stats = {
        'q_tr_weight_norm': 0.0,
        'q_br_weight_norm': 0.0,
        'q_bl_weight_norm': 0.0,
        'q_tl_weight_norm': 0.0,
        'embedding_new_norm': 0.0,
        'lm_head_new_norm': 0.0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, MoEGatedLinear):
            weight = module.weight.detach()
            mask = module.backward_mask
            old_out = int((mask[:, 0] == 0).sum().item())
            old_in = int((mask[0, :] == 0).sum().item())
            
            if old_out > 0 and old_in > 0:
                stats['q_tr_weight_norm'] += weight[:old_out, :old_in].norm().item() ** 2
            if old_out > 0 and weight.shape[1] > old_in:
                stats['q_tl_weight_norm'] += weight[:old_out, old_in:].norm().item() ** 2
            if weight.shape[0] > old_out and old_in > 0:
                stats['q_br_weight_norm'] += weight[old_out:, :old_in].norm().item() ** 2
            if weight.shape[0] > old_out and weight.shape[1] > old_in:
                stats['q_bl_weight_norm'] += weight[old_out:, old_in:].norm().item() ** 2
    
    # Embedding
    if hasattr(model, 'token_emb'):
        emb = model.token_emb.weight.detach()
        old_d = 64
        if emb.shape[1] > old_d:
            stats['embedding_new_norm'] = emb[:, old_d:].norm().item()
            
    # LM Head
    if hasattr(model, 'lm_head'):
        head = model.lm_head.weight.detach()
        old_in = 64
        if head.shape[1] > old_in:
            stats['lm_head_new_norm'] = head[:, old_in:].norm().item()
    
    # Convert
    for k in ['q_tr_weight_norm', 'q_br_weight_norm', 'q_bl_weight_norm', 'q_tl_weight_norm']:
        stats[k] = stats[k] ** 0.5
        
    return stats


def run_diagnostic_experiment():
    """Run experiment with detailed diagnostics."""
    print("=" * 70)
    print("DIAGNOSTIC EXPERIMENT: Dead Sidecar Investigation")
    print("=" * 70)
    
    vocab_size = 500
    d_model = 64
    n_layer = 2
    n_head = 4
    
    # ===== PHASE 1: Train Task A =====
    print("\n[PHASE 1] Training Task A (COUNT_UP)...")
    model = DGESimpleTransformer(vocab_size, d_model, n_layer, n_head)
    optimizer = DGEAdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    for i in range(500):
        x, y = generate_batch(TaskType.COUNT_UP, vocab_size, 32, 32)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
    acc_a = evaluate_task(model, TaskType.COUNT_UP, vocab_size)
    print(f"Task A Accuracy: {acc_a:.1f}%")
    
    # ===== PHASE 2: Expand =====
    print("\n[PHASE 2] Expanding model with Directed Synergy...")
    added_width = 32
    model.expand_model(
        new_input_dim=d_model + added_width,
        new_output_dim=vocab_size,
        router_type='mlp',
        use_orthogonal_init=False,
        use_gradient_rescue=True,
        isolate_cross_terms=False,
        router_init_bias=0.0
    )
    
    # Collect pre-training stats
    print("\n[DIAGNOSTICS] Pre-Training Stats:")
    gate_stats = collect_gate_stats(model)
    weight_stats = collect_weight_stats(model)
    
    print(f"  Gate Bias Range: [{min(gate_stats['gate_biases']):.2f}, {max(gate_stats['gate_biases']):.2f}]")
    print(f"  Gate Open Prob Range: [{min(gate_stats['gate_open_probs']):.2f}, {max(gate_stats['gate_open_probs']):.2f}]")
    print(f"  Mean Gate Open Prob: {sum(gate_stats['gate_open_probs'])/len(gate_stats['gate_open_probs']):.4f}")
    print(f"  Weight Norms:")
    print(f"    Q_TR (Core): {weight_stats['q_tr_weight_norm']:.4f}")
    print(f"    Q_BR (Synergy): {weight_stats['q_br_weight_norm']:.4f}")
    print(f"    Q_BL (Capacity): {weight_stats['q_bl_weight_norm']:.4f}")
    print(f"    Q_TL (Firewall): {weight_stats['q_tl_weight_norm']:.6f}")
    print(f"    Embedding New: {weight_stats['embedding_new_norm']:.4f}")
    print(f"    LM Head New: {weight_stats['lm_head_new_norm']:.4f}")
    
    # Re-init optimizer
    optimizer = DGEAdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # ===== PHASE 3: Train Task B with Diagnostics =====
    print("\n[PHASE 3] Training Task B (COUNT_DOWN) with Diagnostics...")
    print("-" * 70)
    print(f"{'Step':>5} | {'Loss':>8} | {'Active_G':>10} | {'Gate_G':>10} | {'Q_BR_G':>10} | {'Q_BL_G':>10} | {'Probe_A':>8}")
    print("-" * 70)
    
    for i in range(500):
        x, y = generate_batch(TaskType.COUNT_DOWN, vocab_size, 32, 32)
        optimizer.zero_grad()
        _, loss = model(x, y, sparsity_lambda=0.05)
        loss.backward()
        
        if i % 50 == 0 or i == 499:
            grad_stats = collect_gradient_stats(model)
            acc_a_probe = evaluate_task(model, TaskType.COUNT_UP, vocab_size)
            
            print(f"{i:5d} | {loss.item():8.4f} | {grad_stats['active_grad_norm']:10.2e} | "
                  f"{grad_stats['gate_grad_norm']:10.2e} | {grad_stats['q_br_grad']:10.2e} | "
                  f"{grad_stats['q_bl_grad']:10.2e} | {acc_a_probe:7.1f}%")
        
        optimizer.step()
    
    print("-" * 70)
    
    # ===== FINAL EVALUATION =====
    print("\n[FINAL] Evaluation:")
    acc_a_final = evaluate_task(model, TaskType.COUNT_UP, vocab_size)
    acc_b_final = evaluate_task(model, TaskType.COUNT_DOWN, vocab_size)
    
    print(f"  Task A Accuracy: {acc_a_final:.1f}%")
    print(f"  Task B Accuracy: {acc_b_final:.1f}%")
    
    print("\n[FINAL] Gate Stats:")
    gate_stats = collect_gate_stats(model)
    print(f"  Gate Bias Range: [{min(gate_stats['gate_biases']):.2f}, {max(gate_stats['gate_biases']):.2f}]")
    print(f"  Mean Gate Open Prob: {sum(gate_stats['gate_open_probs'])/len(gate_stats['gate_open_probs']):.4f}")
    
    print("\n[FINAL] Weight Stats:")
    weight_stats = collect_weight_stats(model)
    print(f"  Q_TR (Core): {weight_stats['q_tr_weight_norm']:.4f}")
    print(f"  Q_BR (Synergy): {weight_stats['q_br_weight_norm']:.4f}")
    print(f"  Q_BL (Capacity): {weight_stats['q_bl_weight_norm']:.4f}")
    print(f"  Q_TL (Firewall): {weight_stats['q_tl_weight_norm']:.6f}")
    
    # ===== DIAGNOSIS =====
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)
    
    # Check gate openness
    mean_open = sum(gate_stats['gate_open_probs']) / len(gate_stats['gate_open_probs'])
    if mean_open < 0.5:
        print(f"[!] Gates are CLOSED (mean open prob = {mean_open:.4f})")
        print("    -> router_init_bias may not be working as expected")
    else:
        print(f"[OK] Gates are OPEN (mean open prob = {mean_open:.4f})")
        
    # Check Q_BR gradients
    if grad_stats['q_br_grad'] < 1e-6:
        print(f"[!] Q_BR (Synergy) gradients are DEAD ({grad_stats['q_br_grad']:.2e})")
        print("    -> Signal not flowing from Old to New")
    else:
        print(f"[OK] Q_BR (Synergy) has gradients ({grad_stats['q_br_grad']:.2e})")
        
    # Check embedding
    if weight_stats['embedding_new_norm'] < 0.01:
        print(f"[!] Embedding New dimensions are nearly ZERO ({weight_stats['embedding_new_norm']:.4f})")
        print("    -> New tokens have no signal at the start")
    else:
        print(f"[OK] Embedding New has signal ({weight_stats['embedding_new_norm']:.4f})")
        
    # Check LM Head
    if weight_stats['lm_head_new_norm'] < 0.01:
        print(f"[!] LM Head new columns are nearly ZERO ({weight_stats['lm_head_new_norm']:.4f})")
        print("    -> Head cannot READ from new dimensions")
    else:
        print(f"[OK] LM Head can read new dims ({weight_stats['lm_head_new_norm']:.4f})")
    
    print("=" * 70)
    
    return {
        'acc_a': acc_a_final,
        'acc_b': acc_b_final,
        'mean_gate_open': mean_open,
        'q_br_grad': grad_stats['q_br_grad'],
        'weight_stats': weight_stats
    }


if __name__ == "__main__":
    run_diagnostic_experiment()
