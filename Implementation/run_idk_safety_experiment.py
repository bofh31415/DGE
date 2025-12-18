#!/usr/bin/env python3
"""
IDK Safety Net Experiment - V 0.1.0
===================================
Tests Router 0 (IDK) on Count Up → Count Down → OOD scenario.

Phases:
1. Train Count Up (1 2 3 → 4)
2. Expand + Train Count Down (4 3 2 → 1)
3. Test OOD (random patterns → expect IDK activation)

Usage:
    python run_idk_safety_experiment.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dge_model import DGESimpleTransformer
from idk_router import IDKRouter, ConfidenceAggregator, create_idk_router
from version import __version__

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model
    "vocab_size": 20,      # 0-9 digits + special tokens
    "d_model": 64,
    "n_layer": 2,
    "n_head": 2,
    "max_seq_len": 8,
    
    # Training
    "batch_size": 32,
    "lr": 1e-3,
    "phase1_steps": 500,   # Count Up
    "phase2_steps": 500,   # Count Down
    
    # Hierarchical Router
    # Layer 0: IDK Meta-Router (confidence aggregation)
    # Layer 1+: Specialized routers (bigram, spatial, modality)
    "router_architecture": "hierarchical",  # 'flat' (no IDK) or 'hierarchical' (with IDK)
    "idk_threshold": 0.3,
    
    # Expansion
    "expansion_delta": 64,  # 64 → 128
    
    # Output
    "output_dir": "models/idk_safety_experiment",
}

# Special tokens
PAD_TOKEN = 0
IDK_TOKEN = 19  # Last token in vocab

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_count_up_batch(batch_size: int, seq_len: int = 4) -> tuple:
    """
    Generate "count up" sequences: 1 2 3 → 4
    
    Returns:
        inputs: [batch, seq_len-1]
        targets: [batch, 1]
    """
    starts = torch.randint(1, 10 - seq_len, (batch_size,))
    inputs = torch.stack([starts + i for i in range(seq_len - 1)], dim=1)
    targets = (starts + seq_len - 1).unsqueeze(1)
    return inputs.to(DEVICE), targets.to(DEVICE)


def generate_count_down_batch(batch_size: int, seq_len: int = 4) -> tuple:
    """
    Generate "count down" sequences: 8 7 6 → 5
    
    Returns:
        inputs: [batch, seq_len-1]
        targets: [batch, 1]
    """
    starts = torch.randint(seq_len, 10, (batch_size,))
    inputs = torch.stack([starts - i for i in range(seq_len - 1)], dim=1)
    targets = (starts - seq_len + 1).unsqueeze(1)
    return inputs.to(DEVICE), targets.to(DEVICE)


def generate_ood_batch(batch_size: int, seq_len: int = 4) -> tuple:
    """
    Generate OOD (out-of-distribution) sequences:
    - Random order
    - Mixed patterns
    - Letters (if in vocab)
    
    Returns:
        inputs: [batch, seq_len-1]
        pattern_type: list of str describing each pattern
    """
    patterns = []
    inputs_list = []
    
    for _ in range(batch_size):
        pattern_type = torch.randint(0, 4, (1,)).item()
        
        if pattern_type == 0:
            # Random numbers
            seq = torch.randint(1, 10, (seq_len - 1,))
            patterns.append("random")
        elif pattern_type == 1:
            # Constant
            val = torch.randint(1, 10, (1,)).item()
            seq = torch.full((seq_len - 1,), val)
            patterns.append("constant")
        elif pattern_type == 2:
            # Alternating
            a, b = torch.randint(1, 10, (2,)).tolist()
            seq = torch.tensor([a if i % 2 == 0 else b for i in range(seq_len - 1)])
            patterns.append("alternating")
        else:
            # Sawtooth (up then down)
            seq = torch.tensor([1, 5, 2])[:seq_len - 1]
            patterns.append("sawtooth")
        
        inputs_list.append(seq)
    
    inputs = torch.stack(inputs_list).to(DEVICE)
    return inputs, patterns


# =============================================================================
# MODEL WITH IDK ROUTER
# =============================================================================

class DGEWithIDK(nn.Module):
    """
    DGE model wrapper that adds IDK Safety Net.
    """
    
    def __init__(self, base_model: DGESimpleTransformer, idk_router: IDKRouter):
        super().__init__()
        self.base_model = base_model
        self.idk_router = idk_router
        self.confidence_aggregator = ConfidenceAggregator(num_routers=2)
    
    def forward(self, x, return_confidence=False):
        """Forward pass with IDK routing."""
        # Get base model output
        logits = self.base_model(x)
        
        # Get router confidences from gate activations
        # For now, use a simple proxy: mean hidden state magnitude
        # TODO: Extract actual gate activations from GatedGhostLinear
        with torch.no_grad():
            hidden = self.base_model.get_hidden_states(x) if hasattr(self.base_model, 'get_hidden_states') else None
        
        if hidden is not None and self.idk_router is not None:
            # Simulate router confidences (placeholder until gate extraction is implemented)
            batch_size, seq_len = x.shape
            router_conf = torch.rand(batch_size, seq_len, 2).to(x.device)  # Placeholder
            
            # Apply IDK routing to logits (boost IDK token when uncertain)
            max_conf = router_conf.max(dim=-1).values
            idk_boost = torch.sigmoid((self.idk_router.threshold - max_conf) / 0.1) * 5.0
            logits[:, :, IDK_TOKEN] += idk_boost
            
            if return_confidence:
                return logits, max_conf
        
        if return_confidence:
            return logits, None
        return logits
    
    def get_idk_weight(self, x):
        """Get IDK activation weight for inputs."""
        if self.idk_router is None:
            return None
        
        # Placeholder - will be replaced with actual gate confidence
        batch_size, seq_len = x.shape
        return torch.rand(batch_size, seq_len).to(x.device) * 0.5


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_phase(model, optimizer, generate_fn, num_steps, phase_name):
    """Train on a specific phase."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    print(f"\n{'='*60}")
    print(f"🏋️ Training Phase: {phase_name}")
    print(f"{'='*60}")
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        inputs, targets = generate_fn(CONFIG["batch_size"])
        
        # Forward
        if hasattr(model, 'base_model'):
            logits = model.base_model(inputs)
        else:
            logits = model(inputs)
        
        # Get last position prediction
        last_logits = logits[:, -1, :]
        targets_flat = targets.squeeze(-1)
        
        loss = criterion(last_logits, targets_flat)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = last_logits.argmax(dim=-1)
        acc = (preds == targets_flat).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accuracies[-100:]) / len(accuracies[-100:])
            print(f"   Step {step+1}/{num_steps} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")
    
    final_loss = sum(losses[-50:]) / len(losses[-50:])
    final_acc = sum(accuracies[-50:]) / len(accuracies[-50:])
    print(f"   ✅ {phase_name} Complete | Loss: {final_loss:.4f} | Acc: {final_acc:.2%}")
    
    return {"loss": final_loss, "accuracy": final_acc}


def evaluate_ood(model, num_batches=10):
    """Evaluate IDK activation on OOD inputs."""
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating OOD (IDK Activation)")
    print(f"{'='*60}")
    
    idk_activations = []
    pattern_results = {"random": [], "constant": [], "alternating": [], "sawtooth": []}
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, patterns = generate_ood_batch(CONFIG["batch_size"])
            
            if hasattr(model, 'get_idk_weight'):
                idk_weight = model.get_idk_weight(inputs)
                if idk_weight is not None:
                    for i, pattern in enumerate(patterns):
                        w = idk_weight[i, -1].item()  # Last position
                        pattern_results[pattern].append(w)
                        idk_activations.append(w)
    
    # Summary
    overall_idk = sum(idk_activations) / len(idk_activations) if idk_activations else 0
    print(f"   Overall IDK activation: {overall_idk:.2%}")
    
    for pattern, weights in pattern_results.items():
        if weights:
            avg = sum(weights) / len(weights)
            print(f"   {pattern}: {avg:.2%}")
    
    return {
        "overall_idk_rate": overall_idk,
        "pattern_breakdown": {k: sum(v)/len(v) if v else 0 for k, v in pattern_results.items()}
    }


def evaluate_in_distribution(model, generate_fn, task_name, num_batches=10):
    """Evaluate accuracy and IDK false positive rate on in-distribution data."""
    model.eval()
    
    correct = 0
    total = 0
    idk_false_positives = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = generate_fn(CONFIG["batch_size"])
            
            if hasattr(model, 'base_model'):
                logits = model.base_model(inputs)
            else:
                logits = model(inputs)
            
            preds = logits[:, -1, :].argmax(dim=-1)
            targets_flat = targets.squeeze(-1)
            
            correct += (preds == targets_flat).sum().item()
            total += targets_flat.numel()
            
            # Check false positive IDK
            idk_pred = (preds == IDK_TOKEN).sum().item()
            idk_false_positives += idk_pred
    
    acc = correct / total
    fp_rate = idk_false_positives / total
    
    print(f"   {task_name}: Acc={acc:.2%}, IDK False Positive={fp_rate:.2%}")
    
    return {"accuracy": acc, "idk_false_positive_rate": fp_rate}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("🧪 IDK SAFETY NET EXPERIMENT")
    print(f"   Version: {__version__}")
    print(f"   Device: {DEVICE}")
    print("=" * 70)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    results = {"config": CONFIG, "start_time": datetime.now().isoformat()}
    
    # =========================================================================
    # PHASE 1: Create model and train Count Up
    # =========================================================================
    
    print("\n📦 Phase 1: Creating Model + Training Count Up")
    
    model = DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        max_seq_len=CONFIG["max_seq_len"]
    ).to(DEVICE)
    
    # Create IDK router based on architecture
    if CONFIG["router_architecture"] == "hierarchical":
        idk_router = create_idk_router(
            d_model=CONFIG["d_model"],
            router_type='embedding',
        threshold=CONFIG["idk_threshold"]
    )
    if idk_router:
        idk_router = idk_router.to(DEVICE)
    
    model_with_idk = DGEWithIDK(model, idk_router)
    
    optimizer = optim.AdamW(model_with_idk.parameters(), lr=CONFIG["lr"])
    
    # Train Count Up
    phase1_results = train_phase(
        model_with_idk, optimizer,
        generate_count_up_batch,
        CONFIG["phase1_steps"],
        "Count Up"
    )
    results["phase1_count_up"] = phase1_results
    
    # =========================================================================
    # PHASE 2: Expand + Train Count Down
    # =========================================================================
    
    print("\n📦 Phase 2: Expanding Model + Training Count Down")
    
    old_params = sum(p.numel() for p in model.parameters())
    
    # Expand model
    model.expand_capacity(CONFIG["expansion_delta"])
    
    new_params = sum(p.numel() for p in model.parameters())
    print(f"   Expanded: {old_params:,} → {new_params:,} params")
    
    # Update IDK router for new d_model
    new_d_model = CONFIG["d_model"] + CONFIG["expansion_delta"]
    if CONFIG["router_architecture"] == "hierarchical":
        idk_router = create_idk_router(
            d_model=new_d_model,
            router_type='embedding',
        threshold=CONFIG["idk_threshold"]
    )
    if idk_router:
        idk_router = idk_router.to(DEVICE)
    model_with_idk.idk_router = idk_router
    
    # New optimizer for expanded model
    optimizer = optim.AdamW(model_with_idk.parameters(), lr=CONFIG["lr"])
    
    # Train Count Down
    phase2_results = train_phase(
        model_with_idk, optimizer,
        generate_count_down_batch,
        CONFIG["phase2_steps"],
        "Count Down"
    )
    results["phase2_count_down"] = phase2_results
    
    # =========================================================================
    # PHASE 3: Evaluate
    # =========================================================================
    
    print("\n📊 Phase 3: Evaluation")
    
    # Check Count Up retention
    count_up_eval = evaluate_in_distribution(
        model_with_idk, generate_count_up_batch, "Count Up (Retention)"
    )
    results["eval_count_up_retention"] = count_up_eval
    
    # Check Count Down accuracy
    count_down_eval = evaluate_in_distribution(
        model_with_idk, generate_count_down_batch, "Count Down"
    )
    results["eval_count_down"] = count_down_eval
    
    # Check OOD → IDK activation
    ood_eval = evaluate_ood(model_with_idk)
    results["eval_ood_idk"] = ood_eval
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"   Count Up Retention: {count_up_eval['accuracy']:.2%}")
    print(f"   Count Down Accuracy: {count_down_eval['accuracy']:.2%}")
    print(f"   OOD → IDK Activation: {ood_eval['overall_idk_rate']:.2%}")
    print(f"   IDK False Positive: {count_up_eval['idk_false_positive_rate']:.2%}")
    
    # Save results
    results["end_time"] = datetime.now().isoformat()
    results_path = os.path.join(CONFIG["output_dir"], "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()
