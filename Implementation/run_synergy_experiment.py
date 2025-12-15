import os
import torch
import math
from datetime import datetime
from dge_model import DGESimpleTransformer
from dge_training import train_task, TaskType
from dge_utils import DGEAdamW
from dge_logger import DGELogger

def generate_samples(model, task_type, vocab_size, count=10):
    """
    Generates text samples to qualitatively verify skill retention/acquisition.
    """
    model.eval()
    print(f"\n--- Generating 10 Samples for {task_type.name} ---")
    correct_count = 0
    with torch.no_grad():
        for i in range(count):
            # 1. Create a prompt
            if task_type == TaskType.COUNT_UP:
                # Prompt: [n, n+1, n+2] -> Expect n+3
                start = torch.randint(0, vocab_size - 5, (1, 1)).item()
                prompt = torch.tensor([[start, start+1, start+2]], device=model.token_emb.weight.device) # [1, 3]
                expected = start + 3
            elif task_type == TaskType.COUNT_DOWN:
                # Prompt: [n, n-1, n-2] -> Expect n-3
                start = torch.randint(5, vocab_size, (1, 1)).item()
                prompt = torch.tensor([[start, start-1, start-2]], device=model.token_emb.weight.device)
                expected = start - 3
            
            # 2. Forward
            logits, _ = model(prompt) # [1, 3, V]
            next_token_logits = logits[0, -1, :]
            pred = next_token_logits.argmax().item()
            
            # 3. Check
            is_correct = (pred == expected)
            if is_correct:
                correct_count += 1
            
            print(f"Sample {i+1}: Input {prompt.tolist()[0]} -> Pred {pred} (Expected {expected}) {'✅' if is_correct else '❌'}")
            
    print(f"Accuracy on 10 samples: {correct_count}/10")
    model.train()

def run_synergy_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Naming convention: 'synergy' + timestamp
    model_name = f"dge_synergy_v1_{timestamp}"
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    exp_dir = os.path.join(models_dir, model_name)
    
    # Initialize Logger
    logger = DGELogger(exp_dir)
    
    # Defaults (Small Model for Speed)
    vocab_size = 500
    d_model = 64 
    n_layer = 2
    n_head = 4
    
    print(f"Creating model {model_name}...")
    print(f"Goal: Prove 'Directed Synergy' (Old->New) allows learning Task B while freezing Core.")
    
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    logger.log_event("CREATED", {"d_model": d_model}, step=0)
    
    # Optimizer
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    global_step = 0
    
    # --- PHASE 1: SEED FUND (Training Task A) ---
    # Task A = COUNT_UP (Proxy for TinyStories)
    print("\n=== Phase 1: Training Task A (Core Identity) ===")
    global_step = train_task(
        model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer
    )
    
    # Verify A
    generate_samples(model, TaskType.COUNT_UP, vocab_size)
    generate_samples(model, TaskType.COUNT_DOWN, vocab_size) # Should fail
    
    # --- PHASE 2: EXPANSION & FREEZE ---
    print("\n=== Phase 2: Freezing Core & Expanding for Synergy ===")
    current_d_model = model.d_model
    
    # This call uses the NEW default in dge_utils.py:
    # isolate_cross_terms=False (Forced)
    # router_init_bias=0.0 (Forced)
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp', # Contextual Router to help separation
        use_orthogonal_init=True, # Safety default
        use_gradient_rescue=True, # Safety default (though less needed if open)
        isolate_cross_terms=False # EXPLICIT: Enable Old->New Synergy
    )
    
    # --- FORENSIC CHECK FUNCTION ---
    def verify_firewall_integrity(model, step_label):
        print(f"\n--- Forensic Integrity Check ({step_label}) ---")
        integrity_pass = True
        for name, module in model.named_modules():
            if "MoEGatedLinear" in module.__class__.__name__:
                # W shape: [out, in]
                w = module.weight
                old_out = module.gate_col.old_count if hasattr(module, 'gate_col') else 0 # Approximate
                # Actually, we need exact old_in/out dims.
                # MoEGatedLinear doesn't store them explicitly, but HybridGate does 'old_count'.
                # gate_col is input (old_in). gate_row is output (old_out).
                if hasattr(module, 'gate_col') and hasattr(module, 'gate_row'):
                    old_in = module.gate_col.old_count
                    old_out = module.gate_row.old_count
                    
                    # 1. Check Firewall (Top-Right: New In -> Old Out)
                    # Rows: :old_out. Cols: old_in:
                    firewall_block = w[:old_out, old_in:]
                    fw_norm = firewall_block.norm().item()
                    fw_max = firewall_block.abs().max().item()
                    
                    if fw_max > 1e-6:
                        print(f"❌ LEAKAGE DETECTED in {name}: Firewall (New->Old) Max {fw_max:.6f}, Norm {fw_norm:.6f}")
                        integrity_pass = False
                    else:
                        print(f"✅ Firewall Safe in {name}: Max {fw_max:.6e}")
                        
                    # 2. Check Frozen Core (Top-Left)
                    # Ideally we compare against a saved checkpoint, but here we just check if it's not crazy.
                    # This check is weak without reference. Skipped.
                    
        return integrity_pass

    verify_firewall_integrity(model, "Post-Expansion")
    
    # Re-init Optimizer for new params
    print("Re-initializing Optimizer...")
    router_params = [p for n, p in model.named_parameters() if 'router' in n or 'gate' in n]
    default_params = [p for n, p in model.named_parameters() if 'router' not in n and 'gate' not in n]
    
    # Use standard LR for both
    optimizer = DGEAdamW([
        {'params': default_params, 'lr': 1e-3}, 
        {'params': router_params, 'lr': 1e-3}
    ], weight_decay=0.0)
    
    # --- PHASE 3: SYNERGY TEST (Training Task B) ---
    # Task B = COUNT_DOWN (Proxy for BBC News)
    # We probe Task A to measure Stability.
    print("\n=== Phase 3: Training Task B (Directed Synergy) ===")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
    )
    
    print("\n=== Phase 3 Verification ===")
    print("Checking Stability (Skill A):")
    generate_samples(model, TaskType.COUNT_UP, vocab_size)
    print("Checking Plasticity (Skill B):")
    generate_samples(model, TaskType.COUNT_DOWN, vocab_size)
    
    print(f"Experiment Complete. Log: {exp_dir}")
    print("Check training.csv for:")
    print("1. Loss (Task B): Should go low (<1.0) -> Plasticity Success")
    print("2. Probe_Loss (Task A): Should stay low (<1.0) -> Stability Success")
    
    print("\nFinal Firewall Check:")
    verify_firewall_integrity(model, "End-Experiment")

if __name__ == "__main__":
    run_synergy_experiment()
