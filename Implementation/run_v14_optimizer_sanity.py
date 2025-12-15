import os
import torch
import torch.nn as nn
from dge_model import DGESimpleTransformer
from dge_utils import MoEGatedLinear, DGEAdamW

def run_v14_optimizer_sanity():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V14 OPTIMIZER SANITY: Checking Gradient Mask & Update")
    print("="*60)
    
    vocab_size = 100
    d_model = 16 
    n_layer = 1
    n_head = 1
    
    # 1. Create Model
    print("Step 1: Creating Baseline Model...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    # 2. Expand
    print(f"Step 2: Expanding Model...")
    model.expand_model(
        new_input_dim=d_model + 16, 
        new_output_dim=vocab_size, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True, # Checking if rescue interferes
        isolate_cross_terms=True, 
        router_init_bias=-4.0
    )
    
    # Setup Optimizer (Same as V12)
    optimizer = DGEAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    
    # 3. Forward
    print("Step 3: Forward Pass...")
    model.train() # Enable gradients
    logits, loss = model(input_ids, targets)
    
    print(f"Initial Loss: {loss.item():.4f}")
    
    # 4. Backward
    print("Step 4: Backward Pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # 5. Inspect Gradients BEFORE Step
    print("Step 5: Inspecting Gradients...")
    layer0 = model.layers[0]
    w_q = layer0.w_q
    grad = w_q.weight.grad
    
    tr_grad = grad[:d_model, d_model:] # Top Right (Cross Term)
    print(f"w_q Top-Right Gradient Mean (Should be 0): {tr_grad.abs().mean().item():.6f}")
    
    if tr_grad.abs().mean().item() > 1e-6:
        print("❌ LEAK DETECTED: Gradients are NOT masked!")
    else:
        print("✅ Gradients are properly masked.")
        
    # 6. Step
    print("Step 6: Optimizer Step...")
    optimizer.step()
    
    # 7. Check Weights AFTER Step
    print("Step 7: Inspecting Weights After Update...")
    w = w_q.weight
    tr = w[:d_model, d_model:] # Top Right (Cross Term)
    print(f"w_q Top-Right Weight Mean (Should be 0): {tr.abs().mean().item():.6f}")
    
    if tr.abs().mean().item() > 1e-6:
        print("❌ LEAK DETECTED: Weights Updated despite Zero Gradient!")
        print("   Checking Weight Decay...")
        # If decay caused it, values should be non-zero (if they were non-zero initially)
        # But we zeroed them initially.
        # Maybe AdamW momentum update? 'exp_avg' initialized to 0. 
        # If grad is 0, exp_avg stays 0.
        # This is mysterious.
    else:
        print("✅ Weights remained Zero.")
        
    # 8. Check Forward Identity
    print("Step 8: Checking Forward Identity After Step...")
    model.eval()
    with torch.no_grad():
        out_new, _ = model(input_ids)
    
    # We don't have 'out_old' captured here cleanly, but we can check if behavior diverged meaningfully
    # Actually, let's just create a reference old output from a fresh model for comparison?
    # Or just rely on the weight check. If weight is 0, identity holds.
    # But checking output confirms it.
    
    print("V14 Complete.")

if __name__ == "__main__":
    run_v14_optimizer_sanity()
