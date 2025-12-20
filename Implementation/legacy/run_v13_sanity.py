import os
import torch
import torch.nn as nn
from core.model import DGESimpleTransformer
from core.utils import MoEGatedLinear

def run_v13_sanity():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V13 SANITY: Probing for Leaks")
    print("="*60)
    
    vocab_size = 100
    d_model = 16 # Small model for easy inspection
    n_layer = 1
    n_head = 1
    
    # 1. Create Model
    print("Step 1: Creating Baseline Model...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    model.eval()
    
    # 2. Freeze and Capture Output
    input_ids = torch.tensor([[1, 2, 3]]) # [1, 3]
    with torch.no_grad():
        out_old, _ = model(input_ids)
    
    print(f"Old Output Mean: {out_old.mean().item():.6f}")
    
    # 3. Expand
    print(f"Step 2: Expanding Model...")
    model.expand_model(
        new_input_dim=d_model + 16, 
        new_output_dim=vocab_size, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=True, # Should zero cross terms
        router_init_bias=-4.0
    )
    
    # 4. Inspect Weights
    print("Step 3: Inspecting Weights for Zero Cross-Terms...")
    layer0 = model.layers[0]
    # Check w_q (Separate QKV enabled)
    if hasattr(layer0, 'w_q'):
        w = layer0.w_q.weight
        tr = w[:d_model, d_model:] # Top Right
        bl = w[d_model:, :d_model] # Bottom Left
        print(f"w_q Top-Right Mean (Should be 0): {tr.abs().mean().item():.6f}")
        print(f"w_q Bottom-Left Mean (Should be 0): {bl.abs().mean().item():.6f}")
        if tr.abs().mean().item() > 1e-6:
            print("❌ LEAK DETECTED: Cross-Terms are NOT zero!")
        else:
            print("✅ Cross-Terms are clean.")
            
    # 5. Run Forward with Expanded Model
    print("Step 4: Running Forward on Expanded Model...")
    with torch.no_grad():
        out_new, _ = model(input_ids)
        
    # Check if old part matches
    # Output is logits [B, T, Vocab]. 
    # Vocab size didn't change (we passed new_output_dim=vocab_size)
    diff = (out_old - out_new).abs().max().item()
    print(f"Max Diff between Old and New Output: {diff:.6f}")
    
    if diff < 1e-5:
        print("✅ PASS: Model Expansion is Identity Preserving.")
    else:
        print("❌ FAIL: Functional Interference detected.")
        print(f"Old: {out_old[0,0,:5]}")
        print(f"New: {out_new[0,0,:5]}")

if __name__ == "__main__":
    run_v13_sanity()
