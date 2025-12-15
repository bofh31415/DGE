import os
import torch
import torch.nn as nn
from dge_model import DGESimpleTransformer
from dge_utils import DGEAdamW

def run_v15_embedding_leak():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V15 EMBEDDING LEAK CHECK")
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
        use_gradient_rescue=False, 
        isolate_cross_terms=False, # DEBUG: Disable isolation to check flow
        router_init_bias=-4.0
    )
    
    optimizer = DGEAdamW(model.parameters(), lr=1e-2)
    
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    
    # 3. Forward & Backward
    print("Step 3: Forward & Backward...")
    model.train()
    logits, loss = model(input_ids, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Inspect Embedding Gradients
    print("Step 4: Inspecting Embedding Gradients...")
    token_emb = model.token_emb
    grad = token_emb.weight.grad
    
    import sys
    print(f"Token Emb Requires Grad: {token_emb.weight.requires_grad}")
    sys.stdout.flush()
    
    layer0 = model.layers[0]
    w_q = layer0.w_q
    
    if w_q.weight.grad is None:
        print("Layer0 w_q Grad is NONE!")
    else:
        print(f"Layer0 w_q Grad Mean: {w_q.weight.grad.abs().mean().item():.9f}")
    sys.stdout.flush()
    
    if hasattr(w_q, 'last_rescue_scale'):
        print(f"Layer0 w_q Rescue Scale: {w_q.last_rescue_scale}")
    if hasattr(w_q.gate_row, 'last_mean_open'):
        print(f"Layer0 w_q Gate Row Open: {w_q.gate_row.last_mean_open}")
    
    # Old Dimensions: [:, :d_model]
    # New Dimensions: [:, d_model:]
    old_grad = grad[:, :d_model]
    new_grad = grad[:, d_model:]
    
    old_grad_mean = old_grad.abs().mean().item()
    new_grad_mean = new_grad.abs().mean().item()
    
    print(f"Old Embedding Grad Mean (Should be 0): {old_grad_mean:.9f}")
    print(f"New Embedding Grad Mean (Should be >0): {new_grad_mean:.9f}")
    
    if old_grad_mean > 1e-9:
        print("❌ FAIL: LEAK DETECTED! Old Embeddings are updating.")
    else:
        print("✅ PASS: Old Embeddings are frozen.")

if __name__ == "__main__":
    run_v15_embedding_leak()
