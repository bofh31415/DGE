import os
import torch
import torch.nn as nn
from core.model import DGESimpleTransformer
from core.utils import MoEGatedLinear, HybridGate

def run_v11_debug_init():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V11 DEBUG INIT: Inspecting Router Bias after Expansion")
    print("="*60)
    
    # Defaults
    vocab_size = 500
    d_model = 64 
    n_layer = 2 
    n_head = 4
    
    print("Step 1: Creating Baseline Model...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    # 2. Expand with Explicit Bias -4.0
    print(f"Step 2: Expanding Model (Bias=-4.0 requested)...")
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=True,
        router_init_bias=-4.0
    )
    
    print(f"\nStep 3: Inspecting Router Biases...")
    
    found_routers = 0
    for name, module in model.named_modules():
        if isinstance(module, MoEGatedLinear):
            # Check Gate Row
            if module.gate_row is not None:
                gate = module.gate_row
                if isinstance(gate, HybridGate) and gate.new_count > 0:
                    found_routers += 1
                    print(f"  > Found Output Gate in {name} (New Count {gate.new_count})")
                    if isinstance(gate.router, nn.Sequential):
                        # MLP Router: Last layer bias
                        bias = gate.router[-1].bias
                        print(f"    - MLP Output Bias Mean: {bias.mean().item():.4f}")
                        print(f"    - MLP Output Bias Max:  {bias.max().item():.4f}")
                        print(f"    - MLP Output Bias Min:  {bias.min().item():.4f}")
                        print(f"    - MLP Output Bias Std:  {bias.std().item():.4f}")
                    elif isinstance(gate.router, nn.Linear):
                        bias = gate.router.bias
                        print(f"    - Linear Output Bias: {bias.mean().item():.4f}")
            
            # Check Gate Col
            if module.gate_col is not None:
                gate = module.gate_col
                if isinstance(gate, HybridGate) and gate.new_count > 0:
                    found_routers += 1
                    print(f"  > Found Input Gate in {name} (New Count {gate.new_count})")
                    if isinstance(gate.router, nn.Sequential):
                        bias = gate.router[-1].bias
                        print(f"    - MLP Output Bias Mean: {bias.mean().item():.4f}")
                        print(f"    - MLP Output Bias Max:  {bias.max().item():.4f}")
                        print(f"    - MLP Output Bias Min:  {bias.min().item():.4f}")
                    elif isinstance(gate.router, nn.Linear):
                        bias = gate.router.bias
                        print(f"    - Linear Output Bias: {bias.mean().item():.4f}")

    if found_routers == 0:
        print("❌ NO EXPANDED ROUTERS FOUND!")
    else:
        print(f"✅ Inspected {found_routers} routers.")

if __name__ == "__main__":
    run_v11_debug_init()
