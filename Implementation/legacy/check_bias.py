
from core.model import DGESimpleTransformer
import torch

def check_bias():
    print("Initializing Model...")
    model = DGESimpleTransformer(vocab_size=100, d_model=32, n_layer=1, n_head=2)
    
    print("\nExpanding Model with router_init_bias=0.0...")
    model.expand_model(
        new_input_dim=64, 
        new_output_dim=100, 
        router_type='mlp',
        router_init_bias=0.0  # TARGET
    )
    
    # Check Bias
    found_bias = False
    for name, module in model.named_modules():
        if "gate" in name and "router" in name and hasattr(module, 'bias'):
             # This is low level.
             pass
             
    # Iterate named parameters
    for name, param in model.named_parameters():
        if "gate" in name and "router" in name and "bias" in name:
            # Check last layer bias (should be 0.0)
            # MLP: router.2.bias
            # Linear: router.bias
            
            # We used MLP.
            if "router.2.bias" in name or "router.bias" in name:
                mean_val = param.mean().item()
                print(f"Propagated Bias ({name}): {mean_val:.4f}")
                found_bias = True
                
    if not found_bias:
        print("❌ Could not find router bias parameters!")

if __name__ == "__main__":
    check_bias()
