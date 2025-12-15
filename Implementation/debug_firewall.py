"""Debug script to check Firewall initialization."""
import torch
from dge_model import DGESimpleTransformer
from dge_utils import MoEGatedLinear

# Create and expand model
model = DGESimpleTransformer(vocab_size=500, d_model=64, n_layer=2, n_head=4)
model.expand_model(
    new_input_dim=96, 
    new_output_dim=500, 
    router_type='mlp', 
    use_orthogonal_init=False, 
    isolate_cross_terms=False, 
    router_init_bias=0.0
)

print("=" * 60)
print("CHECKING FIREWALL INITIALIZATION")
print("=" * 60)

# Check Layer 0 w_q
layer = model.layers[0].w_q
print('\nLayer 0 w_q:')
print(f'  Weight shape: {layer.weight.shape}')  # Should be (96, 96)
print(f'  Mask shape: {layer.backward_mask.shape}')

# For intermediate layers, old_out=64, old_in=64, added_out=32, added_in=32
old_out, old_in = 64, 64

# Q_TL = [:old_out, old_in:] (New In -> Old Out) = Should be ZERO
q_tl = layer.weight[:old_out, old_in:].detach()
print(f'  Q_TL shape: {q_tl.shape}')  # (64, 32)
print(f'  Q_TL norm: {q_tl.norm().item():.6f}')
print(f'  Q_TL max: {q_tl.abs().max().item():.6f}')
print(f'  Q_TL should be: 0.0 (Firewall)')

# Check backward mask for Q_TL
mask_q_tl = layer.backward_mask[:old_out, old_in:]
print(f'  Q_TL mask: {mask_q_tl.unique().tolist()}')  # Should be [0.0]

# Check LM Head
head = model.lm_head
print(f'\nLM Head:')
print(f'  Weight shape: {head.weight.shape}')  # (500, 96)

# For lm_head, old_out=500, old_in=64, added_out=0, added_in=32
old_out_head, old_in_head = 500, 64

# For lm_head with added_out=0, Q_TL should NOT be zeroed
q_tl_head = head.weight[:, old_in_head:].detach()  # New columns
print(f'  New Columns shape: {q_tl_head.shape}')  # (500, 32)
print(f'  New Columns norm: {q_tl_head.norm().item():.6f}')
print(f'  New Columns should be: NON-ZERO (Trainable for Plasticity)')

# Check backward mask for new columns
mask_new_cols = head.backward_mask[:, old_in_head:]
print(f'  New Cols mask unique: {mask_new_cols.unique().tolist()}')  # Should be [1.0] (trainable)

print("\n" + "=" * 60)
if q_tl.norm().item() < 0.01:
    print("OK: Intermediate layer Q_TL is properly zeroed")
else:
    print("BUG: Intermediate layer Q_TL is NOT zeroed!")
    
if q_tl_head.norm().item() > 0.1:
    print("OK: LM Head new columns are trainable (non-zero)")
else:
    print("BUG: LM Head new columns are near-zero!")
print("=" * 60)
