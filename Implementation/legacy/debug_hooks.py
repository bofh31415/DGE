"""Debug: Check if gradient hooks are preventing Q_TL updates."""
import torch
from core.model import DGESimpleTransformer
from core.training import TaskType, generate_batch
from core.utils import DGEAdamW

# Create and train Task A
model = DGESimpleTransformer(vocab_size=500, d_model=64, n_layer=2, n_head=4)
optimizer = DGEAdamW(model.parameters(), lr=1e-3)

for i in range(100):
    x, y = generate_batch(TaskType.COUNT_UP, 500, 32, 32)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

# Expand
model.expand_model(new_input_dim=96, new_output_dim=500, router_type='mlp', 
                   use_orthogonal_init=False, isolate_cross_terms=False, router_init_bias=0.0)

# Record initial Q_TL state
layer = model.layers[0].w_q
q_tl_initial = layer.weight[:64, 64:].clone().detach()
print(f"Initial Q_TL norm: {q_tl_initial.norm().item():.6f}")

# New optimizer
optimizer = DGEAdamW(model.parameters(), lr=1e-3)

# Train 10 steps on Task B
for i in range(10):
    x, y = generate_batch(TaskType.COUNT_DOWN, 500, 32, 32)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    
    # Check gradient on Q_TL before clipping
    q_tl_grad = layer.weight.grad[:64, 64:].clone() if layer.weight.grad is not None else None
    if q_tl_grad is not None:
        print(f"Step {i}: Q_TL grad norm BEFORE hook: {q_tl_grad.norm().item():.2e}")
    
    optimizer.step()
    
    q_tl_current = layer.weight[:64, 64:].detach()
    print(f"Step {i}: Q_TL norm AFTER step: {q_tl_current.norm().item():.6f}")

# Final check
q_tl_final = layer.weight[:64, 64:].detach()
print(f"\nFinal Q_TL norm: {q_tl_final.norm().item():.6f}")
print(f"Change: {(q_tl_final - q_tl_initial).norm().item():.6f}")

if q_tl_final.norm().item() > 0.01:
    print("\nBUG: Q_TL was modified despite mask=0!")
else:
    print("\nOK: Q_TL remained frozen")
