"""Debug: Check if gates are specializing for different tasks."""
import torch
from core.model import DGESimpleTransformer
from core.training import TaskType, generate_batch, evaluate_task
from core.utils import DGEAdamW, HybridGate

# Create and train Task A
print("Training Task A...")
model = DGESimpleTransformer(vocab_size=500, d_model=64, n_layer=2, n_head=4)
optimizer = DGEAdamW(model.parameters(), lr=1e-3)

for i in range(300):
    x, y = generate_batch(TaskType.COUNT_UP, 500, 32, 32)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

print(f"Task A accuracy: {evaluate_task(model, TaskType.COUNT_UP, 500):.1f}%")

# Expand
print("\nExpanding...")
model.expand_model(new_input_dim=96, new_output_dim=500, router_type='mlp',
                   use_orthogonal_init=False, isolate_cross_terms=False, router_init_bias=0.0)

print(f"Post-expansion Task A: {evaluate_task(model, TaskType.COUNT_UP, 500):.1f}%")

# Train with Experience Replay
print("\nTraining with Experience Replay...")
optimizer = DGEAdamW(model.parameters(), lr=1e-3)

for i in range(200):
    optimizer.zero_grad()
    if i % 2 == 0:
        x, y = generate_batch(TaskType.COUNT_DOWN, 500, 32, 32)
    else:
        x, y = generate_batch(TaskType.COUNT_UP, 500, 32, 32)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

print(f"\nAfter Replay Training:")
print(f"  Task A: {evaluate_task(model, TaskType.COUNT_UP, 500):.1f}%")
print(f"  Task B: {evaluate_task(model, TaskType.COUNT_DOWN, 500):.1f}%")

# Check gate activations for each task
print("\n" + "=" * 60)
print("GATE ACTIVATION ANALYSIS")
print("=" * 60)

model.eval()
with torch.no_grad():
    # Generate Task A input
    x_a, _ = generate_batch(TaskType.COUNT_UP, 500, 4, 32)
    # Generate Task B input
    x_b, _ = generate_batch(TaskType.COUNT_DOWN, 500, 4, 32)
    
    # Check first layer gate responses
    layer = model.layers[0].w_q
    
    # Embed inputs
    emb_a = model.token_emb(x_a) + model.pos_emb[:, :32, :]
    emb_b = model.token_emb(x_b) + model.pos_emb[:, :32, :]
    
    # Get gate values (if HybridGate)
    if hasattr(layer.gate_col, 'router') and layer.gate_col.router is not None:
        # Get router output
        gate_col = layer.gate_col
        
        # Router takes flattened input
        x_flat_a = emb_a.reshape(-1, emb_a.shape[-1])  # [B*T, D]
        x_flat_b = emb_b.reshape(-1, emb_b.shape[-1])
        
        # Get gate values for new dimensions
        gate_a = gate_col(x_flat_a)  # Full gate vector
        gate_b = gate_col(x_flat_b)
        
        # Extract new dimension gates (indices 64+)
        new_gate_a = gate_a[:, 64:].mean().item()  # Mean gate for new dims on Task A
        new_gate_b = gate_b[:, 64:].mean().item()  # Mean gate for new dims on Task B
        
        print(f"New dims gate (Task A input): {new_gate_a:.4f}")
        print(f"New dims gate (Task B input): {new_gate_b:.4f}")
        print(f"Difference: {abs(new_gate_b - new_gate_a):.4f}")
        
        if new_gate_a < 0.3 and new_gate_b > 0.7:
            print("\nOK: Gates are specializing! Close for A, Open for B")
        elif abs(new_gate_b - new_gate_a) < 0.1:
            print("\nISSUE: Gates NOT specializing - same values for both tasks")
        else:
            print("\nPARTIAL: Gates show some differentiation")
