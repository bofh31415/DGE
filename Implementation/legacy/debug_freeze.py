"""Debug: Check all frozen components during training."""
import torch
from core.model import DGESimpleTransformer
from core.training import TaskType, generate_batch, evaluate_task
from core.utils import DGEAdamW

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

acc_a_before = evaluate_task(model, TaskType.COUNT_UP, 500)
print(f"Task A accuracy before expansion: {acc_a_before:.1f}%")

# Save pre-expansion weights
pre_emb = model.token_emb.weight.clone().detach()
pre_q = model.layers[0].w_q.weight.clone().detach()
pre_head = model.lm_head.weight.clone().detach()

# Expand
print("\nExpanding...")
model.expand_model(new_input_dim=96, new_output_dim=500, router_type='mlp', 
                   use_orthogonal_init=False, isolate_cross_terms=False, router_init_bias=0.0)

acc_a_after_expand = evaluate_task(model, TaskType.COUNT_UP, 500)
print(f"Task A accuracy after expansion: {acc_a_after_expand:.1f}%")

# Save post-expansion weights
post_expand_q = model.layers[0].w_q.weight.clone().detach()

# Train on Task B
print("\nTraining Task B...")
optimizer = DGEAdamW(model.parameters(), lr=1e-3)

for i in range(200):
    x, y = generate_batch(TaskType.COUNT_DOWN, 500, 32, 32)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        acc = evaluate_task(model, TaskType.COUNT_UP, 500)
        print(f"Step {i}: Task A accuracy = {acc:.1f}%")

# Final weights
post_train_q = model.layers[0].w_q.weight.clone().detach()
post_train_emb = model.token_emb.weight.clone().detach()
post_train_head = model.lm_head.weight.clone().detach()

print("\n" + "=" * 60)
print("WEIGHT CHANGE ANALYSIS")
print("=" * 60)

# Q_TR (Old Core) - should be unchanged
q_tr_change = (post_train_q[:64, :64] - post_expand_q[:64, :64]).norm().item()
print(f"Q_TR (Core) change: {q_tr_change:.6f}")
print(f"  -> Should be 0.0 (frozen)")

# Q_TL (Firewall) - should stay at 0
q_tl_after = post_train_q[:64, 64:].norm().item()
print(f"Q_TL (Firewall) norm: {q_tl_after:.6f}")
print(f"  -> Should be 0.0 (zeroed and frozen)")

# Q_BR (Synergy) - can change
q_br_change = (post_train_q[64:, :64] - post_expand_q[64:, :64]).norm().item()
print(f"Q_BR (Synergy) change: {q_br_change:.6f}")
print(f"  -> Should be > 0 (trainable)")

# Embedding old dims
emb_old_change = (post_train_emb[:, :64] - pre_emb).norm().item()
print(f"Embedding old dims change: {emb_old_change:.6f}")
print(f"  -> Should be 0.0 (frozen)")

# LM Head old columns
head_old_change = (post_train_head[:, :64] - pre_head).norm().item()
print(f"LM Head old cols change: {head_old_change:.6f}")
print(f"  -> Should be 0.0 (frozen)")

# LM Head new columns
head_new_norm = post_train_head[:, 64:].norm().item()
print(f"LM Head new cols norm: {head_new_norm:.6f}")
print(f"  -> Should be > 0 (trainable)")

print("=" * 60)
