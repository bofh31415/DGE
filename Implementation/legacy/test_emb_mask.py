"""Test embedding dge_mask fix for DGEAdamW weight decay protection."""
import torch
import torch.nn as nn
from core.utils import expand_embedding, DGEAdamW

# Create and expand embedding
emb = nn.Embedding(10, 4)
with torch.no_grad():
    emb.weight.fill_(1.0)

expanded = expand_embedding(emb, 2)

old_sum = expanded.weight[:, :4].sum().item()
print(f'Initial old dims sum: {old_sum:.2f}')

# Check dge_mask
has_mask = hasattr(expanded.weight, 'dge_mask')
print(f'dge_mask exists: {has_mask}')
if has_mask:
    print(f'dge_mask old cols: {expanded.weight.dge_mask[0, :4].tolist()}')
    print(f'dge_mask new cols: {expanded.weight.dge_mask[0, 4:].tolist()}')

# Use DGEAdamW with weight decay
optimizer = DGEAdamW([expanded.weight], lr=0.1, weight_decay=0.01)

# Simulate 100 training steps
for i in range(100):
    x = torch.tensor([0, 1, 2])
    out = expanded(x)
    loss = out[:, 4:].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

new_sum = expanded.weight[:, :4].sum().item()
print(f'Final old dims sum: {new_sum:.2f}')
print(f'Change: {abs(new_sum - old_sum):.6f}')

if abs(new_sum - old_sum) < 0.01:
    print('OK: Old dims preserved!')
else:
    print('BUG: Weight decay still affecting frozen weights')
