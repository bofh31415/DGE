"""
DGE Replay Buffer Module

Provides automatic replay buffer management for continual learning.
Uses Reservoir Sampling for uniform coverage without knowing dataset size upfront.

Version: 0.5.0
"""

import random
import torch
import os
import json
from typing import List, Tuple, Optional

class ReplayBuffer:
    """
    Fixed-size replay buffer with Reservoir Sampling.
    
    Automatically maintains a representative sample of training data
    for use during Asymmetric Replay in continual learning.
    
    Attributes:
        max_size: Maximum number of samples to store.
        buffer: List of (input, target) tensor tuples.
        seen_count: Total samples seen (for reservoir sampling).
        task_name: Optional name of the task this buffer represents.
    """
    
    def __init__(self, max_size: int = 5000, task_name: str = "unknown"):
        """
        Initialize an empty replay buffer.
        
        Args:
            max_size: Maximum number of samples to retain.
            task_name: Name of the task for logging/saving.
        """
        self.max_size = max_size
        self.task_name = task_name
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_count = 0
        self.metadata = {
            "task_name": task_name,
            "max_size": max_size,
            "created_at": None,
            "embedding_variance": None  # For complexity estimation
        }
    
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a sample to the buffer using Reservoir Sampling.
        
        Ensures uniform probability of any sample being in the buffer
        regardless of when it was seen.
        
        Args:
            x: Input tensor (can be batched or single sample).
            y: Target tensor (matching x shape).
        """
        # Handle batched inputs
        if x.dim() > 1 and x.size(0) > 1:
            for i in range(x.size(0)):
                self._add_single(x[i].clone(), y[i].clone())
        else:
            self._add_single(x.clone(), y.clone())
    
    def _add_single(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a single sample using Reservoir Sampling."""
        self.seen_count += 1
        
        if len(self.buffer) < self.max_size:
            # Buffer not full, just append
            self.buffer.append((x.cpu(), y.cpu()))
        else:
            # Reservoir sampling: replace with probability max_size/seen_count
            idx = random.randint(0, self.seen_count - 1)
            if idx < self.max_size:
                self.buffer[idx] = (x.cpu(), y.cpu())
    
    def sample(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.
        
        Args:
            batch_size: Number of samples to return.
            
        Returns:
            Tuple of (inputs, targets) tensors, batched.
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Clamp batch size to buffer size
        actual_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, actual_size)
        
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        
        return xs, ys
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def save(self, path: str) -> None:
        """
        Save buffer to disk.
        
        Args:
            path: Directory path to save buffer files.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save tensors
        if len(self.buffer) > 0:
            xs = torch.stack([s[0] for s in self.buffer])
            ys = torch.stack([s[1] for s in self.buffer])
            torch.save({'x': xs, 'y': ys}, os.path.join(path, 'buffer.pt'))
        
        # Save metadata
        self.metadata['size'] = len(self.buffer)
        self.metadata['seen_count'] = self.seen_count
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"💾 Replay buffer saved: {len(self.buffer)} samples for '{self.task_name}'")
    
    def load(self, path: str) -> None:
        """
        Load buffer from disk.
        
        Args:
            path: Directory path containing buffer files.
        """
        buffer_path = os.path.join(path, 'buffer.pt')
        meta_path = os.path.join(path, 'metadata.json')
        
        if os.path.exists(buffer_path):
            data = torch.load(buffer_path)
            xs, ys = data['x'], data['y']
            self.buffer = [(xs[i], ys[i]) for i in range(xs.size(0))]
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                self.task_name = self.metadata.get('task_name', 'unknown')
                self.seen_count = self.metadata.get('seen_count', len(self.buffer))
        
        print(f"📂 Replay buffer loaded: {len(self.buffer)} samples for '{self.task_name}'")


def estimate_replay_ratio(buffer: ReplayBuffer, model=None, sample_size: int = 100) -> float:
    """
    Estimate optimal replay ratio based on buffer complexity.
    
    Heuristic: Higher embedding variance = more diverse data = higher replay needed.
    
    Args:
        buffer: ReplayBuffer to analyze.
        model: Optional model with embedding layer for variance calculation.
        sample_size: Number of samples to use for estimation.
        
    Returns:
        Recommended replay ratio (0.05 - 0.20).
    """
    MIN_RATIO = 0.05  # 5% minimum
    MAX_RATIO = 0.20  # 20% maximum
    
    if buffer.is_empty():
        return MIN_RATIO
    
    if model is None:
        # Fallback: Use raw tensor variance
        xs, _ = buffer.sample(min(sample_size, len(buffer)))
        variance = torch.var(xs.float()).item()
        # Normalize assuming token IDs in range [0, 1000]
        normalized_var = min(variance / (500 ** 2), 1.0)
    else:
        # Use embedding variance (more meaningful)
        xs, _ = buffer.sample(min(sample_size, len(buffer)))
        with torch.no_grad():
            if hasattr(model, 'token_emb'):
                embeddings = model.token_emb(xs)
                variance = torch.var(embeddings).item()
                # Normalize assuming d_model scale
                normalized_var = min(variance / 1.0, 1.0)
            else:
                normalized_var = 0.5  # Default mid-range
    
    # Linear interpolation
    ratio = MIN_RATIO + (MAX_RATIO - MIN_RATIO) * normalized_var
    
    return round(ratio, 3)


# === Unit Tests ===
if __name__ == "__main__":
    print("Testing ReplayBuffer...")
    
    # Test basic operations
    buf = ReplayBuffer(max_size=100, task_name="test")
    
    # Add samples - use 2D batches (batch_size, seq_len)
    for i in range(200):
        x = torch.randint(0, 100, (8, 16))  # Batch of 8 samples, each with 16 tokens
        y = x + 1
        buf.add(x, y)
    
    assert len(buf) == 100, f"Expected 100, got {len(buf)}"
    assert buf.seen_count == 200 * 8, f"Expected 1600, got {buf.seen_count}"
    
    # Test sampling - each sample should be (16,) since we store individual samples
    xs, ys = buf.sample(10)
    assert xs.shape == (10, 16), f"Expected (10, 16), got {xs.shape}"
    
    # Test save/load
    buf.save("./test_buffer")
    buf2 = ReplayBuffer()
    buf2.load("./test_buffer")
    assert len(buf2) == len(buf)
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_buffer")
    
    print("✅ All tests passed!")
