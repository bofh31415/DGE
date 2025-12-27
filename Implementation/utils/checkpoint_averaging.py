"""
Checkpoint Averaging Utilities (V 0.1.0)
========================================
Implements LAWA (LAtest Weight Averaging) and EMA for LLM pretraining.

Based on: "Early Weight Averaging meets High Learning Rates for LLM Pre-training" (ICLR 2024)

Key findings from the paper:
- LAWA outperforms EMA and SWA for LLM pretraining
- Best with higher learning rates (1.5-2x normal)
- Start averaging after warmup phase
- Use larger checkpoint spacing for better results
"""

import torch
from collections import deque
from typing import Dict, Optional, Any
import copy


class LAWAManager:
    """
    LAtest Weight Averaging (LAWA) Manager.
    
    Maintains a sliding window of the last K checkpoints and computes
    their uniform average for improved generalization.
    
    Args:
        window_size: Number of checkpoints to average (default: 5)
        device: Device to store checkpoints (default: 'cpu' to save GPU memory)
    """
    
    def __init__(self, window_size: int = 5, device: str = 'cpu'):
        self.window_size = window_size
        self.device = device
        self.buffer: deque = deque(maxlen=window_size)
        self.update_count = 0
    
    def update(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Add a new checkpoint to the buffer.
        
        Args:
            state_dict: Model state dict to add
        """
        # Deep copy and move to CPU to save GPU memory
        cpu_state = {k: v.detach().clone().to(self.device) 
                     for k, v in state_dict.items()}
        self.buffer.append(cpu_state)
        self.update_count += 1
    
    def get_averaged(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute the uniform average of all checkpoints in the buffer.
        
        Returns:
            Averaged state dict, or None if buffer is empty
        """
        if not self.buffer:
            return None
        
        if len(self.buffer) == 1:
            return copy.deepcopy(self.buffer[0])
        
        # Compute uniform average
        averaged = {}
        for key in self.buffer[0].keys():
            stacked = torch.stack([ckpt[key].float() for ckpt in self.buffer])
            averaged[key] = stacked.mean(dim=0).to(self.buffer[0][key].dtype)
        
        return averaged
    
    def get_buffer_size(self) -> int:
        """Return current number of checkpoints in buffer."""
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Return True if buffer has at least 2 checkpoints for meaningful averaging."""
        return len(self.buffer) >= 2
    
    def clear(self) -> None:
        """Clear the checkpoint buffer."""
        self.buffer.clear()
        self.update_count = 0


class EMAManager:
    """
    Exponential Moving Average (EMA) Manager.
    
    Maintains an exponentially weighted average of model weights.
    Formula: ema = decay * ema + (1 - decay) * current
    
    Args:
        decay: EMA decay rate (default: 0.999, higher = slower adaptation)
        device: Device to store EMA weights
    """
    
    def __init__(self, decay: float = 0.999, device: str = 'cpu'):
        self.decay = decay
        self.device = device
        self.shadow: Optional[Dict[str, torch.Tensor]] = None
        self.update_count = 0
    
    def update(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update EMA with new model weights.
        
        Args:
            state_dict: Current model state dict
        """
        if self.shadow is None:
            # Initialize shadow with first state
            self.shadow = {k: v.detach().clone().to(self.device) 
                          for k, v in state_dict.items()}
        else:
            # EMA update
            with torch.no_grad():
                for k, v in state_dict.items():
                    self.shadow[k] = (self.decay * self.shadow[k] + 
                                     (1 - self.decay) * v.to(self.device))
        
        self.update_count += 1
    
    def get_averaged(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get the current EMA weights.
        
        Returns:
            EMA state dict, or None if not initialized
        """
        if self.shadow is None:
            return None
        return copy.deepcopy(self.shadow)
    
    def is_ready(self) -> bool:
        """Return True if EMA has been initialized."""
        return self.shadow is not None
    
    def clear(self) -> None:
        """Reset EMA state."""
        self.shadow = None
        self.update_count = 0


class CheckpointAverager:
    """
    Unified checkpoint averaging manager supporting multiple strategies.
    
    Args:
        strategy: 'lawa' or 'ema'
        window_size: For LAWA, number of checkpoints to average
        decay: For EMA, decay rate
        device: Device to store checkpoints
    """
    
    def __init__(
        self, 
        strategy: str = 'lawa',
        window_size: int = 5,
        decay: float = 0.999,
        device: str = 'cpu'
    ):
        self.strategy = strategy.lower()
        
        if self.strategy == 'lawa':
            self.manager = LAWAManager(window_size=window_size, device=device)
        elif self.strategy == 'ema':
            self.manager = EMAManager(decay=decay, device=device)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'lawa' or 'ema'.")
    
    def update(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Update with new checkpoint."""
        self.manager.update(state_dict)
    
    def get_averaged(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get averaged weights."""
        return self.manager.get_averaged()
    
    def is_ready(self) -> bool:
        """Check if averaging is ready."""
        return self.manager.is_ready()
    
    @property
    def update_count(self) -> int:
        """Number of updates."""
        return self.manager.update_count


# Convenience functions
def create_lawa(window_size: int = 5) -> CheckpointAverager:
    """Create LAWA averager with specified window size."""
    return CheckpointAverager(strategy='lawa', window_size=window_size)


def create_ema(decay: float = 0.999) -> CheckpointAverager:
    """Create EMA averager with specified decay."""
    return CheckpointAverager(strategy='ema', decay=decay)
