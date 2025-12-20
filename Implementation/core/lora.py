"""
LoRA (Low-Rank Adaptation) Baseline Implementation

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

This provides a fair comparison baseline for DGE. LoRA adds trainable low-rank
matrices alongside frozen base weights, but does NOT gate them dynamically.

Comparison with DGE:
- LoRA: frozen_base + (alpha/rank) * B @ A @ x
- DGE:  frozen_core + gate * new_weights @ x

Win Condition for DGE:
- DGE should show lower "Forgetting" (better Stability) than LoRA
- Even if LoRA learns slightly faster, DGE should preserve old task better
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a Linear Layer.
    
    Implements: y = base(x) + (alpha/rank) * dropout(B @ A @ x)
    
    Where:
        - base: Frozen pre-trained linear layer
        - A: [rank, in_features] - Low-rank input projection
        - B: [out_features, rank] - Low-rank output projection
        - alpha: Scaling factor
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int = 8, 
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Low-rank matrices
        # A: [rank, in] - initialized with Kaiming uniform
        # B: [out, rank] - initialized to zero (so LoRA starts as identity)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA matrices per the paper."""
        # A: Random Gaussian (or Kaiming)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: Zero (so LoRA contribution starts at zero = identity)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            y: Output tensor [..., out_features]
        """
        # Base output (frozen)
        base_output = self.base_layer(x)
        
        # LoRA output: (alpha/r) * B @ A @ x
        # x: [..., in]
        # A @ x: [..., rank] (via linear)
        # B @ (A @ x): [..., out]
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        
        return base_output + self.scaling * lora_output
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base layer for inference.
        Returns a standard nn.Linear with merged weights.
        """
        merged = nn.Linear(self.in_features, self.out_features, 
                          bias=self.base_layer.bias is not None)
        
        with torch.no_grad():
            # W_merged = W_base + (alpha/r) * B @ A
            delta_w = self.lora_B @ self.lora_A
            merged.weight.copy_(self.base_layer.weight + self.scaling * delta_w)
            
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
                
        return merged
        
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha})')


class LoRABlock(nn.Module):
    """
    Transformer block with LoRA applied to QKV and MLP layers.
    Mirrors DGEBlock structure for fair comparison.
    """
    def __init__(self, base_block: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.d_model = base_block.d_model
        self.n_head = base_block.n_head
        self.head_dim = base_block.head_dim
        
        # Copy and wrap layers with LoRA
        # Attention: QKV and Out projections
        self.ln1 = base_block.ln1
        self.qkv = LoRALinear(base_block.qkv, rank=rank, alpha=alpha)
        self.out_proj = LoRALinear(base_block.out_proj, rank=rank, alpha=alpha)
        
        # MLP
        self.ln2 = base_block.ln2
        self.fc1 = LoRALinear(base_block.fc1, rank=rank, alpha=alpha)
        self.fc2 = LoRALinear(base_block.fc2, rank=rank, alpha=alpha)
        
    def forward(self, x):
        """Standard transformer forward with LoRA layers."""
        B, T, D = x.shape
        
        # Self-attention
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        
        x = x + out
        
        # MLP
        h = self.ln2(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        
        x = x + h
        return x


def wrap_model_with_lora(
    model: nn.Module, 
    rank: int = 8, 
    alpha: float = 16.0,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Wrap a model's linear layers with LoRA adapters.
    
    Args:
        model: Pre-trained model to adapt
        rank: LoRA rank (lower = more parameter efficient)
        alpha: LoRA scaling factor
        target_modules: List of module names to wrap (default: all Linear)
        
    Returns:
        Model with LoRA adapters (base weights frozen)
    """
    if target_modules is None:
        target_modules = ['qkv', 'out_proj', 'fc1', 'fc2', 'lm_head']
        
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    
                # Replace with LoRA
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                
    return model


def count_trainable_params(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters.
    
    Returns:
        (trainable_params, total_params, percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable / total if total > 0 else 0.0
    
    return trainable, total, percentage
