"""
IDK Safety Net Router (Router 0) - V 0.1.0
==========================================
Confidence-aware fallback router that activates when all specialized
routers are uncertain, preventing hallucinations on OOD inputs.

Architecture:
    Input → Router 1 (Task A) → confidence_1
         → Router 2 (Task B) → confidence_2
         → Router 0 (IDK)    → ALWAYS ACTIVE (residual)
         
    If max(confidence_1, confidence_2, ...) < threshold:
        → Blend in IDK embedding → Safe fallback output

Usage:
    from idk_router import IDKRouter, ConfidenceAggregator
    
    idk_router = IDKRouter(d_model=384, threshold=0.3)
    output = idk_router(hidden_states, router_confidences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class IDKRouter(nn.Module):
    """
    Router 0: The "I Don't Know" Safety Net.
    
    Activates when all specialized routers have low confidence,
    blending in a learned fallback embedding to prevent hallucination.
    """
    
    def __init__(
        self, 
        d_model: int, 
        threshold: float = 0.3,
        temperature: float = 1.0,
        learnable_threshold: bool = False
    ):
        """
        Args:
            d_model: Hidden dimension size
            threshold: Confidence threshold below which IDK activates
            temperature: Softness of the activation transition
            learnable_threshold: If True, threshold is a learnable parameter
        """
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # The learned "I don't know" state
        self.idk_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Optional learnable threshold
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))
        
        # Optional projection for IDK embedding (makes it context-dependent)
        self.idk_projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.idk_projection.weight)  # Start as identity
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        router_confidences: torch.Tensor,
        return_idk_weight: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Blend IDK embedding when confidence is low.
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            router_confidences: [batch, seq_len, num_routers] or [batch, num_routers]
            return_idk_weight: If True, return the IDK activation weight
            
        Returns:
            output: [batch, seq_len, d_model] - potentially modified hidden states
            idk_weight: [batch, seq_len] - how much IDK was blended in (optional)
        """
        # Get max confidence across all routers
        if router_confidences.dim() == 3:
            max_conf = router_confidences.max(dim=-1).values  # [batch, seq_len]
        else:
            max_conf = router_confidences.max(dim=-1).values.unsqueeze(1)  # [batch, 1]
        
        # Calculate IDK activation weight (higher when confidence is low)
        # Using smooth sigmoid transition
        idk_weight = torch.sigmoid(
            (self.threshold - max_conf) / self.temperature
        )  # [batch, seq_len]
        
        # Project IDK embedding
        idk_emb = self.idk_projection(self.idk_embedding)  # [d_model]
        
        # Blend: output = (1 - w) * hidden + w * idk
        idk_weight_expanded = idk_weight.unsqueeze(-1)  # [batch, seq_len, 1]
        output = (1 - idk_weight_expanded) * hidden_states + idk_weight_expanded * idk_emb
        
        if return_idk_weight:
            return output, idk_weight
        return output, None
    
    def get_idk_activation_rate(self, router_confidences: torch.Tensor) -> float:
        """Calculate what fraction of inputs would trigger IDK."""
        with torch.no_grad():
            if router_confidences.dim() == 3:
                max_conf = router_confidences.max(dim=-1).values
            else:
                max_conf = router_confidences.max(dim=-1).values
            
            idk_activated = (max_conf < self.threshold).float().mean()
            return idk_activated.item()


class ConfidenceAggregator(nn.Module):
    """
    Aggregates confidence scores from multiple routers.
    
    Each router outputs a gate activation; this module converts
    those activations into normalized confidence scores.
    """
    
    def __init__(self, num_routers: int = 2, aggregation: str = 'max'):
        """
        Args:
            num_routers: Number of specialized routers
            aggregation: 'max', 'mean', or 'learned'
        """
        super().__init__()
        self.num_routers = num_routers
        self.aggregation = aggregation
        
        if aggregation == 'learned':
            self.router_weights = nn.Parameter(torch.ones(num_routers))
    
    def forward(self, gate_activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert gate activations to confidence scores.
        
        Args:
            gate_activations: List of [batch, seq_len] tensors (mean gate activation per router)
            
        Returns:
            confidences: [batch, seq_len, num_routers]
        """
        # Stack: [batch, seq_len, num_routers]
        stacked = torch.stack(gate_activations, dim=-1)
        
        # Normalize to [0, 1] confidence scores
        # Gate activations are typically in [-inf, inf], so we use sigmoid
        confidences = torch.sigmoid(stacked)
        
        return confidences
    
    def get_aggregate_confidence(self, confidences: torch.Tensor) -> torch.Tensor:
        """
        Get single aggregate confidence value.
        
        Args:
            confidences: [batch, seq_len, num_routers]
            
        Returns:
            aggregate: [batch, seq_len]
        """
        if self.aggregation == 'max':
            return confidences.max(dim=-1).values
        elif self.aggregation == 'mean':
            return confidences.mean(dim=-1)
        elif self.aggregation == 'learned':
            weights = F.softmax(self.router_weights, dim=0)
            return (confidences * weights).sum(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class IDKTokenHead(nn.Module):
    """
    Special output head that can produce an [IDK] token.
    
    Instead of blending embeddings, this produces a special token
    when the model is uncertain.
    """
    
    def __init__(
        self, 
        d_model: int, 
        vocab_size: int,
        idk_token_id: int = None,
        threshold: float = 0.3
    ):
        """
        Args:
            d_model: Hidden dimension
            vocab_size: Vocabulary size
            idk_token_id: Token ID for [IDK] (if None, uses last token)
            threshold: Confidence threshold
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.idk_token_id = idk_token_id or (vocab_size - 1)
        self.threshold = threshold
    
    def forward(
        self, 
        logits: torch.Tensor, 
        router_confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Boost [IDK] token probability when confidence is low.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            router_confidences: [batch, seq_len, num_routers]
            
        Returns:
            modified_logits: [batch, seq_len, vocab_size]
        """
        # Get max confidence
        max_conf = router_confidences.max(dim=-1).values  # [batch, seq_len]
        
        # Calculate IDK boost (higher when less confident)
        idk_boost = torch.sigmoid((self.threshold - max_conf) / 0.1) * 10.0
        
        # Add boost to IDK token logit
        modified_logits = logits.clone()
        modified_logits[:, :, self.idk_token_id] += idk_boost.unsqueeze(-1).squeeze(-1)
        
        return modified_logits


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_idk_router(
    d_model: int,
    router_type: str = 'embedding',
    threshold: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create IDK routers.
    
    Args:
        d_model: Hidden dimension
        router_type: 'embedding' (blend), 'token' (special token), or 'none'
        threshold: Confidence threshold
        
    Returns:
        IDK router module
    """
    if router_type == 'embedding':
        return IDKRouter(d_model, threshold=threshold, **kwargs)
    elif router_type == 'token':
        vocab_size = kwargs.get('vocab_size', 50257)
        return IDKTokenHead(d_model, vocab_size, threshold=threshold)
    elif router_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown IDK router type: {router_type}")
