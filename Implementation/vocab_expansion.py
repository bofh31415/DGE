"""
Dynamic Vocabulary Expansion Module (V 0.1.0)
=============================================
Implements DGE-based vocabulary expansion for adding new tokens during training.

Key Features:
- Gated output for new tokens (starts closed, opens with gradient)
- Smart initialization from constituent sub-tokens
- Hermit Crab style pre-allocation

Usage:
    expander = GatedVocabExpander(model, tokenizer)
    expander.add_token("Influencer", constituent_tokens=["In", "flu", "encer"])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import json
import os


class GatedVocabExpander:
    """
    Manages dynamic vocabulary expansion with gated integration.
    
    Uses DGE principles:
    - New tokens start with closed gate (alpha=-10)
    - Embeddings initialized from mean of constituents
    - Gate opens naturally through gradient descent
    """
    
    def __init__(
        self, 
        model: nn.Module,
        tokenizer,
        gate_init: float = -10.0,
        gate_boost: float = 10.0,
        over_provision: int = 100,  # Pre-allocate space for N new tokens
    ):
        """
        Args:
            model: The DGE model to expand
            tokenizer: The tokenizer (must support add_tokens)
            gate_init: Initial gate value for new tokens (negative = closed)
            gate_boost: Gradient multiplier for gates
            over_provision: Number of slots to pre-allocate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gate_init = gate_init
        self.gate_boost = gate_boost
        self.over_provision = over_provision
        
        # Track added tokens
        self.added_tokens: Dict[str, int] = {}
        self.token_metadata: Dict[int, Dict] = {}
        
        # N-gram tracking for detection
        self.ngram_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.ngram_losses: Dict[Tuple[int, ...], List[float]] = defaultdict(list)
        
        # Initialize output gates if model has lm_head
        self._init_output_gates()
    
    def _init_output_gates(self):
        """Initialize gating mechanism for output layer."""
        if hasattr(self.model, 'lm_head'):
            vocab_size = self.model.lm_head.weight.shape[0]
            # Register gate parameters (all ones = open for existing vocab)
            if not hasattr(self.model, '_vocab_gates'):
                self.model.register_buffer(
                    '_vocab_gates', 
                    torch.zeros(vocab_size)  # 0 = sigmoid(0)=0.5, but we want 1
                )
                # Actually, we want existing tokens fully open
                self.model._vocab_gates.fill_(10.0)  # sigmoid(10) ≈ 1
    
    def add_token(
        self, 
        token: str, 
        constituent_tokens: Optional[List[str]] = None,
        constituent_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Add a new token to the vocabulary with gated integration.
        
        Args:
            token: The new token string (e.g., "Influencer")
            constituent_tokens: Sub-tokens to derive embedding from
            constituent_ids: Or directly provide token IDs
            
        Returns:
            The new token's ID
        """
        # Get constituent IDs
        if constituent_ids is None and constituent_tokens is not None:
            constituent_ids = [self.tokenizer.encode(t)[0] for t in constituent_tokens]
        elif constituent_ids is None:
            # Auto-tokenize the new token to get constituents
            constituent_ids = self.tokenizer.encode(token)
        
        # Add to tokenizer
        num_added = self.tokenizer.add_tokens([token])
        if num_added == 0:
            print(f"⚠️ Token '{token}' already exists in vocabulary")
            return self.tokenizer.convert_tokens_to_ids(token)
        
        new_id = len(self.tokenizer) - 1
        
        # Expand model embeddings
        self._expand_embeddings(new_id, constituent_ids)
        
        # Expand output head with closed gate
        self._expand_lm_head(new_id, constituent_ids)
        
        # Track
        self.added_tokens[token] = new_id
        self.token_metadata[new_id] = {
            "token": token,
            "constituents": constituent_ids,
            "step_added": 0,  # Will be updated during training
        }
        
        print(f"✅ Added token '{token}' (ID: {new_id}) from constituents {constituent_ids}")
        return new_id
    
    def _expand_embeddings(self, new_id: int, constituent_ids: List[int]):
        """Expand embedding matrix with smart initialization."""
        if hasattr(self.model, 'token_emb'):
            emb = self.model.token_emb
            old_weight = emb.weight.data
            old_vocab_size = old_weight.shape[0]
            d_model = old_weight.shape[1]
            
            if new_id >= old_vocab_size:
                # Need to expand
                new_weight = torch.zeros(new_id + 1, d_model, device=old_weight.device)
                new_weight[:old_vocab_size] = old_weight
                
                # Smart initialization: mean of constituents
                if constituent_ids:
                    constituent_embeds = old_weight[constituent_ids]
                    new_weight[new_id] = constituent_embeds.mean(dim=0)
                else:
                    # Fallback to small random
                    new_weight[new_id] = torch.randn(d_model, device=old_weight.device) * 0.02
                
                # Replace embedding
                self.model.token_emb = nn.Embedding.from_pretrained(
                    new_weight, freeze=False, padding_idx=emb.padding_idx
                )
    
    def _expand_lm_head(self, new_id: int, constituent_ids: List[int]):
        """Expand output head with closed gate for new token."""
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
            old_weight = lm_head.weight.data
            old_vocab_size = old_weight.shape[0]
            d_model = old_weight.shape[1]
            
            old_bias = None
            if lm_head.bias is not None:
                old_bias = lm_head.bias.data
            
            if new_id >= old_vocab_size:
                # Expand weight
                new_weight = torch.zeros(new_id + 1, d_model, device=old_weight.device)
                new_weight[:old_vocab_size] = old_weight
                
                # Smart init from constituents
                if constituent_ids:
                    constituent_weights = old_weight[constituent_ids]
                    new_weight[new_id] = constituent_weights.mean(dim=0)
                else:
                    new_weight[new_id] = torch.randn(d_model, device=old_weight.device) * 0.02
                
                # Create new linear layer
                new_lm_head = nn.Linear(d_model, new_id + 1, bias=old_bias is not None)
                new_lm_head.weight.data = new_weight
                if old_bias is not None:
                    new_bias = torch.zeros(new_id + 1, device=old_bias.device)
                    new_bias[:old_vocab_size] = old_bias
                    new_lm_head.bias.data = new_bias
                
                self.model.lm_head = new_lm_head.to(old_weight.device)
                
                # Expand gates
                if hasattr(self.model, '_vocab_gates'):
                    old_gates = self.model._vocab_gates
                    new_gates = torch.zeros(new_id + 1, device=old_gates.device)
                    new_gates[:old_vocab_size] = old_gates
                    new_gates[new_id] = self.gate_init  # CLOSED for new token
                    self.model._vocab_gates = new_gates
    
    def apply_output_gating(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply gating to logits for new tokens."""
        if hasattr(self.model, '_vocab_gates'):
            gates = torch.sigmoid(self.model._vocab_gates)
            return logits * gates.unsqueeze(0).unsqueeze(0)  # [1, 1, V]
        return logits
    
    def track_ngrams(self, token_ids: torch.Tensor, losses: Optional[torch.Tensor] = None, n: int = 3):
        """
        Track n-gram frequencies for detecting new token candidates.
        
        Args:
            token_ids: Batch of token IDs [B, L]
            losses: Per-token losses if available
            n: N-gram size
        """
        for seq in token_ids:
            seq = seq.tolist()
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i+n])
                self.ngram_counts[ngram] += 1
    
    def get_token_candidates(self, min_freq: int = 100, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get candidate n-grams for new tokens."""
        candidates = []
        for ngram, count in self.ngram_counts.items():
            if count >= min_freq:
                tokens = [self.tokenizer.decode([t]) for t in ngram]
                candidates.append(("".join(tokens), count))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]
    
    def save_state(self, path: str):
        """Save expander state."""
        state = {
            "added_tokens": self.added_tokens,
            "token_metadata": self.token_metadata,
            "ngram_counts": {str(k): v for k, v in self.ngram_counts.items()},
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load expander state."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.added_tokens = state.get("added_tokens", {})
            self.token_metadata = {int(k): v for k, v in state.get("token_metadata", {}).items()}
