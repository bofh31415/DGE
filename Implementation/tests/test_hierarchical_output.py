"""
Tests for Hierarchical Output Architecture (V 0.13.0)
=====================================================
TDD: These tests are written BEFORE implementation.

This architecture enables true additive synergy by giving each skill
its own dedicated output head (LM head).
"""

import unittest
import torch
import sys
import os

# Add implementation (parent directory) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from dge_model import DGESimpleTransformer
from dge_utils import DGEAdamW


class TestHierarchicalOutput(unittest.TestCase):
    """Tests for HierarchicalOutputHead and skill-specific output isolation."""
    
    def setUp(self):
        """Create base model for testing."""
        self.vocab = 100
        self.d_model = 32
        
    def test_base_head_only(self):
        """Model without skills should work with just base head."""
        model = DGESimpleTransformer(
            vocab_size=self.vocab, 
            d_model=self.d_model, 
            n_layer=1, 
            n_head=4,
            synergy_mode='additive'
        )
        
        x = torch.randint(0, self.vocab, (2, 10))
        logits, _ = model(x)
        
        self.assertEqual(logits.shape, (2, 10, self.vocab))
        
    def test_add_skill_head(self):
        """Expanding for skill should create skill-specific output head."""
        model = DGESimpleTransformer(
            vocab_size=self.vocab, 
            d_model=self.d_model, 
            n_layer=1, 
            n_head=4,
            synergy_mode='additive'
        )
        
        # Expand for skill - should add skill head
        skill_id = model.expand_for_skill("Skill_A", expansion_delta=16)
        
        # Check skill head exists
        self.assertTrue(hasattr(model.lm_head, 'skill_heads'), 
                       "lm_head should have skill_heads attribute")
        self.assertEqual(len(model.lm_head.skill_heads), 1, 
                        "Should have 1 skill head after expansion")
        
    def test_freeze_skill_head(self):
        """Frozen skill head should not be modified during training."""
        model = DGESimpleTransformer(
            vocab_size=self.vocab, 
            d_model=self.d_model, 
            n_layer=1, 
            n_head=4,
            synergy_mode='additive'
        )
        
        # Expand and train skill A
        skill_a = model.expand_for_skill("A", 16)
        optimizer = DGEAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        
        for _ in range(20):
            x = torch.randint(1, 5, (4, 1))
            target = torch.full((4, 1), 10, dtype=torch.long)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab), target.view(-1))
            loss.backward()
            optimizer.step()
            
        # Capture skill A head weights
        skill_a_head_before = model.lm_head.skill_heads[0].weight.clone()
        
        # Freeze skill A
        model.freeze_skill(skill_a)
        
        # Expand and train skill B
        skill_b = model.expand_for_skill("B", 16)
        optimizer = DGEAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        
        for _ in range(20):
            x = torch.randint(1, 5, (4, 1))
            target = torch.full((4, 1), 11, dtype=torch.long)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab), target.view(-1))
            loss.backward()
            optimizer.step()
            
        # Skill A head should be unchanged
        skill_a_head_after = model.lm_head.skill_heads[0].weight.clone()
        diff = (skill_a_head_before - skill_a_head_after).abs().max().item()
        
        self.assertLess(diff, 1e-6, "Frozen skill head should not change during Skill B training")
        
    def test_additive_synergy_works(self):
        """
        CRITICAL TEST: With hierarchical output, both skills should contribute.
        
        This is the test that FAILED with shared weights in V0.12.0.
        It should PASS with skill-specific output heads.
        """
        model = DGESimpleTransformer(
            vocab_size=self.vocab, 
            d_model=self.d_model, 
            n_layer=1, 
            n_head=4,
            router_type='rbf',
            initial_gating=True,
            synergy_mode='additive'
        )
        
        # Train Skill A to output token 10
        skill_a = model.expand_for_skill("Pipes", 16)
        optimizer = DGEAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        for _ in range(50):
            x = torch.randint(1, 5, (4, 1))
            target = torch.full((4, 1), 10, dtype=torch.long)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab), target.view(-1))
            loss.backward()
            optimizer.step()
            
        # Freeze Skill A
        model.freeze_skill(skill_a)
        
        # Train Skill B to output token 11
        skill_b = model.expand_for_skill("Circles", 16)
        optimizer = DGEAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        for _ in range(50):
            x = torch.randint(1, 5, (4, 1))
            target = torch.full((4, 1), 11, dtype=torch.long)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab), target.view(-1))
            loss.backward()
            optimizer.step()
            
        # Test: BOTH tokens should have elevated probability
        model.eval()
        x_test = torch.tensor([[3]])
        logits, _ = model(x_test)
        probs = torch.nn.functional.softmax(logits[0, 0], dim=-1)
        
        p_pipe = probs[10].item()
        p_circle = probs[11].item()
        
        print(f"Hierarchical Output - P(pipe): {p_pipe:.4f}, P(circle): {p_circle:.4f}")
        
        # With hierarchical output, BOTH should contribute
        self.assertGreater(p_pipe, 0.05, "Pipe skill should contribute")
        self.assertGreater(p_circle, 0.05, "Circle skill should contribute")


if __name__ == '__main__':
    unittest.main()
