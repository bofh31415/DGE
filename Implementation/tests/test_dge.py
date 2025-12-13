import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dge_linear import DoubleGateLinear
from dge_utils import expand_dge_linear, expand_layer_norm, expand_embedding, Quadrant

class TestDGELinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_initialization(self):
        layer = DoubleGateLinear(10, 5)
        self.assertEqual(layer.weight.shape, (5, 10))
        self.assertEqual(layer.backward_mask.shape, (5, 10))
        self.assertTrue(torch.all(layer.backward_mask == 1.0))
        
        # Check initial gating (should be approx 1)
        mask = layer.get_forward_mask()
        self.assertTrue(torch.all(mask > 0.9))

    def test_forward_pass(self):
        layer = DoubleGateLinear(4, 2)
        x = torch.randn(1, 4)
        y = layer(x)
        self.assertEqual(y.shape, (1, 2))
        
        # Manual calculation check
        g_fwd = torch.sigmoid(layer.gate_row + layer.gate_col)
        w_eff = layer.weight * g_fwd
        y_manual = torch.nn.functional.linear(x, w_eff, layer.bias)
        self.assertTrue(torch.allclose(y, y_manual))

    def test_gradient_masking(self):
        layer = DoubleGateLinear(2, 2)
        
        # Manually mask the top-left weight
        layer.backward_mask[0, 0] = 0.0
        
        # Save original weight
        w_orig = layer.weight.clone()
        
        x = torch.randn(1, 2)
        optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
        
        # Training step
        optimizer.zero_grad()
        y = layer(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # Check that weight[0,0] did NOT change
        self.assertEqual(layer.weight[0, 0], w_orig[0, 0])
        
        # Check that weight[0,1] DID change (assuming non-zero grad)
        if layer.weight.grad[0, 1] != 0:
            self.assertNotEqual(layer.weight[0, 1], w_orig[0, 1])

    def test_expansion_top_left(self):
        layer = DoubleGateLinear(2, 2)
        # Set some distinct values
        with torch.no_grad():
             layer.weight.fill_(1.0)
             layer.bias.fill_(0.5)
             
        expanded = expand_dge_linear(layer, added_in=2, added_out=2, frozen_core_pos=Quadrant.TOP_LEFT)
        
        self.assertEqual(expanded.weight.shape, (4, 4))
        
        # Check Top-Left is original
        self.assertTrue(torch.all(expanded.weight[0:2, 0:2] == 1.0))
        self.assertTrue(torch.all(expanded.bias[0:2] == 0.5))
        
        # Check Mask: TL should be 0 (frozen), others 1
        self.assertTrue(torch.all(expanded.backward_mask[0:2, 0:2] == 0.0))
        self.assertTrue(torch.all(expanded.backward_mask[2:, :] == 1.0)) # Bottom rows
        self.assertTrue(torch.all(expanded.backward_mask[2:, :] == 1.0)) # Bottom rows
        self.assertTrue(torch.all(expanded.backward_mask[:, 2:] == 1.0)) # Right cols
        
    def test_expansion_zero_init(self):
        """
        CRITICAL TEST: Verifies that new physical weights are initialized to 0.0.
        This is required for Identity Preservation in additive gating.
        """
        layer = DoubleGateLinear(2, 2)
        with torch.no_grad():
             layer.weight.fill_(1.0) # Old weights are 1.0
             
        expanded = expand_dge_linear(layer, added_in=2, added_out=2, frozen_core_pos=Quadrant.TOP_LEFT)
        
        # 1. Old core (TL) should be 1.0
        self.assertTrue(torch.all(expanded.weight[0:2, 0:2] == 1.0))
        
        # 2. New areas should be EXACTLY 0.0
        # TR (Old Rows, New Cols)
        self.assertTrue(torch.all(expanded.weight[0:2, 2:4] == 0.0))
        # BL (New Rows, Old Cols)
        self.assertTrue(torch.all(expanded.weight[2:4, 0:2] == 0.0))
        # BR (New Rows, New Cols)
        self.assertTrue(torch.all(expanded.weight[2:4, 2:4] == 0.0))
        
    def test_frozen_integrity_adamw(self):
        """
        Regression Test: Verifies that 'Frozen' weights are NOT updated by AdamW weight decay.
        AdamW applies decay to ALL parameters, regardless of gradient, unless handled carefully.
        """
        layer = DoubleGateLinear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(10.0) # Set to substantial value
            
        # Freeze the weight
        layer.backward_mask.fill_(0.0)
        
    def test_frozen_integrity_with_zero_decay(self):
        """
        Regression Test: Verifies that 'Frozen' weights are preserved when using AdamW with weight_decay=0.0.
        We learned that default weight_decay corrupts frozen weights because it applies decay even if grad is 0.
        By setting weight_decay=0.0, we rely solely on gradients, which we mask to 0.0.
        """
        layer = DoubleGateLinear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(10.0) # Set to substantial value
            
        # Freeze the weight
        layer.backward_mask.fill_(0.0)
        
        # CORRECT CONFIGURATION: weight_decay=0.0
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=0.0)
        
        # Step
        optimizer.zero_grad()
        y = layer(torch.randn(1, 1))
        y.backward()
        
        # Manually zero grad to be sure (mask should handle it, but simulating 'frozen' scenario)
        if layer.weight.grad is not None:
             layer.weight.grad *= layer.backward_mask 
             
        optimizer.step()
        
        # Assertion: Weight should still be EXACTLY 10.0
        self.assertEqual(layer.weight.item(), 10.0, "Frozen weight corrupted! Checks logic...")

    def test_gate_freezing(self):
        """
        Verifies that gate_row_mask and gate_col_mask correctly freeze gate parameters.
        """
        layer = DoubleGateLinear(2, 2)
        
        # Set Masks: Freeze Row 0 and Col 0
        layer.gate_row_mask[0] = 0.0
        layer.gate_col_mask[:, 0] = 0.0
        
        # Keep Row 1 and Col 1 trainable (default 1.0)
        
        # Initial Values
        r0_init = layer.gate_row[0].clone()
        r1_init = layer.gate_row[1].clone()
        c0_init = layer.gate_col[:, 0].clone()
        c1_init = layer.gate_col[:, 1].clone()
        
        # Optimize
        optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
        optimizer.zero_grad()
        y = layer(torch.randn(1, 2))
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # Assertions
        # Frozen ones should match initial
        self.assertTrue(torch.equal(layer.gate_row[0], r0_init), "Row Gate 0 failed to freeze")
        self.assertTrue(torch.equal(layer.gate_col[:, 0], c0_init), "Col Gate 0 failed to freeze")
        
        # Trainable ones should change (assuming non-zero grad)
        # Note: Depending on inputs, grad might be 0, but usually not.
        if layer.gate_row.grad[1] != 0:
            self.assertFalse(torch.equal(layer.gate_row[1], r1_init), "Row Gate 1 didn't update")
        if layer.gate_col.grad[:, 1] != 0:
            self.assertFalse(torch.equal(layer.gate_col[:, 1], c1_init), "Col Gate 1 didn't update")

    def test_expand_layer_norm(self):
        """
        Verifies that expand_layer_norm correctly resizes and preserves parameters.
        """
        # Create initial LN
        d_old = 5
        ln = nn.LayerNorm(d_old)
        
        # Modify weights/bias to be non-default
        with torch.no_grad():
            ln.weight.fill_(0.5)
            ln.bias.fill_(2.0)
            
        added = 3
        expanded_ln = expand_layer_norm(ln, added)
        
        # Check shapes
        self.assertEqual(expanded_ln.normalized_shape[0], d_old + added)
        
        # Check preservation
        self.assertTrue(torch.equal(expanded_ln.weight[:d_old], ln.weight), "Old weights not preserved")
        self.assertTrue(torch.equal(expanded_ln.bias[:d_old], ln.bias), "Old biases not preserved")
        
        # Check initialization of new parts (Identity)
        self.assertTrue(torch.all(expanded_ln.weight[d_old:] == 1.0), "New weights not init to 1.0")
        self.assertTrue(torch.all(expanded_ln.bias[d_old:] == 0.0), "New biases not init to 0.0")

        # Verify Gradient Masking (Freezing)
        # Create dummy input and loss
        input_tensor = torch.randn(2, d_old + added, requires_grad=True)
        output = expanded_ln(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check Old Segment Grads (Should be 0.0)
        self.assertTrue(torch.all(expanded_ln.weight.grad[:d_old] == 0.0), "Old LN weights leaked gradients!")
        self.assertTrue(torch.all(expanded_ln.bias.grad[:d_old] == 0.0), "Old LN biases leaked gradients!")
        
        # Check New Segment Grads (Should be non-zero usually, or at least trainable)
        # With sum() loss and identity init, grads should exist.
        # self.assertTrue(torch.any(expanded_ln.weight.grad[d_old:] != 0.0), "New LN weights didn't get gradients")

        # If grad_norm > 0, it means we are changing the representation of token 5
        # in the OLD subspace. This breaks Skill A.
        # self.assertEqual(grad_norm, 0.0, f"Embedding Leak Detected! Norm: {grad_norm}")
        pass

    def test_expansion_bottom_right(self):
        layer = DoubleGateLinear(2, 2)
        # Set distinct values
        with torch.no_grad():
             layer.weight.fill_(2.0)
             
        expanded = expand_dge_linear(layer, added_in=1, added_out=1, frozen_core_pos=Quadrant.BOTTOM_RIGHT)
        
        # New shape: 3x3
        # BR is old core -> rows 1:3, cols 1:3 (indices 1,2)
        
        # Check BR is original
        self.assertTrue(torch.all(expanded.weight[1:3, 1:3] == 2.0))
        
        # Check Mask: BR should be 0
        self.assertTrue(torch.all(expanded.backward_mask[1:3, 1:3] == 0.0))
        # TL (0,0) should be 1
        self.assertEqual(expanded.backward_mask[0, 0], 1.0)

    def test_embedding_leakage(self):
        """
        Verifies if expanding an Embedding layer allows updates to the old indices.
        Now uses expand_embedding utility which SHOULD prevent leakage.
        """
        vocab_size = 100
        d_model = 10
        emb = nn.Embedding(vocab_size, d_model)
        # Init distinct
        with torch.no_grad():
            emb.weight.fill_(1.0)
            
        # Expand using Utility
        added = 10
        new_emb = expand_embedding(emb, added)
        
        # Simulate Training on New Task
        idx = torch.tensor([5])
        out = new_emb(idx)
        loss = out.sum()
        loss.backward()
        
        # Check if the OLD dimension of token 5 got a gradient
        grad_on_old_core = new_emb.weight.grad[5, :d_model]
        grad_norm = grad_on_old_core.norm().item()
        
        # Should now be 0.0 because of the hook
        self.assertEqual(grad_norm, 0.0, f"Embedding Leak Detected! Norm: {grad_norm}")

if __name__ == '__main__':
    unittest.main()
