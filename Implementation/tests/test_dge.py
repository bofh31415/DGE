import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dge_linear import DoubleGateLinear
from dge_utils import expand_dge_linear, Quadrant

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

if __name__ == '__main__':
    unittest.main()
