import unittest
import torch
import shutil
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import DGELab
from dge_utils import DGEAdamW
from dge_training import train_task

class TestExperimentChainV2(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_chain_v2')
        os.makedirs(self.test_dir, exist_ok=True)
        self.lab = DGELab()
        self.lab.models_dir = self.test_dir
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mlp_router_chain_setup(self):
        """
        Simulates the logic of Experiment Option 3 (MLP Router).
        """
        print("\n[Integration] Testing V2 Chain Setup...")
        
        # 1. Reset Model
        self.lab.reset_model()
        self.assertIsNotNone(self.lab.model)
        self.assertEqual(self.lab.model.d_model, 64)
        
        # 2. Mock Initial Training (Skipped, assume pre-trained)
        
        # 3. Expansion with MLP
        current_in = self.lab.model.d_model
        current_out = self.lab.model.token_emb.num_embeddings
        
        print(f"Expanding: {current_in} -> {current_in + 16} (MLP)")
        self.lab.model.expand_model(new_input_dim=current_in + 16, new_output_dim=current_out, router_type='mlp')
        
        self.assertEqual(self.lab.model.d_model, 80)
        # Check if router is MLP
        # Access a gate to verify
        layer = self.lab.model.layers[0].w_q  # V12+: Separate Q, K, V layers
        # Check router type attribute if accessible, or structure
        self.assertEqual(layer.router_type, 'mlp')
        self.assertIsInstance(layer.gate_row.router, torch.nn.Sequential)
        
        # 4. Optimizer Setup (Diff LR)
        router_params = []
        default_params = []
        for name, param in self.lab.model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            else:
                default_params.append(param)
        
        optimizer = DGEAdamW([
            {'params': default_params, 'lr': 1e-3},
            {'params': router_params, 'lr': 1e-4}
        ])
        
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[1]['lr'], 0.0001)
        
        # 5. Train Step (Dry Run)
        print("Running 1 epoch dry run...")
        # Use existing loaders if mockable, else generate on fly.
        # Calling train_task requires data loaders.
        # We can skip full training if we verified init and expansion.
        # But let's verify forward/backward pass.
        
        x = torch.randint(0, 10, (2, 10))
        y = x # Mock
        
        optimizer.zero_grad()
        logits, loss = self.lab.model(x, y)
        loss.backward()
        optimizer.step()
        
        print("[OK] Integration Test Passed.")

if __name__ == '__main__':
    unittest.main()
