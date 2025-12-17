import unittest
import os
import shutil
import torch
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

# Import the module to be tested (we'll implement the logic in run_tinystories_gsm8k_chain.py)
# For now, we assume the existence of these functions or we will refactor the script to have them.
# Import the module to be tested
import sys
import os
sys.path.append(os.getcwd()) # Ensure current directory is in path
try:
    import run_tinystories_gsm8k_chain
except ImportError:
    # If running from parent dir
    sys.path.append("Implementation")
    import run_tinystories_gsm8k_chain
class TestMoonLanding(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = "test_moon_landing_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.mock_model = MagicMock()
        self.mock_model.state_dict.return_value = {"weight": "tensor"}
        self.mock_model.d_model = 128
        self.mock_model.layers = [MagicMock()]
        self.mock_model.layers[0].n_head = 4
        self.mock_model.token_emb.num_embeddings = 100
        self.mock_model.max_seq_len = 512
        
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.state_dict.return_value = {"state": "optim_state"}

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch("torch.save")
    @patch("run_tinystories_gsm8k_chain.upload_to_hf_async")
    def test_milestone_checkpoint_weights_only(self, mock_upload, mock_torch_save):
        """
        Requirement: Milestone checkpoints must ONLY contain weights (and config), NO optimizer.
        They must be permanent (unique paths).
        """
        # Mock torch.save to actually create the file so os.path.exists works
        mock_torch_save.side_effect = lambda obj, path: open(path, 'w').close()
        
        from run_tinystories_gsm8k_chain import save_checkpoint
        
        step = 1000
        phase_name = "tinystories_final"
        path = os.path.join(self.output_dir, f"milestone_{phase_name}")
        
        # ACT: Save a milestone checkpoint
        # We need to modify save_checkpoint to accept an 'is_milestone' flag or similar
        # For this test, we assume the current implementation allows selective saving
        # or we will introduce a new function save_milestone.
        # Let's assume we modify save_checkpoint to take `save_optimizer=False` for milestones.
        
        config = {"test": "config"}
        
        # New signature proposal: save_checkpoint(model, optimizer, path, step, config, save_optimizer=True, is_rolling=False)
        # Using a wrapper or modified function in the actual code.
        # Here we test the logic.
        
        # If we use the existing function, we'd expect it to save optimizer. 
        # So we MUST modify the function.
        # Let's pretend we pass save_optimizer=False
        
        # For the sake of the TDD, I'll invoke the function with the expected new argument
        # We might need to handle the ImportError locally if the script isn't updated yet, 
        # but TDD says write test first.
        try:
            save_checkpoint(self.mock_model, self.mock_optimizer, path, step, config, save_optimizer=False, is_rolling=False)
        except TypeError as e:
            self.fail(f"save_checkpoint signature mismatch: {e}")

        # ASSERT
        # 1. Check that torch.save was called for weights
        weights_call = False
        optimizer_call = False
        
        for call_args in mock_torch_save.call_args_list:
            arg_path = call_args[0][1] # second argument is path
            if "weights.pt" in arg_path:
                weights_call = True
            if "optimizer.pt" in arg_path:
                optimizer_call = True
                
        self.assertTrue(weights_call, "Milestone MUST save weights.pt")
        self.assertFalse(optimizer_call, "Milestone MUST NOT save optimizer.pt")
        
        # 2. Check config exists
        self.assertTrue(os.path.exists(os.path.join(path, "config.json")))

    @patch("torch.save")
    @patch("run_tinystories_gsm8k_chain.upload_to_hf_async")
    @patch("shutil.rmtree")
    def test_resume_checkpoint_overwrites(self, mock_rmtree, mock_upload, mock_torch_save):
        """
        Requirement: Resume checkpoint contains EVERYTHING (weights + optimizer).
        It overwrites the previous resume checkpoint on HF Hub (conceptually) 
        and deletes local previous checkpoint to save space.
        """
        # Mock torch.save to actually create the file
        mock_torch_save.side_effect = lambda obj, path: open(path, 'w').close()
        
        from run_tinystories_gsm8k_chain import save_checkpoint
        
        # Setup: Create a "previous" checkpoint existence state
        config = {"test": "config"}
        
        # 1. Save Resume Checkpoint Step 100
        step1 = 100
        path1 = os.path.join(self.output_dir, "resume_checkpoint") # Standard name for rolling
        
        # ACT 1
        save_checkpoint(self.mock_model, self.mock_optimizer, path1, step1, config, save_optimizer=True, is_rolling=True)
        
        # ASSERT 1
        # Check optimizer saved
        optimizer_call = any("optimizer.pt" in call[0][1] for call in mock_torch_save.call_args_list)
        self.assertTrue(optimizer_call, "Resume checkpoint MUST save optimizer.pt")
        
        # Check upload called
        mock_upload.assert_called_with(path1, step1)
        
        # Reset mocks
        mock_torch_save.reset_mock()
        mock_upload.reset_mock()
        
        # 2. Save Resume Checkpoint Step 200 (Simulating next interval with SAME path)
        # Since we use fixed path "resume_checkpoint", it overwrites in place.
        # rmtree should NOT be called (efficiency).
        
        # ACT 2
        save_checkpoint(self.mock_model, self.mock_optimizer, path1, 200, config, save_optimizer=True, is_rolling=True)
            
        # ASSERT 2
        # Should NOT delete path1 explicitly (same path)
        mock_rmtree.assert_not_called()
        
        # Should upload again with new step
        mock_upload.assert_called_with(path1, 200)
    
if __name__ == "__main__":
    unittest.main()
