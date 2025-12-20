import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_german_tinystories, load_mgsm, HF_AVAILABLE
from core.training import train_task, TaskType
from core.model import DGESimpleTransformer

class TestV09Integration(unittest.TestCase):
    
    @unittest.skipIf(not HF_AVAILABLE, "HuggingFace not available")
    @patch('data.load_dataset')
    def test_load_german_tinystories(self, mock_load_dataset):
        """Test that German TinyStories loader calls correct HF dataset."""
        print("\nTesting German TinyStories Loader...")
        
        # Mock the dataset return
        mock_data = [{'text': 'Hallo Welt'}] * 10
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: mock_data[idx]
        # Allow iteration
        mock_dataset.__iter__.return_value = iter(mock_data)
        mock_dataset.select.return_value = mock_dataset
        
        mock_load_dataset.return_value = mock_dataset
        
        # Call loader
        loader = load_german_tinystories(max_samples=5, tokenizer_name=None, batch_size=2)
        
        # Verify call
        mock_load_dataset.assert_called_with("SkySyrup/tinystories_german", split='train')
        self.assertIsNotNone(loader)
        print("✅ German TinyStories Loader OK")

    @unittest.skipIf(not HF_AVAILABLE, "HuggingFace not available")
    @patch('data.load_dataset')
    def test_load_mgsm(self, mock_load_dataset):
        """Test MGSM loader."""
        print("\nTesting MGSM Loader...")
        
        mock_data = [{'question': '1+1?', 'answer': '2', 'answer_number': 2}]
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.__iter__.return_value = iter(mock_data)
        
        mock_load_dataset.return_value = mock_dataset
        
        loader = load_mgsm(lang='de', tokenizer_name=None)
        
        mock_load_dataset.assert_called_with("juletxara/mgsm", 'de', split='test')
        self.assertIsNotNone(loader)
        print("✅ MGSM Loader OK")

    @patch('torch.cuda.device_count')
    @patch('torch.nn.DataParallel')
    def test_multi_gpu_trigger(self, mock_dp, mock_device_count):
        """Test that multi-GPU logic triggers when >1 GPU is detected."""
        print("\nTesting Multi-GPU Trigger...")
        
        # Simulate 2 GPUs
        mock_device_count.return_value = 2
        
        model = DGESimpleTransformer(d_model=16, n_layer=1, n_head=2, vocab_size=100)
        
        # We need to mock train_task's internal logic or extract the wrapping logic?
        # Since train_task is a function, we can run it briefly with steps=0 or mock the loop.
        # But wait, train_task wraps the model internally.
        
        # Let's mock the optimizer and loop to avoid real training
        with patch('dge_training.optim.AdamW'), \
             patch('dge_training.generate_batch', return_value=(torch.zeros(2,5).long(), torch.zeros(2,5).long())):
            
            try:
                # We expect DataParallel to be called
                train_task(model, TaskType.COUNT_UP, steps=1, device=torch.device('cpu'))
            except Exception:
                pass # We just want to check the mock call
            
            # Verify DataParallel was initialized
            mock_dp.assert_called()
            print("✅ Multi-GPU Trigger OK")

if __name__ == '__main__':
    unittest.main()
