import unittest
import torch
import shutil
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dge_model import DGESimpleTransformer
from dge_utils import Quadrant
from dge_training import TaskType, generate_batch
from main import DGELab

class TestDGEModel(unittest.TestCase):
    def test_init(self):
        model = DGESimpleTransformer(vocab_size=100, d_model=32, n_layer=1, n_head=4)
        self.assertEqual(model.d_model, 32)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.token_emb.num_embeddings, 100)

    def test_forward(self):
        model = DGESimpleTransformer(vocab_size=100, d_model=32)
        x = torch.zeros((1, 10), dtype=torch.long)
        logits, _ = model(x)
        self.assertEqual(logits.shape, (1, 10, 100))

    def test_model_expansion(self):
        model = DGESimpleTransformer(d_model=32, n_head=4) # head_dim=8
        
        # Expand by 32 (should add 32/8 = 4 heads)
        model.expand_model(added_d_model=32, quadrant=Quadrant.TOP_LEFT)
        
        self.assertEqual(model.d_model, 64)
        self.assertEqual(model.layers[0].n_head, 8)
        # Check shapes
        self.assertEqual(model.token_emb.weight.shape, (1000, 64))
        # Forward check
        x = torch.zeros((1, 10), dtype=torch.long)
        logits, _ = model(x)
        self.assertEqual(logits.shape, (1, 10, 1000))

class TestDGETraining(unittest.TestCase):
    def test_batch_generation_up(self):
        x, y = generate_batch(TaskType.COUNT_UP, vocab_size=100, batch_size=2, seq_len=5)
        self.assertEqual(x.shape, (2, 5))
        self.assertTrue(torch.all(y == x + 1))
        
    def test_batch_generation_down(self):
        x, y = generate_batch(TaskType.COUNT_DOWN, vocab_size=100, batch_size=2, seq_len=5)
        self.assertEqual(x.shape, (2, 5))
        self.assertTrue(torch.all(y == x - 1))

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_models')
        os.makedirs(self.test_dir, exist_ok=True)
        self.lab = DGELab()
        self.lab.models_dir = self.test_dir
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_load(self):
        # Create and Save
        self.lab.model = DGESimpleTransformer(vocab_size=50, d_model=16)
        self.lab.model_name = "UnitTestModel"
        self.lab.save_model()
        
        # Verify file existence
        save_path = os.path.join(self.test_dir, "UnitTestModel")
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "weights.pt")))
        
        # Load
        # Clear current model
        self.lab.model = None
        
        # Load using the actual application logic
        self.lab.load_model_by_name("UnitTestModel")
        
        loaded_model = self.lab.model
        
        self.assertIsNotNone(loaded_model)
        
        self.assertEqual(loaded_model.d_model, 16)
        self.assertEqual(loaded_model.token_emb.num_embeddings, 50)

if __name__ == '__main__':
    unittest.main()
