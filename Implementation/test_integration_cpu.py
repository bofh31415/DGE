
import os
import shutil
import torch
import tempfile
import unittest
from dge_model import DGESimpleTransformer
from dge_training import train_dataset
from replay_buffer import ReplayBuffer
import data
from torch.utils.data import DataLoader


def test_end_to_end_training():
    print("\n--- Test: End-to-End Dataset Training ---")
    
    # Create temp dir
    test_dir = tempfile.mkdtemp()
    original_data_store = data.DATA_STORE_PATH
    data.DATA_STORE_PATH = test_dir
    
    try:
        # Create a dummy text file
        text_file = os.path.join(test_dir, "dummy.txt")
        with open(text_file, "w") as f:
            # Write enough text for a few chunks
            f.write("Alice went to the store to buy some apples. " * 100)

        # 1. Import Data
        print("Importing data...")
        dataset_path = data.import_text_file(text_file, local_name="test_dataset", chunk_size=50)
        
        # 2. Load Data
        print("Loading data...")
        dataloader = data.load_local_dataset("test_dataset", seq_len=32, batch_size=4, vocab_size=256)
        
        # 3. Initialize Model
        print("Initializing model...")
        model = DGESimpleTransformer(vocab_size=256, d_model=32, n_layer=1, n_head=2)
        
        # 4. Initialize Replay Buffer
        replay_buffer = ReplayBuffer(max_size=100, task_name="test_replay")
        
        # 5. Train
        print("Training...")
        # Train for 2 epochs
        train_dataset(
            model=model,
            dataloader=dataloader,
            epochs=2,
            replay_buffer=replay_buffer,
            replay_ratio=0.1,
            task_name="test_run"
        )
        
        # 6. Verify Replay Buffer
        print(f"Replay Buffer Size: {len(replay_buffer)}")
        if len(replay_buffer) > 0:
            print("✅ Replay buffer populated.")
        else:
            print("❌ Replay buffer empty!")
            exit(1)
        
        # 7. Verify we can sample from it
        x_r, y_r = replay_buffer.sample(2)
        print(f"Sampled Replay Shape: {x_r.shape}")
        
        print("✅ End-to-End Test Passed!")
        
    finally:
        shutil.rmtree(test_dir)
        data.DATA_STORE_PATH = original_data_store

if __name__ == "__main__":
    test_end_to_end_training()

