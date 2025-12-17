import torch
import data
import dge_model
import torch.optim as optim
import neuro_bodybuilding
import time
import os
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN, etc.)
load_dotenv()

def run_experiment():
    print("\n" + "="*60)
    print("🏋️ NEURO-BODYBUILDING EXPERIMENT (Sparsity Tuning)")
    print("="*60)
    
    # 1. Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    from types import SimpleNamespace
    config = SimpleNamespace(
        vocab_size=50257,
        d_model=384,
        n_layer=6, # Small for testing
        n_head=6,
        batch_size=16,
        seq_len=128
    )
    
    # 2. Model
    print("Creating Model...")
    model = dge_model.DGESimpleTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layer=config.n_layer,
        n_head=config.n_head,
        max_seq_len=config.seq_len
    ).to(device)
    # Ideally load pre-trained weights here
    # model.load_state_dict(torch.load("..."))
    
    # 3. Data
    print("Loading Dataset (Standard Format)...")
    try:
        # Load the processed standard dataset
        dataloader = data.load_local_dataset(
            'german_psycho_train', 
            seq_len=config.seq_len, 
            batch_size=config.batch_size, 
            tokenizer_name='gpt2'
        )
        print("✅ Loaded 'german_psycho_train'")
    except Exception as e:
        print(f"⚠️ Primary dataset not found or error ({e}). Using TinyStories fallback.")
        dataloader = data.load_tinystories(
            split='train', 
            max_samples=1000, 
            seq_len=config.seq_len, 
            batch_size=config.batch_size
        )

    
    # 4. Trainer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Define simple CrossEntropy wrapper
    # DGE model returns (logits, loss) usually, need to check
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = neuro_bodybuilding.NeuroTrainer(model, dataloader, criterion, optimizer, device)
    
    # 5. Loop
    print("\nStarting Workout...")
    epochs = 5
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # The trainer handles FWD, GA Masking, Backprop, Step
            loss, fitness = trainer.train_step(x, y)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss:.4f} | Fitness: {fitness:.4f}")
                
            total_steps += 1
            if total_steps >= 200: # Short proof of concept
                break
                
    end_time = time.time()
    print(f"\n✅ Workout Complete. Time: {end_time - start_time:.2f}s")
    print("Check loss history to verify learning under sparsity constraints.")

if __name__ == "__main__":
    run_experiment()
