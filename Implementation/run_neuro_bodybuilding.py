import torch
import data
import dge_model
import torch.optim as optim
import neuro_bodybuilding
import time
import os

def run_experiment():
    print("\n" + "="*60)
    print("🏋️ NEURO-BODYBUILDING EXPERIMENT (Sparsity Tuning)")
    print("="*60)
    
    # 1. Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = dge_model.DGEConfig(
        vocab_size=50257,
        d_model=384,
        n_layer=6, # Small for testing
        n_head=6,
        batch_size=16,
        seq_len=128
    )
    
    # 2. Model
    print("Creating Model...")
    model = dge_model.DGETransformer(config).to(device)
    # Ideally load pre-trained weights here
    # model.load_state_dict(torch.load("..."))
    
    # 3. Data
    print("Loading TinyStories...")
    # Using a small subset/split for the experiment
    dataloader = data.load_tinystories(split="train", batch_size=config.batch_size, seq_len=config.seq_len, max_samples=1000)
    
    # 4. Trainer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Define simple CrossEntropy wrapper
    # DGE model returns (logits, loss) usually, need to check
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = neuro_bodybuilding.NeuroTrainer(model, dataloader, criterion, optimizer, device)
    
    # 5. Loop
    print("\nStarting Workout...")
    epochs = 1
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
            if total_steps >= 50: # Short proof of concept
                break
                
    end_time = time.time()
    print(f"\n✅ Workout Complete. Time: {end_time - start_time:.2f}s")
    print("Check loss history to verify learning under sparsity constraints.")

if __name__ == "__main__":
    run_experiment()
