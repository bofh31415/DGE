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
    psycho_path = os.path.join("data_store", "german_psycho.jsonl")
    if os.path.exists(psycho_path):
        print(f"🧠 Loading German Psycho Dataset from {psycho_path}...")
        try:
            from transformers import AutoTokenizer
            from torch.utils.data import DataLoader
            import german_psycho_data
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            dataset = german_psycho_data.GermanPsychoDataset(psycho_path, tokenizer, max_length=config.seq_len)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            print(f"   Loaded {len(dataset)} psycho-samples.")
        except Exception as e:
            print(f"   Error loading psycho data: {e}. Fallback to TinyStories.")
            dataloader = data.load_tinystories(split="train", batch_size=config.batch_size, seq_len=config.seq_len, max_samples=1000)
    else:
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
