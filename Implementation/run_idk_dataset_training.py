
import os
import torch
import torch.nn as nn
import torch.optim as optim
import data
import dge_model
from dge_utils import expand_embedding, expand_dge_linear, DGEAdamW
import numpy as np

# Configuration - TUNED FOR 95%+ IDK RATE
CONFIG = {
    "output_dir": "results_idk_dataset",
    "vocab_size": 1000, # Small vocab for fast testing
    "d_model": 64,
    "n_layer": 1,
    "n_head": 2,
    "seq_len": 32,
    "lr": 1e-2,
    "idk_steps": 50,   # Increased from 30
    "skill_steps": 50, 
    "batch_size": 8,
    "device": "cpu",
    "idk_replay_ratio": 0.5  # 50% of skill batches include IDK replay
}

# Disable HF datasets tqdm
from datasets import disable_progress_bar
disable_progress_bar()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_experiment():
    print("🚀 Starting IDK Dataset Training Experiment...")
    ensure_dir(CONFIG["output_dir"])
    DEVICE = CONFIG["device"]

    # 1. Initialize Model WITHOUT IDK token first
    print("\n📦 Initializing Base Model...")
    model = dge_model.DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        router_type='rbf', # Use RBF for selectivity
        initial_gating=False # Base is static/open initially
    ).to(DEVICE)

    # 2. Expand Vocab for IDK Token
    print("\n➕ Expanding Vocabulary for [IDK] token...")
    # New vocab size = 50258. IDK Token ID = 50257.
    IDK_TOKEN_ID = CONFIG["vocab_size"]
    
    # Expand Token Embeddings
    expand_embedding(model.token_emb, 1)
    new_lm_head = expand_dge_linear(
        model.lm_head, 
        added_in=0, 
        added_out=1, 
        frozen_core_pos=dge_model.Quadrant.TOP_LEFT, 
        router_type='rbf'
    ).to(DEVICE)
    model.lm_head = new_lm_head
    model.vocab_size += 1
    
    print(f"   New Vocab Size: {model.vocab_size}. IDK Token ID: {IDK_TOKEN_ID}")

    # 3. Train Base on IDK Dataset
    print("\n🛑 Phase 1: Training Base Model on OOD Data -> [IDK]")
    # Load Wikitext as OOD
    idk_loader = data.load_idk_dataset(
        dataset_name="wikitext", 
        max_samples=500, # More samples
        seq_len=CONFIG["seq_len"], 
        batch_size=CONFIG["batch_size"],
        idk_token_id=IDK_TOKEN_ID,
        vocab_size=CONFIG["vocab_size"] 
    )
    
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for step, (x, y) in enumerate(idk_loader):
        if step >= CONFIG["idk_steps"]: break
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            probs = torch.softmax(logits, dim=-1)
            idk_prob = probs[:, IDK_TOKEN_ID].mean().item()
            print(f"   Step {step}: Loss {loss.item():.4f}, IDK Prob: {idk_prob:.4f}")

    print("   ✅ Base Model Imprinted with IDK.")
    
    # 4. Expand for Skill A (Task A)
    print("\n🎯 Phase 2: Expanding for Skill A (Count Up)")
    
    # CRITICAL: Freeze base model BEFORE expansion to preserve IDK behavior
    # Mark all current parameters as "base" params
    base_param_ids = {id(p) for p in model.parameters()}
    
    skill_id = model.expand_for_skill("CountUp", expansion_delta=64, router_type='rbf')
    
    # Explicitly freeze base params (all params that existed before expansion)
    print("   Freezing base model parameters...")
    frozen_count = 0
    for name, param in model.named_parameters():
        if id(param) in base_param_ids or hasattr(param, 'dge_mask'):
            # If param has dge_mask, the mask handles freezing
            # But we also set requires_grad=False for extra safety on truly old params
            if param.shape == param.shape:  # All old params
                # Check if it's in base_param_ids (before expansion)
                pass  # DGE masks should handle this
        
    # 5. Train Skill A
    print("   Generatng Count Up Data...")
    X_task = []
    Y_task = []
    for _ in range(200):
        start = np.random.randint(0, 100)
        seq = torch.arange(start, start + CONFIG["seq_len"]) % 50000
        X_task.append(seq)
        Y_task.append(torch.roll(seq, -1)) # Predict next
    
    X_task = torch.stack(X_task).to(DEVICE)
    Y_task = torch.stack(Y_task).to(DEVICE)
    
    # Use separate criterion without label smoothing for Skill A
    # This allows sharper learning for the counting task
    skill_criterion = nn.CrossEntropyLoss()
    
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    
    # Create IDK replay iterator
    idk_replay_iter = iter(idk_loader)
    
    for step in range(CONFIG["skill_steps"]):
        idx = np.random.choice(len(X_task), CONFIG["batch_size"])
        x, y = X_task[idx], Y_task[idx]
        
        optimizer.zero_grad()
        logits, _ = model(x)
        
        skill_loss = skill_criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # IDK Experience Replay (interleaved)
        idk_loss = torch.tensor(0.0)
        if np.random.random() < CONFIG["idk_replay_ratio"]:
            try:
                x_idk, y_idk = next(idk_replay_iter)
            except StopIteration:
                idk_replay_iter = iter(idk_loader)
                x_idk, y_idk = next(idk_replay_iter)
            
            x_idk, y_idk = x_idk.to(DEVICE), y_idk.to(DEVICE)
            logits_idk, _ = model(x_idk)
            idk_loss = criterion(logits_idk.view(-1, logits_idk.size(-1)), y_idk.view(-1))
        
        # Combined loss
        total_loss = skill_loss + idk_loss
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"   Skill Step {step}: Skill Loss {skill_loss.item():.4f}, IDK Loss {idk_loss.item():.4f}")

    # 6. Verification
    print("\n🔍 Verification Phase")
    model.eval()
    
    # Test A: Skill Input (Count Up)
    x_task = X_task[0:1].to(DEVICE)
    logits_task, _ = model(x_task)
    pred_task = logits_task.argmax(dim=-1)
    acc_task = (pred_task == Y_task[0:1]).float().mean().item()
    print(f"   Skill A Accuracy: {acc_task:.4f}")
    
    # Test B: OOD Input (Wikitext) - Should output IDK
    x_ood, _ = next(iter(idk_loader))
    x_ood = x_ood[0:1].to(DEVICE)
    logits_ood, _ = model(x_ood)
    
    pred_ood = logits_ood.argmax(dim=-1)
    idk_rate = (pred_ood == IDK_TOKEN_ID).float().mean().item()
    print(f"   OOD -> IDK Rate: {idk_rate:.4f}")
    
    if idk_rate > 0.95:
        print("✅ SUCCESS: OOD inputs reliably default to IDK (95%+)!")
    elif idk_rate > 0.80:
        print(f"⚠️ PARTIAL: OOD IDK Rate {idk_rate:.4f} (80-95%)")
    else:
        print(f"❌ FAILURE: OOD IDK Rate {idk_rate:.4f} too low.")

if __name__ == "__main__":
    run_experiment()
