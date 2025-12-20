
import os
import torch
import torch.nn as nn
import torch.optim as optim
import data
import dge_model
from dge_utils import expand_embedding, expand_dge_linear, DGEAdamW
import numpy as np

# Configuration
CONFIG = {
    "output_dir": "results_idk_fast",
    "vocab_size": 1000, # Small vocab for speed
    "d_model": 32,
    "n_layer": 1,
    "n_head": 2,
    "seq_len": 16,
    "lr": 1e-2,
    "idk_steps": 20, 
    "skill_steps": 50, 
    "batch_size": 4,
    "device": "cpu" # Force CPU for reliability
}

def run_experiment():
    print("🚀 FAST IDK TEST")
    DEVICE = CONFIG["device"]

    model = dge_model.DGESimpleTransformer(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        router_type='rbf', 
        initial_gating=False 
    ).to(DEVICE)

    # Expand Vocab
    IDK_TOKEN_ID = CONFIG["vocab_size"]
    expand_embedding(model.token_emb, 1)
    new_lm_head = expand_dge_linear(
        model.lm_head, added_in=0, added_out=1, frozen_core_pos=dge_model.Quadrant.TOP_LEFT
    ).to(DEVICE)
    model.lm_head = new_lm_head
    model.vocab_size += 1
    
    print(f"Vocab: {model.vocab_size}. IDK: {IDK_TOKEN_ID}")

    # IDK Dataset (Synthetic)
    # Just list of strings
    texts = ["This is random text " * 10 for _ in range(100)]
    
    # IDK Dataset
    # Tokenizer None -> Char level (mod vocab)
    idk_ds = data.IDKDataset(texts, None, CONFIG["seq_len"], CONFIG["vocab_size"], idk_token_id=IDK_TOKEN_ID)
    idk_loader = torch.utils.data.DataLoader(idk_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    print("Phase 1: IDK Train")
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
        if step % 5 == 0:
            probs = torch.softmax(logits, dim=-1)
            print(f"Step {step}: Loss {loss.item():.4f}, IDK Prob: {probs[:, IDK_TOKEN_ID].mean().item():.4f}")

    # Phase 2: Skill A
    print("Phase 2: Expand Skill A")
    model.expand_for_skill("TaskA", 32, 'rbf')
    
    # Count Up Data
    X_task = []
    Y_task = []
    for _ in range(100):
        seq = torch.arange(0, CONFIG["seq_len"]) 
        X_task.append(seq)
        Y_task.append(torch.roll(seq, -1))
    X_task = torch.stack(X_task).to(DEVICE)
    Y_task = torch.stack(Y_task).to(DEVICE)
    
    optimizer = DGEAdamW(model.parameters(), lr=CONFIG["lr"])
    
    for step in range(CONFIG["skill_steps"]):
        idx = np.random.choice(len(X_task), CONFIG["batch_size"])
        x, y = X_task[idx], Y_task[idx]
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
             print(f"Skill Step {step}: Loss {loss.item():.4f}")

    # Verification
    print("Verification")
    model.eval()
    
    # Task
    x = X_task[0:1].to(DEVICE)
    logits, _ = model(x)
    pred_task = logits.argmax(dim=-1)
    acc = (pred_task == Y_task[0:1]).float().mean().item()
    print(f"Task Accuracy: {acc:.4f}")
    
    # OOD (Random) -> IDK
    x = torch.randint(10, 50, (1, CONFIG["seq_len"])).to(DEVICE)
    logits, _ = model(x)
    pred_ood = logits.argmax(dim=-1)
    idk_rate = (pred_ood == IDK_TOKEN_ID).float().mean().item()
    print(f"OOD IDK Rate: {idk_rate:.4f}")
    
    if idk_rate > 0.8:
        print("SUCCESS")
    else:
        print("FAILURE")

if __name__ == "__main__":
    run_experiment()
