
import os
import torch
from datetime import datetime
from core.model import DGESimpleTransformer
from core.training import train_task, TaskType
from core.utils import DGEAdamW
from utils.logger import DGELogger

# V 0.15.0: Longevity Chain Experiment
# Tests if Skill 1 remains 100% active after N sequential expansions.

def evaluate_skill_1(model, vocab_size):
    """Probes Skill 1 (Count Up) and returns accuracy."""
    model.eval()
    correct = 0
    total = 50
    with torch.no_grad():
        for _ in range(total):
            start = torch.randint(0, vocab_size - 5, (1, 1)).item()
            prompt = torch.tensor([[start, start+1, start+2]], device=model.token_emb.weight.device)
            expected = start + 3
            logits, _ = model(prompt)
            pred = logits[0, -1, :].argmax().item()
            if pred == expected:
                correct += 1
    model.train()
    return correct / total

def run_longevity_chain(num_skills=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"longevity_chain_{timestamp}"
    exp_dir = os.path.join("models", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = DGELogger(exp_dir)

    vocab_size = 500
    d_model = 64
    
    print(f"🚀 Starting Longevity Chain: {num_skills} sequential skills...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=2, n_head=4)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Train Initial Skill (Skill 0)
    print("Training Skill 0 (Core: Count Up)...")
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    train_task(model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=300, logger=logger, optimizer=optimizer)
    
    history = []
    acc0 = evaluate_skill_1(model, vocab_size)
    history.append(acc0)
    print(f"Skill 0 accuracy: {acc0:.2%}")

    # 2. Sequential Expansion Loop
    for i in range(1, num_skills):
        print(f"\n--- Expanding for Skill {i} ---")
        
        # Freeze existing skills
        # In current API, we freeze by ID if we use expand_for_skill
        # But here we use a simpler manual loop with freeze_skill on the last one.
        # Actually expand_for_skill handles head creation too in V0.14.0.
        
        # Note: Skill ID for current (soon to be frozen) is i-1
        model.freeze_skill(i-1)
        
        # Expand
        model.expand_for_skill(f"Skill_{i}", router_type="rbf")
        
        # Training a generic alternative task (Count Down) or similar
        # For a stress test, the exact task doesn't matter as much as the interference check.
        # We alternate between UP and DOWN to prevent biasing.
        task = TaskType.COUNT_DOWN if i % 2 == 1 else TaskType.COUNT_UP
        
        optimizer = DGEAdamW(model.parameters(), lr=1e-3)
        train_task(model, task, vocab_size=vocab_size, steps=200, logger=logger, optimizer=optimizer)
        
        # Verify Skill 0 (Stability check)
        acc0 = evaluate_skill_1(model, vocab_size)
        history.append(acc0)
        print(f"Skill 0 accuracy after {i} expansions: {acc0:.2%}")
        
        if acc0 < 0.8:
            print(f"⚠️ DANGER: Significant forgetting detected at Skill {i}!")

    print("\n✅ Longevity Chain Complete.")
    print(f"Stability History: {history}")
    
    # Save results
    with open(os.path.join(exp_dir, "longevity_results.json"), "w") as f:
        import json
        json.dump({"history": history}, f)

if __name__ == "__main__":
    run_longevity_chain()
