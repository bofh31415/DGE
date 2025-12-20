import sys
import os
import torch
from main import DGELab, TaskType
from core.training import train_task, evaluate_task

def run_experiment():
    print("\n" + "="*60)
    print("🧪 DGE REPLAY RATIO SENSITIVITY EXPERIMENT")
    print("="*60)
    print("Goal: Determine minimal replay data needed for stability.")
    print("Ratios: 50%, 10%, 1%, 0.75%, 0.5%, 0.2%, 0.05%, 0.01%")
    
    # User requested percentages
    ratios_percent = [50.0, 10.0, 1.0, 0.75, 0.5, 0.2, 0.05, 0.01]
    
    results = []
    
    # Standard Config
    d_model = 64
    vocab_size = 1000
    steps_a = 500
    steps_b = 2000 # Increased steps to stress-test sparse replay over time
    
    for pct in ratios_percent:
        ratio = pct / 100.0
        print(f"\n---------------------------------------------------")
        print(f"🔄 TESTING REPLAY RATIO: {pct}% (Prob: {ratio:.5f})")
        print(f"---------------------------------------------------")
        
        lab = DGELab()
        # Create fresh model manually to match menu creation
        # Using standard 64/1000 defaults
        lab.model_name = f"replay_exp_{pct}"
        from core.model import DGESimpleTransformer
        lab.model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=2, n_head=4)
        lab.model_name = f"replay_ratio_{pct}"
        lab.optimizer = torch.optim.AdamW(lab.model.parameters(), lr=1e-3, weight_decay=0.0) # Match create_model logic
        
        # 1. Train Task A
        print(f"1. Training Task A (Count Up) - {steps_a} steps...")
        train_task(lab.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=steps_a, optimizer=lab.optimizer)
        lab.trained_skills.add(TaskType.COUNT_UP.name)
        acc_a_init = evaluate_task(lab.model, TaskType.COUNT_UP, vocab_size=vocab_size)
        print(f"   Task A Accuracy (Init): {acc_a_init:.2f}%")
        
        if acc_a_init < 99.0:
            print("   ⚠️ Warning: Initialization failed? Continuing...")
            
        # 2. Expand
        print(f"2. Expanding Model (+64)...")
        lab.model.expand_model(
            new_input_dim=d_model + 64, 
            new_output_dim=vocab_size,
            router_type='bigram',
            router_init_bias=0.0,
            use_gradient_rescue=True,
            use_orthogonal_init=False,
            isolate_cross_terms=False
        )
        # Re-init optimizer (Standard procedure in 'expand_menu')
        # We let train_task handle optimizer creation if None, or create here?
        # main.py expand_menu creates specific optimizer. We should mimic that.
        router_params = []
        default_params = []
        for name, param in lab.model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            else:
                default_params.append(param)
        from core.utils import DGEAdamW
        lab.optimizer = DGEAdamW([
            {'params': default_params, 'lr': 1e-3},
            {'params': router_params, 'lr': 1e-4} 
        ], weight_decay=0.0)
        
        # 3. Train Task B with Replay Ratio
        print(f"3. Training Task B (Count Down) - {steps_b} steps with Ratio {ratio}...")
        train_task(
            lab.model, 
            TaskType.COUNT_DOWN, 
            vocab_size=vocab_size, 
            steps=steps_b,
            optimizer=lab.optimizer,
            replay_tasks=[TaskType.COUNT_UP],
            replay_ratio=ratio # <-- The Variable
        )
        
        # 4. Evaluate
        acc_a_final = evaluate_task(lab.model, TaskType.COUNT_UP, vocab_size=vocab_size)
        acc_b_final = evaluate_task(lab.model, TaskType.COUNT_DOWN, vocab_size=vocab_size)
        
        print(f"   RESULTS: A={acc_a_final:.2f}%, B={acc_b_final:.2f}%")
        results.append({
            "Ratio (%)": pct,
            "Replay Steps (Approx)": int(steps_b * ratio),
            "Task A (Stability)": acc_a_final,
            "Task B (Plasticity)": acc_b_final
        })
        
    print("\n" + "="*80)
    print(f"{'Ratio (%)':<10} | {'Replay Steps':<15} | {'Stability (A)':<15} | {'Plasticity (B)':<15}")
    print("-" * 80)
    for res in results:
        print(f"{res['Ratio (%)']:<10.2f} | {res['Replay Steps']:<15} | {res['Task A (Stability)']:<15.2f} | {res['Task B (Plasticity)']:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    run_experiment()
