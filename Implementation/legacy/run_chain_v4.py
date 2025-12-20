import os
import torch
import torch.nn.functional as F
from datetime import datetime
from core.model import DGESimpleTransformer
from core.training import train_task, TaskType
from core.utils import DGEAdamW
from utils.logger import DGELogger
import time

def run_experiment_config(name, config):
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING CONFIG: {name}")
    print(f"Hyperparameters: {config}")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dge_v4_{name.lower().replace(' ', '_')}_{timestamp}"
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    exp_dir = os.path.join(models_dir, model_name)
    
    logger = DGELogger(exp_dir)
    
    # Defaults
    vocab_size = 500
    d_model = 64 
    n_layer = 2
    n_head = 4
    
    # 1. Create & Train Skill A
    print("Step 1: Training Skill A (COUNT_UP)...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    # Basic LR for Phase 1
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    global_step = 0
    
    global_step = train_task(
        model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer
    )
    
    # Verify Skill A
    # We assume it learns to 100%, but good to check.
    
    # 2. Expand with Config
    print(f"Step 2: Expanding Model (Bias={config['router_init_bias']})...")
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=True,
        router_init_bias=config['router_init_bias']
    )
    
    # 3. Re-init Optimizer with Differential LRs
    print(f"Step 3: Optimizer Setup (Router LR={config['router_lr']})...")
    router_params = []
    default_params = []
    for n, p in model.named_parameters():
        if 'router' in n or 'gate' in n:
            router_params.append(p)
        else:
            default_params.append(p)
            
    optimizer = DGEAdamW([
        {'params': default_params, 'lr': 1e-3},
        {'params': router_params, 'lr': config['router_lr']}
    ], weight_decay=0.0)
    
    # 4. Train Skill B with Sparsity
    print(f"Step 4: Training Skill B (Sparsity={config['sparsity_lambda']})...")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP, 
        sparsity_lambda=config['sparsity_lambda']
    )
    
    # 5. Final Report
    print(f"✅ Experiment {name} Complete. Logs in {exp_dir}")


def run_chain_v4():
    configs = {
        "Baseline":        {"sparsity_lambda": 0.05, "router_init_bias": -4.0, "router_lr": 1e-4},
        "High_Plasticity": {"sparsity_lambda": 0.01, "router_init_bias": -2.0, "router_lr": 5e-4},
        "Max_Stability":   {"sparsity_lambda": 0.20, "router_init_bias": -4.0, "router_lr": 5e-5},
        "Open_Gates":      {"sparsity_lambda": 0.05, "router_init_bias": -2.0, "router_lr": 1e-4},
        "No_Sparsity":     {"sparsity_lambda": 0.00, "router_init_bias": -4.0, "router_lr": 1e-4},
        "Balanced":        {"sparsity_lambda": 0.02, "router_init_bias": -3.0, "router_lr": 2e-4},
    }
    
    print("🧪 STARTING EXPERIMENT CHAIN V4: THE PLASTICITY SWEEP")
    for name, config in configs.items():
        try:
            run_experiment_config(name, config)
        except Exception as e:
            print(f"❌ Config {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_chain_v4()
