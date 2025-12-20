import os
import torch
from datetime import datetime
from core.model import DGESimpleTransformer
from core.training import train_task, TaskType
from core.utils import DGEAdamW
from utils.logger import DGELogger

def run_v9_boost():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V9 BOOST: Clamp 100 + VERY HIGH LR (1e-2) + Low Sparsity (0.01)")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dge_v9_boost_{timestamp}"
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
    
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    global_step = train_task(
        model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=0, optimizer=optimizer
    )
    
    # 2. Expand: Zero Init + Clamped Rescue (Implicit in dge_utils)
    print(f"Step 2: Expanding Model (Bias=-4.0, Zero Init)...")
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=True,
        router_init_bias=-4.0
    )
    
    # 3. Optimizer with VERY HIGH LR for Router
    print(f"Step 3: Optimizer Setup (Router LR=1e-2)...")
    router_params = []
    default_params = []
    for n, p in model.named_parameters():
        if 'router' in n or 'gate' in n:
            router_params.append(p)
        else:
            default_params.append(p)
    
    # 1e-2 for Router/Gates (Massive Boost)
    optimizer = DGEAdamW([
        {'params': default_params, 'lr': 1e-3},
        {'params': router_params, 'lr': 1e-2} 
    ], weight_decay=0.0)
    
    # 4. Train Skill B with LOW Sparsity
    print(f"Step 4: Training Skill B (Sparsity=0.01)...")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP, 
        sparsity_lambda=0.01
    )
    
    print(f"✅ V9 Experiment Complete. Logs in {exp_dir}")

if __name__ == "__main__":
    run_v9_boost()
