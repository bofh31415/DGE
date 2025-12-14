import os
import torch
from datetime import datetime
from dge_model import DGESimpleTransformer
from dge_training import train_task, TaskType
from dge_utils import DGEAdamW
from dge_logger import DGELogger

def run_h4():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dge_exp_h4_manual_{timestamp}"
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    exp_dir = os.path.join(models_dir, model_name)
    
    # Initialize Logger (This will use the UPDATE dge_logger.py on disk)
    logger = DGELogger(exp_dir)
    
    # Defaults
    vocab_size = 500
    d_model = 64 
    n_layer = 2
    n_head = 4
    
    print(f"Creating model {model_name}...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    logger.log_event("CREATED", {"d_model": d_model}, step=0)
    
    # Optimizer
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    global_step = 0
    
    # 1. Train Skill A
    print("Training Skill A (COUNT_UP)...")
    global_step = train_task(
        model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer
    )
    
    # 2. Expand (H4)
    print("Expanding model with Block Diagonal Isolation (H4)...")
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        isolate_cross_terms=True 
    )
    
    # Re-init Optimizer
    print("Re-initializing Optimizer...")
    router_params = [p for n, p in model.named_parameters() if 'router' in n or 'gate' in n]
    default_params = [p for n, p in model.named_parameters() if 'router' not in n and 'gate' not in n]
    optimizer = DGEAdamW([{'params': default_params, 'lr': 1e-3}, {'params': router_params, 'lr': 1e-4}], weight_decay=0.0)
    
    # 3. Train Skill B
    print("Training Skill B (COUNT_DOWN)...")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
    )
    
    print(f"Experiment Complete. Log: {exp_dir}")

if __name__ == "__main__":
    run_h4()
