import os
import torch
from datetime import datetime
from dge_model import DGESimpleTransformer
from dge_training import train_task, TaskType
from dge_utils import DGEAdamW
from dge_logger import DGELogger

def run_v17_moderate():
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING V17 MODERATE: Clean Architecture + Safe LR (1e-3)")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dge_v17_moderate_{timestamp}"
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
    
    # 2. Expand: Complete Protection Suite
    # - Sep QKV (Implicit in model)
    # - Split LN (Implicit in model expansion)
    # - Zero Cross Terms (isolate_cross_terms=True)
    # - Frozen Embeddings & PosEmb (Patched dge_utils)
    # - Gradient Rescue (Enabled)
    print(f"Step 2: Expanding Model (Bias=-4.0, Zero Init, Clean Cross, Sep QKV, Split LN)...")
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 32, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='mlp',
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        cross_term_policy='imprint', # V20: Asymmetric Shielding (Active BL, Frozen TR)
        router_init_bias=-4.0 # V20: Start Closed to protect Identity. Rescue will open it.
    )
    
    # 3. Optimizer with MODERATE LR
    print(f"Step 3: Optimizer Setup (Router LR=1e-3, Default LR=1e-3)...")
    # We DO NOT boost Router LR to 1e-2. We keep it 1e-3 to prevent explosion.
    # Gradient Rescue should handle the signal strength.
    optimizer = DGEAdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # 4. Train Skill B with LOW Sparsity
    print(f"Step 4: Training Skill B (Sparsity=0.01)...")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP, 
        sparsity_lambda=0.01
    )
    
    print(f"✅ V17 Experiment Complete. Logs in {exp_dir}")

if __name__ == "__main__":
    run_v17_moderate()
