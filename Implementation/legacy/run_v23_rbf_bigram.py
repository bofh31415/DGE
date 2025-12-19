import torch
import torch.nn as nn
import torch.nn.functional as F
from dge_model import DGESimpleTransformer
from dge_utils import set_seed, Quadrant, DGEAdamW
from dge_training import TaskType
import os
import datetime
import math

class ExperimentLogger:
    def __init__(self, run_name):
        self.run_dir = os.path.join("models", run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "logs", "training.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Header
        with open(self.log_file, "w") as f:
            f.write("Step,Task,Loss,Perplexity,Memory_MB,Fried_Grad_Norm,Gate_Grad_Norm,Max_Gate_Mag,Active_Grad_Norm,Gate_Bias_Max,Gate_Bias_Min,Gate_Bias_Mean,Router_Weight_Norm,Probe_Loss,Probe_PPL,Probe_Gate_Bias_Max,Rescue_Openness_Mean,Rescue_Scale_Mean,Rescue_Scale_Max\n")
            
    def log(self, step, task, loss, ppl, fried_grad, gate_grad, max_gate, active_grad, gate_bias_max, gate_bias_min, gate_bias_mean, router_norm, probe_results=None, rescue_metrics=None):
        def mk_float(v):
            if hasattr(v, 'item'): return v.item()
            if hasattr(v, 'cpu'): return v.cpu().numpy().item() # helper
            return float(v)
            
        loss = mk_float(loss)
        ppl = mk_float(ppl)
        fried_grad = mk_float(fried_grad)
        gate_grad = mk_float(gate_grad)
        max_gate = mk_float(max_gate)
        active_grad = mk_float(active_grad)
        gate_bias_max = mk_float(gate_bias_max)
        gate_bias_min = mk_float(gate_bias_min)
        gate_bias_mean = mk_float(gate_bias_mean)
        router_norm = mk_float(router_norm)
        
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        probe_loss = 0.0
        probe_ppl = 0.0
        probe_gate = 0.0
        if probe_results:
            probe_loss = mk_float(probe_results.get('loss', 0.0))
            probe_ppl = mk_float(probe_results.get('ppl', 0.0))
            probe_gate = mk_float(probe_results.get('gate_max', 0.0))
            
        rescue_open = 0.0
        rescue_scale = 0.0
        rescue_max = 0.0
        if rescue_metrics:
            rescue_open = mk_float(rescue_metrics.get('openness', 0.0))
            rescue_scale = mk_float(rescue_metrics.get('scale_mean', 0.0))
            rescue_max = mk_float(rescue_metrics.get('scale_max', 0.0))
            
        with open(self.log_file, "a") as f:
            f.write(f"{step},{task},{loss:.6f},{ppl:.6f},{mem:.2f},{fried_grad:.9f},{gate_grad:.9f},{max_gate:.4f},{active_grad:.6f},{gate_bias_max:.4f},{gate_bias_min:.4f},{gate_bias_mean:.4f},{router_norm:.4f},{probe_loss:.6f},{probe_ppl:.6f},{probe_gate:.4f},{rescue_open:.6f},{rescue_scale:.2f},{rescue_max:.2f}\n")

def warmup_rbf_centroids(model, x):
    """
    Runs a forward pass to imprint RBF centroids on the provided data.
    Registers a hook on all RBFRouters to capture input stats upon next forward.
    """
    print("  > Registering Imprint Hooks...")
    hooks = []
    
    def imprint_hook(module, input, output):
        # input is tuple (x,)
        # For RBFBigramRouter, x matches input_dim.
        # But wait, RBFBigramRouter.imprint_from_batch calls _make_bigram internally.
        # So it expects raw input x.
        # So we just pass input[0].
        module.imprint_from_batch(input[0])
        print(f"  > Imprinted {module.__class__.__name__} with {input[0].shape[0]} samples.")
        
    count = 0
    for name, module in model.named_modules():
        if "RBF" in module.__class__.__name__ and "Router" in module.__class__.__name__:
            h = module.register_forward_hook(imprint_hook)
            hooks.append(h)
            count += 1
            
    print(f"  > Found {count} RBF Routers to warmup.")
    
    if count > 0:
        model.eval()
        with torch.no_grad():
            model(x)
        
        for h in hooks:
            h.remove()
        print("  > Warmup Complete. Hooks removed.")
    else:
        print("  > No RBF Routers found! Warning.")

def get_batch(task_type, vocab_size, batch_size=32, seq_len=64, device='cpu'):
    # Generate valid sequences that follow the rule/pattern
    # This ensures context (history) is distinct between tasks.
    
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    sequences = [starts]
    
    curr = starts
    for _ in range(seq_len):
        if task_type == TaskType.COUNT_UP:
            nxt = (curr + 1) % vocab_size
        elif task_type == TaskType.COUNT_DOWN:
            nxt = (curr - 1) % vocab_size
        sequences.append(nxt)
        curr = nxt
        
    # Stack: [Batch, Seq+1] (Start + Seq_Len steps)
    full_seq = torch.cat(sequences, dim=1)
    
    # Input: 0..Seq-1
    # Target: 1..Seq
    data = full_seq[:, :-1].contiguous()
    targets = full_seq[:, 1:].contiguous()
    
    return data, targets

def probe_task(model, task_type, vocab_size, device):
    """Checks performance on a task without training."""
    model.eval()
    with torch.no_grad():
        x, y = get_batch(task_type, vocab_size, batch_size=16, seq_len=32, device=device)
        logits, _ = model(x) # Unpack tuple
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        gate_val = 0.0
        try:
            if hasattr(model.layers[0], 'w_mlp_in') and hasattr(model.layers[0].w_mlp_in, 'gate_row'):
                 gate_module = model.layers[0].w_mlp_in.gate_row
                 # Check last mean open
                 if hasattr(gate_module, 'last_mean_open'):
                    gate_val = gate_module.last_mean_open.item() if isinstance(gate_module.last_mean_open, torch.Tensor) else gate_module.last_mean_open
        except:
            pass
        
        return {'loss': loss.item(), 'ppl': torch.exp(loss).item(), 'gate_max': gate_val}

def train_task(model, task_type, vocab_size, steps, logger, start_step, optimizer, probe_task_type=None):
    model.train()
    device = model.token_emb.weight.device
    
    for i in range(steps):
        # try: removed for debugging
        x, y = get_batch(task_type, vocab_size, device=device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        
        try:
            rescue_metrics = model.validate_dge_integrity()
        except Exception:
            rescue_metrics = {}
        
        fried_norm = rescue_metrics.get('Fried_Grad_Norm', 0.0)
        gate_grad_norm = 0.0
        gate_bias_max = 0.0
        gate_bias_min = 0.0
        gate_bias_mean = 0.0
        active_grad_norm = 0.0
        max_gate_mag = 0.0
        
        try:
            l0 = model.layers[0]
            if hasattr(l0, 'w_mlp_in'):
                 if l0.w_mlp_in.weight.grad is not None:
                     active_grad_norm = l0.w_mlp_in.weight.grad.norm().item()
            
            gate_module = None
            if hasattr(l0, 'w_mlp_in') and hasattr(l0.w_mlp_in, 'gate_row'):
                gate_module = l0.w_mlp_in.gate_row
            
            if gate_module and hasattr(gate_module, 'router') and gate_module.router is not None:
                if hasattr(gate_module.router, 'log_beta'):
                     gate_bias_mean = gate_module.router.log_beta.mean().item()
        except Exception:
            pass 
                 
        probe_res = None
        if probe_task_type and i % 10 == 0:
             probe_res = probe_task(model, probe_task_type, vocab_size, device)
             # Re-fetch gate val using hack
             try:
                if hasattr(model.layers[0], 'w_mlp_in') and hasattr(model.layers[0].w_mlp_in, 'gate_row'):
                     gate_module = model.layers[0].w_mlp_in.gate_row
                     if hasattr(gate_module, 'last_mean_open'):
                        gate_val = gate_module.last_mean_open.item() if isinstance(gate_module.last_mean_open, torch.Tensor) else gate_module.last_mean_open
                        if probe_res: probe_res['gate_max'] = gate_val
             except:
                pass
             model.train()
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {steps+i if task_type==TaskType.COUNT_DOWN else i} ({'+'+str(i)}) | Loss: {loss.item():.4f}")
            logger.log(start_step + i, task_type.name, loss.item(), torch.exp(loss).item(), 
                       fried_norm, gate_grad_norm, max_gate_mag, active_grad_norm, 
                       gate_bias_max, gate_bias_min, gate_bias_mean, 0.0,
                       probe_res, rescue_metrics)
        # except Exception:
        #    import traceback
        #    traceback.print_exc()
        #    break
            
    return start_step + steps

def run_experiment():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    vocab_size = 100
    d_model = 32
    n_layer = 2
    n_head = 4
    
    run_name = f"dge_v23_rbf_bigram_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(run_name)
    
    # 1. Pre-train Skill A (Count Up)
    print("Step 1: Training Skill A (Count Up)...")
    model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    
    optimizer = DGEAdamW(model.parameters(), lr=1e-3)
    global_step = train_task(
        model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=0, optimizer=optimizer
    )
    
    # 2. Expand: V23 RBF Bigram Configuration
    print(f"Step 2: Expanding Model (RBF-Bigram Router)...")
    
    current_d_model = model.d_model
    model.expand_model(
        new_input_dim=current_d_model + 128, 
        new_output_dim=model.token_emb.num_embeddings, 
        router_type='rbf_bigram', # V23 Change
        use_orthogonal_init=True,
        use_gradient_rescue=True,
        cross_term_policy='imprint',
    )
    
    # 3. Optimizer
    print(f"Step 3: Optimizer Setup...")
    optimizer = DGEAdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # 3.5 Warmup RBF Centroids
    # Imprint on Skill B data (using bigram logic internally)
    # V23 Fix: Use Large Batch to cover all 100 bigram transitions
    print(f"Step 3.5: Warming up RBF-Bigram Centroids (Imprinting)...")
    warmup_batch_x, _ = get_batch(TaskType.COUNT_DOWN, vocab_size, batch_size=500, seq_len=2, device=device)
    warmup_rbf_centroids(model, warmup_batch_x)
    
    # 4. Train Skill B (Count Down)
    print(f"Step 4: Training Skill B (Count Down) + Probing Skill A...")
    train_task(
        model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
        logger=logger, start_step=global_step, optimizer=optimizer,
        probe_task_type=TaskType.COUNT_UP # Monitor Forgetting
    )
    
    print(f"Experiment Complete. Check {logger.log_file}")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception:
        import traceback
        with open("crash.log", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
