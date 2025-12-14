import torch
import torch.optim as optim
import time
from enum import Enum
import math
import os

try:
    import psutil
except ImportError:
    psutil = None

class TaskType(Enum):
    COUNT_UP = 1
    COUNT_DOWN = 2

def generate_batch(task_type, vocab_size, batch_size, seq_len):
    """
    Generates a batch of data for the specified task.
    """
    if task_type == TaskType.COUNT_UP:
        # [n, n+1, n+2] -> n+3
        max_start = max(1, vocab_size - seq_len - 1)
        start_nums = torch.randint(0, max_start, (batch_size, 1))
        offset = torch.arange(seq_len).unsqueeze(0)
        x = start_nums + offset
        y = x + 1
        
    elif task_type == TaskType.COUNT_DOWN:
        # [n, n-1, n-2] -> n-3
        # Start must be at least seq_len
        min_start = seq_len + 1
        # vocab_size must be > seq_len
        if vocab_size <= min_start:
             # Fallback if vocab too small, though unlikely for valid test
             x = torch.zeros(batch_size, seq_len, dtype=torch.long)
             y = torch.zeros(batch_size, seq_len, dtype=torch.long)
             return x, y
             
        start_nums = torch.randint(min_start, vocab_size, (batch_size, 1))
        offset = torch.arange(seq_len).unsqueeze(0) # [0, 1, 2...]
        x = start_nums - offset # [n, n-1, n-2...]
        y = x - 1
        
    else:
        raise ValueError("Unknown TaskType")
        
    # Clamp to vocab
    x = torch.clamp(x, 0, vocab_size - 1)
    y = torch.clamp(y, 0, vocab_size - 1)
    
    return x, y

def train_task(model, task_type, vocab_size=1000, steps=500, batch_size=32, seq_len=32, 
               logger=None, start_step=0, checkpoint_fn=None, optimizer=None, probe_task_type=None,
               sparsity_lambda=0.05):
    print(f"\n--- Starting Training: {task_type.name} ({steps} steps) ---")
    
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        print("⚠️ Warning: Using default AdamW instead of passed optimizer.")
        
    model.train()
    start_time = time.time()
    
    # Initialize process for memory tracking
    process = psutil.Process(os.getpid()) if psutil else None
    
    # Use existing unified log
    current_step = start_step # Initialize to start
    
    for i in range(steps):
        current_step = start_step + i
        
        x, y = generate_batch(task_type, vocab_size, batch_size, seq_len)
        
        optimizer.zero_grad()
        logits, loss = model(x, y, sparsity_lambda=sparsity_lambda)
        loss.backward()
        optimizer.step()
        
        # --- Forensic Logging ---
        # --- Forensic Logging ---
        if logger:
            # 1. Capture Training Metrics (from the backward pass just finished)
            train_metrics = model.validate_dge_integrity()
            
            # 2. Cross-Task Probing (Forensic Analysis)
            probe_loss = 0.0
            probe_ppl = 0.0
            if probe_task_type:
                # Capture state *specifically* for probe inputs
                model.eval()
                with torch.no_grad():
                    xp, yp = generate_batch(probe_task_type, vocab_size, batch_size, seq_len)
                    # Forward pass updates gate statistics (last_mean_open)
                    _, ploss = model(xp, yp)
                    probe_loss = ploss.item()
                    probe_ppl = math.exp(probe_loss) if probe_loss < 100 else float('inf')
                    
                    # Capture metrics NOW (post-probe forward) to get Probe Gate Activity
                    probe_metrics_raw = model.validate_dge_integrity()
                    
                    # Namespace them
                    for k, v in probe_metrics_raw.items():
                        if "Gate" in k or "Router" in k: # Only care about routing behavior for probe
                            train_metrics[f"Probe_{k}"] = v
                            
                model.train()
                # Re-run forward on training data? No, just proceed. 
                # Note: last_mean_open is now from probe. Next training step will overwrite it.
                # Ideally, we should maybe restore it, but for simple logging it's fine 
                # as long as we know 'train_metrics' captured it BEFORE probe.
            
            # Metrics
            loss_val = loss.item()
            ppl = math.exp(loss_val) if loss_val < 100 else float('inf')
            
            mem_mb = 0.0
            if process:
                mem_mb = process.memory_info().rss / 1024 / 1024
            
            # Add probe scalar metrics
            if probe_task_type:
                train_metrics["Probe_Loss"] = probe_loss
                train_metrics["Probe_PPL"] = probe_ppl
                
            logger.log_training_step(current_step, task_type.name, loss_val, ppl, mem_mb, train_metrics)
        # ------------------------
        
        if i % 50 == 0 or i == steps - 1:
            print(f"Step {current_step} (+{i}) | Loss: {loss.item():.4f}")
        
        # Checkpoint every 500 steps
        if checkpoint_fn and (i > 0 and i % 500 == 0):
            print(f"[Checkpoint] Saving at step {current_step}...")
            checkpoint_fn(current_step)
            
    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f}s")
    
    if logger:
        loss_val = loss.item()
        ppl = math.exp(loss_val) if loss_val < 100 else float('inf')
        summary = {
            "task": task_type.name,
            "steps_trained": steps,
            "start_step": start_step,
            "end_step": current_step,
            "final_loss": loss_val,
            "final_ppl": ppl,
            "duration_sec": duration
        }
        logger.log_event("TRAINED", summary, step=current_step)
        
    # Final checkpoint
    if checkpoint_fn:
        print(f"[Checkpoint] Saving final state at step {current_step}...")
        checkpoint_fn(current_step)
        
    return current_step + 1 # Return next available step (e.g. if ended at 499, return 500)

def evaluate_task(model, task_type, vocab_size=1000, samples=100, seq_len=32):
    """
    Evaluates accuracy on the specified task.
    """
    model.eval()
    correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        # Generate one large batch or multiple small ones
        # Use single batch for simplicity of 'samples'
        x, y = generate_batch(task_type, vocab_size, samples, seq_len)
        
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        
        # Check all tokens? Or just the next prediction logic?
        # Standard accuracy: compare pred match with y
        matches = (preds == y)
        correct = matches.sum().item()
        total_tokens = matches.numel()
        
    acc = correct / total_tokens * 100.0
    model.train()
    return acc
