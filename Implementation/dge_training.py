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

# Device detection - automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """Return the current training device (cuda or cpu)."""
    return DEVICE

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
               sparsity_lambda=0.05, **kwargs):
    print(f"\n--- Starting Training: {task_type.name} ({steps} steps) ---")
    
    # Device handling - move model to GPU if available
    device = DEVICE
    model = model.to(device)
    print(f"🖥️ Training on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        print("⚠️ Warning: Using default AdamW instead of passed optimizer.")
        
    model.train()
    start_time = time.time()
    
    # Initialize process for memory tracking
    process = psutil.Process(os.getpid()) if psutil else None
    
    # Use existing unified log
    current_step = start_step # Initialize to start
    
    replay_tasks = kwargs.get('replay_tasks', [])
    replay_ratio = kwargs.get('replay_ratio', 1.0) # Default to 1:1 if not specified
    replay_debt = 0.0

    
    for i in range(steps):
        current_step = start_step + i
        
        # Determine if this step is for Replay
        is_replay_step = False
        if len(replay_tasks) > 0:
            if replay_ratio >= 1.0:
                 # Legacy/Default 50/50 Mode (Odd steps)
                 is_replay_step = (i % 2 != 0)
            else:
                 # Sparse Replay Mode (Accumulator)
                 # We want P(Replay) = replay_ratio
                 # Add ratio to debt. If debt >= 1, trigger replay.
                 # But we must balance the loop.
                 # Actually, simpler: if debt >= 1.0, this step is Replay.
                 replay_debt += replay_ratio
                 if replay_debt >= 1.0:
                     is_replay_step = True
                     replay_debt -= 1.0
        
        if is_replay_step:
            # Replay Logic: Train Router Only
            # Pick a task (round robin or random - use simple modulo for now)
            # Use a separate counter for replay steps to ensure round robin works smoothly? 
            # Or just use i. Using i is fine for random, but for round robin with sparse steps, 
            # we should use a counter.
            replay_idx = (i // 1) % len(replay_tasks) # Simplified
            r_task = replay_tasks[replay_idx]
            
            x, y = generate_batch(r_task, vocab_size, batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y, sparsity_lambda=sparsity_lambda)
            loss.backward()
            
            # ASYMMETRIC REPLAY: Zero out gradients for non-router parameters
            # This forces the router to learn to CLOSE for the old task (rejection)
            for name, param in model.named_parameters():
                if "router" not in name and "gate" not in name:
                    param.grad = None
            
            optimizer.step()
            
            # For logging/printing, we might want to skip or log specially
            # But to keep main loop simple, let's just proceed.
            # Maybe print specific message?
            # print(f"  [Replay {r_task.name}] Loss: {loss.item():.4f}")
            
        else:
            # Main Task Logic: Train Everything
            x, y = generate_batch(task_type, vocab_size, batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            
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
                    xp, yp = xp.to(device), yp.to(device)
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


def train_dataset(model, dataloader, epochs=1, optimizer=None, logger=None, 
                  start_step=0, checkpoint_fn=None, checkpoint_interval=500,
                  replay_buffer=None, replay_ratio=0.1, sparsity_lambda=0.05,
                  task_name="dataset", auto_populate_buffer=True):
    """
    Train model on a PyTorch DataLoader with epoch/batch loops.
    
    Supports automatic ReplayBuffer population and Asymmetric Replay.
    
    Args:
        model: DGE model to train.
        dataloader: PyTorch DataLoader providing (x, y) batches.
        epochs: Number of full passes through the dataset.
        optimizer: Optimizer to use (creates default AdamW if None).
        logger: Optional DGELogger for forensic logging.
        start_step: Starting global step number.
        checkpoint_fn: Callback for checkpointing (called with step number).
        checkpoint_interval: Steps between checkpoints.
        replay_buffer: Optional ReplayBuffer for Asymmetric Replay.
        replay_ratio: Probability of replay step (0.0-1.0).
        sparsity_lambda: Gate sparsity penalty weight.
        task_name: Name of this training task for logging.
        auto_populate_buffer: If True, add samples to replay_buffer during training.
        
    Returns:
        Next available step number.
    """
    from replay_buffer import ReplayBuffer, estimate_replay_ratio
    
    print(f"\n--- Starting Dataset Training: {task_name} ({epochs} epochs) ---")
    
    # Device handling - move model to GPU if available
    device = DEVICE
    model = model.to(device)
    print(f"🖥️ Training on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        print("⚠️ Warning: Using default AdamW instead of passed optimizer.")
    
    model.train()
    start_time = time.time()
    
    # Initialize process for memory tracking
    process = psutil.Process(os.getpid()) if psutil else None
    
    current_step = start_step
    replay_debt = 0.0
    total_batches = len(dataloader) * epochs
    
    # Auto-estimate replay ratio if buffer provided but ratio not specified
    if replay_buffer and not replay_buffer.is_empty() and replay_ratio is None:
        replay_ratio = estimate_replay_ratio(replay_buffer, model)
        print(f"📊 Auto-estimated replay ratio: {replay_ratio:.2%}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # --- Determine if this is a Replay Step ---
            is_replay_step = False
            if replay_buffer and not replay_buffer.is_empty() and replay_ratio:
                if replay_ratio >= 1.0:
                    is_replay_step = (current_step % 2 != 0)
                else:
                    replay_debt += replay_ratio
                    if replay_debt >= 1.0:
                        is_replay_step = True
                        replay_debt -= 1.0
            
            if is_replay_step:
                # --- Asymmetric Replay: Train Router Only ---
                x_r, y_r = replay_buffer.sample(x.size(0))
                x_r, y_r = x_r.to(device), y_r.to(device)
                
                optimizer.zero_grad()
                logits, loss = model(x_r, y_r, sparsity_lambda=sparsity_lambda)
                loss.backward()
                
                # Zero out non-router gradients
                for name, param in model.named_parameters():
                    if "router" not in name and "gate" not in name:
                        param.grad = None
                
                optimizer.step()
            else:
                # --- Main Task Training ---
                optimizer.zero_grad()
                logits, loss = model(x, y, sparsity_lambda=sparsity_lambda)
                loss.backward()
                optimizer.step()
                
                # Auto-populate replay buffer
                if auto_populate_buffer and replay_buffer is not None:
                    replay_buffer.add(x.detach(), y.detach())
            
            epoch_loss += loss.item()
            batch_count += 1
            current_step += 1
            
            # --- Logging ---
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
            # --- Checkpointing ---
            if checkpoint_fn and current_step > 0 and current_step % checkpoint_interval == 0:
                print(f"[Checkpoint] Saving at step {current_step}...")
                checkpoint_fn(current_step)
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} Complete | Avg Loss: {avg_loss:.4f}")
    
    duration = time.time() - start_time
    print(f"Dataset training finished in {duration:.2f}s ({current_step - start_step} steps)")
    
    # Log summary
    if logger:
        summary = {
            "task": task_name,
            "epochs": epochs,
            "start_step": start_step,
            "end_step": current_step,
            "final_loss": loss.item() if 'loss' in dir() else 0,
            "duration_sec": duration,
            "replay_ratio": replay_ratio,
            "buffer_size": len(replay_buffer) if replay_buffer else 0
        }
        logger.log_event("TRAINED_DATASET", summary, step=current_step)
    
    # Final checkpoint
    if checkpoint_fn:
        print(f"[Checkpoint] Saving final state at step {current_step}...")
        checkpoint_fn(current_step)
    
    return current_step + 1


def evaluate_task(model, task_type, vocab_size=1000, samples=100, seq_len=32):
    """
    Evaluates accuracy on the specified task.
    """
    # Get device from model
    device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        # Generate one large batch or multiple small ones
        # Use single batch for simplicity of 'samples'
        x, y = generate_batch(task_type, vocab_size, samples, seq_len)
        x, y = x.to(device), y.to(device)
        
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

