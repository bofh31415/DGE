"""
Progress tracking for RunPod experiments.
Writes progress to /workspace/progress.json for remote monitoring.
"""
import json
import os
from datetime import datetime

PROGRESS_FILE = "/workspace/progress.json"

def update_progress(
    stage: int,
    total_stages: int,
    stage_name: str,
    status: str = "running",
    epoch: int = 0,
    total_epochs: int = 0,
    batch: int = 0,
    total_batches: int = 0,
    loss: float = None,
    extra: dict = None
):
    """
    Update the progress file with current training state.
    
    Args:
        stage: Current stage number (1-indexed)
        total_stages: Total number of stages
        stage_name: Name of the current stage
        status: "running", "completed", "failed"
        epoch: Current epoch (if applicable)
        total_epochs: Total epochs (if applicable)
        batch: Current batch (if applicable)
        total_batches: Total batches (if applicable)
        loss: Current loss value (if applicable)
        extra: Any additional data to include
    """
    # Calculate overall percentage
    stage_weight = 100 / total_stages
    stage_progress = ((stage - 1) / total_stages) * 100
    
    if total_epochs > 0 and epoch > 0:
        epoch_progress = (epoch / total_epochs) * stage_weight
    elif total_batches > 0 and batch > 0:
        epoch_progress = (batch / total_batches) * stage_weight
    else:
        epoch_progress = 0
    
    overall_percent = stage_progress + epoch_progress
    
    progress = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "total_stages": total_stages,
        "stage_name": stage_name,
        "status": status,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "loss": loss,
        "overall_percent": round(overall_percent, 1),
        "extra": extra or {}
    }
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update progress file: {e}")


def get_progress():
    """Read the current progress from file."""
    try:
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: Could not read progress file: {e}")
        return None


def clear_progress():
    """Clear the progress file."""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    except Exception:
        pass
