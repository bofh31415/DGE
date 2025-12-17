#!/usr/bin/env python3
"""
HuggingFace Restorepoint Utilities
==================================
Shared utilities for downloading pre-trained model checkpoints from HuggingFace Hub.
This allows experiments to skip baseline training if a restorepoint is available.
"""

import os

# Known HF repos with pre-trained TinyStories base models
HF_TINYSTORIES_REPOS = [
    "darealSven/dge-tinystories-gsm8k",        # GSM8K experiment (has milestone_tinystories)
    "darealSven/dge-tinystories-german-psycho", # German Psycho experiment
]


def download_hf_restorepoint(repo_id, folder_name, local_path):
    """
    Download a specific folder (restorepoint) from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "darealSven/dge-tinystories-gsm8k")
        folder_name: Name of the folder in the repo (e.g., "milestone_tinystories")
        local_path: Local path to download to
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        # Check if folder exists in repo
        files = list_repo_files(repo_id)
        folder_files = [f for f in files if f.startswith(folder_name + "/")]
        
        if not folder_files:
            print(f"   ⚠️ Restorepoint '{folder_name}' not found in {repo_id}")
            return False
        
        print(f"   📥 Downloading restorepoint '{folder_name}' from {repo_id}...")
        os.makedirs(local_path, exist_ok=True)
        
        for remote_file in folder_files:
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_file,
                local_dir=os.path.dirname(local_path),
                local_dir_use_symlinks=False
            )
        
        print(f"   ✅ Restorepoint downloaded to {local_path}")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Could not download restorepoint: {e}")
        return False


def check_for_tinystories_restorepoint(output_dir, ensure_restored_fn):
    """
    Check if a pre-trained TinyStories model exists locally or on HF Hub.
    Downloads from HF Hub if available but not locally.
    
    Args:
        output_dir: Local output directory for the experiment
        ensure_restored_fn: Function to verify checkpoint is restored (handles chunked archives)
        
    Returns:
        Path to restorepoint if available, None otherwise.
    """
    local_milestone = os.path.join(output_dir, "milestone_tinystories")
    
    # 1. Check Local First
    if ensure_restored_fn(local_milestone):
        print("   ✅ Found local TinyStories restorepoint.")
        return local_milestone
    
    # 2. Check HuggingFace Hub (try each known repo)
    print("   🔍 Checking HuggingFace Hub for pre-trained TinyStories...")
    for repo_id in HF_TINYSTORIES_REPOS:
        try:
            if download_hf_restorepoint(repo_id, "milestone_tinystories", local_milestone):
                if ensure_restored_fn(local_milestone):
                    return local_milestone
        except Exception as e:
            print(f"   ⚠️ Could not check {repo_id}: {e}")
            continue
    
    print("   ℹ️ No TinyStories restorepoint found. Will train from scratch.")
    return None


def generate_model_card(output_dir, experiment_name, config, verification_results=None):
    """
    Generate a HuggingFace Model Card (README.md) from experiment data.
    
    Args:
        output_dir: Directory containing experiment results
        experiment_name: Human-readable name for the experiment
        config: Experiment configuration dictionary
        verification_results: Optional validation results dictionary
    """
    import json
    from datetime import datetime
    
    # Try to load experiment results if they exist
    results_path = os.path.join(output_dir, "experiment_results.json")
    experiment_log = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                experiment_log = json.load(f)
        except:
            pass
    
    # Build Model Card
    card = f"""---
license: apache-2.0
tags:
  - dge
  - dynamic-growth-expansion
  - continual-learning
  - pytorch
language:
  - en
  - de
---

# {experiment_name}

This model was trained using **Dynamic Growth Expansion (DGE)**, a novel architecture for continual learning without catastrophic forgetting.

## Model Details

| Property | Value |
|----------|-------|
| **Base d_model** | {config.get('tinystories_d_model', 'N/A')} |
| **Expanded d_model** | {config.get('gsm8k_d_model', config.get('psycho_d_model', 'N/A'))} |
| **Layers** | {config.get('n_layer', 'N/A')} |
| **Context Length** | {config.get('max_seq_len', 'N/A')} |
| **Vocab Size** | {config.get('vocab_size', 'N/A')} |

## Training

- **Phase 1**: Trained on TinyStories (English baseline)
- **Phase 2**: Expanded model using DGE
- **Phase 3**: Fine-tuned on target task with asymmetric replay

## Evaluation Results

"""

    # Add verification results if available
    if verification_results:
        card += "| Checkpoint | Dataset | PPL | Loss |\n"
        card += "|------------|---------|-----|------|\n"
        for ckpt_name, datasets in verification_results.items():
            for ds_name, metrics in datasets.items():
                if isinstance(metrics, dict):
                    ppl = metrics.get('ppl', 'N/A')
                    loss = metrics.get('loss', 'N/A')
                    card += f"| {ckpt_name} | {ds_name} | {ppl:.2f} | {loss:.4f} |\n"
    else:
        card += "*Evaluation results will be added after experiment completion.*\n"

    card += """
## Usage

```python
from dge_model import DGESimpleTransformer
import torch

# Load the model
model = DGESimpleTransformer(...)
model.load_state_dict(torch.load("weights.pt"))
```

## Citation

If you use this model, please cite the DGE paper (forthcoming).
"""

    # Write to file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    
    print(f"📝 Model Card generated: {readme_path}")
    return readme_path
