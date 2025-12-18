"""
HuggingFace Base Models Utility (V 0.1.0)
=========================================
Shared utilities for downloading/uploading pre-trained base models
to avoid redundant training across experiments.

Usage:
    from hf_base_models import get_or_train_base_model
    
    model, was_pretrained = get_or_train_base_model(
        model, "tinystories", d_model=384, n_head=6, n_layer=12,
        train_fn=lambda m: train_tinystories(m)
    )
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Callable, Tuple

# Base repository for shared models
BASE_REPO = "darealSven/dge-base-models"

def get_model_key(dataset: str, d_model: int, n_head: int, n_layer: int) -> str:
    """
    Generate unique key for model configuration.
    
    Args:
        dataset: Training dataset name (e.g., "tinystories")
        d_model: Model dimension
        n_head: Number of attention heads
        n_layer: Number of layers
        
    Returns:
        Unique key like "tinystories_384_6head_12layer"
    """
    return f"{dataset}_{d_model}_{n_head}head_{n_layer}layer"


def download_base_if_exists(
    dataset: str,
    d_model: int, 
    n_head: int, 
    n_layer: int,
    local_cache: str = "models/base_cache"
) -> Optional[str]:
    """
    Download pre-trained base model from HuggingFace if it exists.
    
    Args:
        dataset: Dataset the base was trained on
        d_model, n_head, n_layer: Model config
        local_cache: Local directory to cache downloads
        
    Returns:
        Path to weights.pt if found, None otherwise
    """
    model_key = get_model_key(dataset, d_model, n_head, n_layer)
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        hf_token = os.environ.get("HF_TOKEN")
        
        # Check if model exists in repo
        try:
            files = list_repo_files(BASE_REPO, token=hf_token)
        except Exception:
            print(f"   ℹ️ Base repo {BASE_REPO} not accessible")
            return None
        
        weights_path = f"{model_key}/weights.pt"
        if weights_path not in files:
            print(f"   ℹ️ No pre-trained base found for {model_key}")
            return None
        
        # Download
        print(f"   ⬇️ Downloading pre-trained base: {model_key}")
        os.makedirs(local_cache, exist_ok=True)
        
        local_weights = hf_hub_download(
            BASE_REPO, 
            weights_path, 
            token=hf_token,
            local_dir=local_cache
        )
        
        # Also download config if exists
        config_path = f"{model_key}/config.json"
        if config_path in files:
            hf_hub_download(BASE_REPO, config_path, token=hf_token, local_dir=local_cache)
        
        print(f"   ✅ Downloaded pre-trained base from HuggingFace")
        return local_weights
        
    except ImportError:
        print("   ⚠️ huggingface_hub not installed")
        return None
    except Exception as e:
        print(f"   ⚠️ Could not download base: {e}")
        return None


def upload_base_model(
    model: torch.nn.Module,
    dataset: str,
    d_model: int,
    n_head: int,
    n_layer: int,
    step: int,
    temp_dir: str = "models/base_upload_temp"
) -> bool:
    """
    Upload trained base model to shared HuggingFace repository.
    
    Args:
        model: The trained model
        dataset: Dataset it was trained on
        d_model, n_head, n_layer: Model config
        step: Training step reached
        temp_dir: Temporary directory for staging
        
    Returns:
        True if upload succeeded
    """
    model_key = get_model_key(dataset, d_model, n_head, n_layer)
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("   ⚠️ No HF_TOKEN, skipping base model upload")
            return False
        
        api = HfApi(token=hf_token)
        
        # Ensure repo exists
        try:
            create_repo(BASE_REPO, token=hf_token, private=True, exist_ok=True)
        except Exception:
            pass  # Repo might already exist
        
        # Prepare upload
        upload_path = os.path.join(temp_dir, model_key)
        os.makedirs(upload_path, exist_ok=True)
        
        # Save weights
        torch.save(model.state_dict(), os.path.join(upload_path, "weights.pt"))
        
        # Save config
        config = {
            "dataset": dataset,
            "d_model": d_model,
            "n_head": n_head,
            "n_layer": n_layer,
            "step": step,
            "uploaded_at": datetime.now().isoformat(),
        }
        with open(os.path.join(upload_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Upload
        print(f"   ☁️ Uploading base model: {model_key}")
        api.upload_folder(
            folder_path=upload_path,
            path_in_repo=model_key,
            repo_id=BASE_REPO,
            repo_type="model",
            commit_message=f"Base model {model_key} at step {step}"
        )
        
        print(f"   ✅ Uploaded base model to {BASE_REPO}/{model_key}")
        
        # Cleanup
        import shutil
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to upload base model: {e}")
        return False


def get_or_train_base_model(
    model: torch.nn.Module,
    dataset: str,
    d_model: int,
    n_head: int,
    n_layer: int,
    train_fn: Callable[[torch.nn.Module], int],
    device: torch.device = None
) -> Tuple[torch.nn.Module, bool, int]:
    """
    Get pre-trained base or train from scratch and upload.
    
    Args:
        model: Initialized model
        dataset: Dataset name
        d_model, n_head, n_layer: Model config
        train_fn: Function that trains model, returns final step
        device: Device to load model on
        
    Returns:
        (model, was_pretrained, step)
    """
    # Try to download existing base
    weights_path = download_base_if_exists(dataset, d_model, n_head, n_layer)
    
    if weights_path and os.path.exists(weights_path):
        print(f"   🎯 Loading pre-trained {dataset} base model...")
        model.load_state_dict(torch.load(weights_path, map_location=device or "cpu"))
        if device:
            model = model.to(device)
        
        # Read step from config if available
        config_path = weights_path.replace("weights.pt", "config.json")
        step = 0
        if os.path.exists(config_path):
            with open(config_path) as f:
                step = json.load(f).get("step", 0)
        
        return model, True, step
    
    # Train from scratch
    print(f"   🏋️ Training {dataset} base from scratch...")
    step = train_fn(model)
    
    # Upload for future use
    upload_base_model(model, dataset, d_model, n_head, n_layer, step)
    
    return model, False, step
