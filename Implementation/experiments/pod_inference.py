#!/usr/bin/env python3
"""
Pod Inference Script
====================
Standalone script for SSH-based model inference on RunPod.

Usage:
1. SSH into your RunPod instance
2. Run: python experiments/pod_inference.py
3. Select model, chat with it

No local dependencies required - runs entirely on the pod.
"""

import os
import sys
import json
import torch
from huggingface_hub import hf_hub_download, HfApi

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import DGESimpleTransformer

HF_REPO = os.getenv("HF_REPO", "darealSven/dge")
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "models/inference_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_setup():
    """Validate environment setup."""
    if not HF_TOKEN:
        print("\n❌ HF_TOKEN not set!")
        print("\n📋 Setup Instructions:")
        print("   1. Get your token: https://huggingface.co/settings/tokens")
        print("   2. Set it: export HF_TOKEN=hf_YOUR_TOKEN_HERE")
        print("   3. Run this script again")
        print("\n💡 Quick fix:")
        print("   export HF_TOKEN=hf_xxx && python experiments/pod_inference.py")
        return False
    return True

def scan_models():
    """Scan HF repo for available models."""
    print(f"\n🔍 Scanning {HF_REPO}...")
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(HF_REPO, token=HF_TOKEN)
        
        configs = {f for f in files if f.endswith("/config.json") or f == "config.json"}
        weights = {f for f in files if f.endswith("/weights.pt") or f == "weights.pt"}
        
        models = []
        if "config.json" in configs and "weights.pt" in weights:
            models.append(".")
        for cfg in configs:
            prefix = os.path.dirname(cfg)
            if prefix and f"{prefix}/weights.pt" in weights:
                models.append(prefix)
        models.sort()
        return models
    except Exception as e:
        print(f"❌ Error scanning repo: {e}")
        return []

def load_model(prefix):
    """Load model from HF."""
    print(f"\n🔄 Loading {prefix}...")
    
    # Download config
    config_file = hf_hub_download(
        HF_REPO,
        filename=f"{prefix}/config.json" if prefix != "." else "config.json",
        local_dir=CACHE_DIR,
        token=HF_TOKEN
    )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Build model
    model = DGESimpleTransformer(
        vocab_size=config.get("vocab_size", 50257),
        d_model=config.get("d_model", 384),
        n_layer=config.get("n_layer", 12),
        n_head=config.get("n_head", 6),
        max_seq_len=config.get("max_seq_len", 1024)
    )
    
    # Download weights
    weights_file = hf_hub_download(
        HF_REPO,
        filename=f"{prefix}/weights.pt" if prefix != "." else "weights.pt",
        local_dir=CACHE_DIR,
        token=HF_TOKEN
    )
    
    # Load state dict
    state_dict = torch.load(weights_file, map_location=DEVICE)
    
    # NEW: Handle Architectural Shifting (Legacy Gated Head -> Hierarchical Head)
    # If checkpoint has 'lm_head.weight' but model expects 'lm_head.base_head.weight'
    if "lm_head.weight" in state_dict and "lm_head.base_head.weight" not in state_dict:
        print("   🔍 Remapping legacy lm_head to Hierarchical structure...")
        new_sd = state_dict.copy()
        
        # Map main weights/bias
        new_sd["lm_head.base_head.weight"] = new_sd.pop("lm_head.weight")
        if "lm_head.bias" in new_sd:
            new_sd["lm_head.base_head.bias"] = new_sd.pop("lm_head.bias")
            
        # Drop masks and gates that are only in the old MoEGatedLinear head
        keys_to_drop = [
            "lm_head.active_mask", "lm_head.frozen_bias_mask", "lm_head.backward_mask",
            "lm_head.gate_row.old_gate", "lm_head.gate_col.old_gate",
            "lm_head.gate_row.router.weight", "lm_head.gate_row.router.bias",
            "lm_head.gate_col.router.weight", "lm_head.gate_col.router.bias"
        ]
        for k in keys_to_drop:
            if k in new_sd:
                new_sd.pop(k)
        
        state_dict = new_sd

    # Load
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ Loaded on {DEVICE}")
    return model, config

def generate_text(model, prompt, max_tokens=100):
    """Generate text from prompt."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except:
        print("❌ transformers not installed")
        return None
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_tokens):
            ctx = generated[:, -model.max_seq_len:]
            logits, _ = model(ctx)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    print("""
╔══════════════════════════════════════════════════╗
║      DGE Pod Inference - Interactive Chat        ║
╚══════════════════════════════════════════════════╝
""")
    
    print(f"📡 Device: {DEVICE}")
    print(f"🏠 Repo: {HF_REPO}")
    
    # Check setup
    if not check_setup():
        return
    
    # Scan models
    models = scan_models()
    if not models:
        print("\n❌ No models found!")
        return
    
    print(f"\n✅ Found {len(models)} models:\n")
    for i, m in enumerate(models, 1):
        display_name = m if m != "." else "(root)"
        print(f"  {i}. {display_name}")
    
    # Select model
    try:
        choice = int(input("\nSelect model #: ").strip())
        if choice < 1 or choice > len(models):
            print("❌ Invalid selection")
            return
        selected = models[choice - 1]
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Cancelled")
        return
    
    # Load model
    try:
        model, config = load_model(selected)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return
    
    # Chat loop
    print("\n" + "="*50)
    print("💬 CHAT MODE")
    print("="*50)
    print("Type 'exit' or Ctrl+C to quit\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                break
            
            if not prompt:
                continue
            
            print("   Generating...", end='', flush=True)
            output = generate_text(model, prompt, max_tokens=100)
            print(f"\r{' '*20}\r", end='')  # Clear line
            
            if output:
                print(f"AI:  {output}\n")
            else:
                print("❌ Generation failed\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    print("\nSession ended.")

if __name__ == "__main__":
    main()
