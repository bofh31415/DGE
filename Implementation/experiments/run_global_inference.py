#!/usr/bin/env python3
"""
Global Model Inference
======================
Scans the HuggingFace repository for ALL available DGE models and checkpoints,
running inference on each to verify behavior and quality.

Three Modes:
1. Automated Suite: Runs standard prompts (Story, German, Math).
2. Broadcast Manual: User inputs one prompt, it runs on ALL models.
3. Interactive Chat: User selects ONE model and chats with it.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.model import DGESimpleTransformer

HF_REPO = "darealSven/dge"
CACHE_DIR = "models/inference_cache"
REPORT_FILE = "global_inference_report.md"

PROMPTS = [
    {"name": "📖 Story",  "text": "Once upon a time, there was a little robot who"},
    {"name": "🇩🇪 German", "text": "Es war einmal, ein kleiner Roboter, der"},
    {"name": "🧮 Math",   "text": "Question: What is 15 + 7? Answer:"}
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scan_repository():
    """Find all valid checkpoint prefixes in the repo."""
    print(f"\n🔍 Scanning {HF_REPO}...")
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(HF_REPO)
    except Exception as e:
        print(f"❌ Failed to list repo files: {e}")
        return []

    # Identify directories containing both config.json and weights.pt
    configs = {f for f in files if f.endswith("/config.json") or f == "config.json"}
    weights = {f for f in files if f.endswith("/weights.pt") or f == "weights.pt"}
    
    # Found prefixes
    valid_prefixes = []
    
    # Handle root config (if any)
    if "config.json" in configs and "weights.pt" in weights:
        valid_prefixes.append(".")
        
    # Handle subdirectories
    for cfg in configs:
        prefix = os.path.dirname(cfg)
        if prefix == "": continue
        
        # Check if corresponding weights exist
        weight_path = f"{prefix}/weights.pt"
        if weight_path in weights:
            valid_prefixes.append(prefix)
            
    # Sort for consistent order (Foundations first, then experiments)
    valid_prefixes.sort()
    return valid_prefixes

def generate_text(model, tokenizer_name, prompt_text, max_new_tokens=50):
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    except:
        return "[Error: Tokenizer not found]"

    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context
            ctx = generated[:, -model.max_seq_len:]
            
            logits, _ = model(ctx) # shape (B, T, V)
            next_token_logits = logits[:, -1, :]
            
            # Greedy
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def load_and_run(prefix, prompts, max_tokens=50, verbose=True):
    """
    Load model 'prefix' and run list of prompts.
    prompts: list of {"name": str, "text": str}
    Returns list of strings (results).
    """
    if verbose: print(f"\n👉 Processing: {prefix}")
    local_dir = os.path.join(CACHE_DIR, prefix)
    os.makedirs(local_dir, exist_ok=True)
    
    results = []
    
    try:
        # Download Config
        if verbose: print("   ⬇️ Downloading Config...")
        config_file = hf_hub_download(HF_REPO, filename=f"{prefix}/config.json" if prefix != "." else "config.json", local_dir=CACHE_DIR)
        
        with open(config_file, 'r') as cf:
            config = json.load(cf)
            
        # Construct model
        if verbose: print(f"   🏗️ Building Model (d_model={config.get('d_model')})...")
        model = DGESimpleTransformer(
            vocab_size=config.get("vocab_size", 50257),
            d_model=config.get("d_model", 384),
            n_layer=config.get("n_layer", 12),
            n_head=config.get("n_head", 6),
            max_seq_len=config.get("max_seq_len", 1024)
        )
        
        # Download Weights
        if verbose: print("   ⬇️ Downloading Weights...")
        weights_file = hf_hub_download(HF_REPO, filename=f"{prefix}/weights.pt" if prefix != "." else "weights.pt", local_dir=CACHE_DIR)
        
        # Load
        model.load_state_dict(torch.load(weights_file, map_location=DEVICE))
        model.to(DEVICE)
        
        if verbose: print("   🔮 Generating...")
        
        for prompt in prompts:
            output = generate_text(model, "gpt2", prompt["text"], max_new_tokens=max_tokens)
            output_flat = output.replace("\n", " ") 
            results.append(f"**{prompt['name']}:** {output_flat}")
            if verbose: print(f"      {prompt['name']}: {output_flat[:60]}...")
            
        # Cleanup VRAM
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        if verbose: print(f"   ❌ FAILED: {e}")
        results.append(f"❌ **ERROR**: {e}")
        
    return results

def run_automated_suite(prefixes):
    print("\n🚀 Running Automated Suite (Story, German, Math)...")
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Global Inference Report\n")
        f.write(f"Date: {os.environ.get('DATE', 'Today')}\n\n")
        
    for prefix in prefixes:
        results = load_and_run(prefix, PROMPTS)
        
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## {prefix}\n")
            for res in results:
                f.write(f"- {res}\n")
            f.write("\n")
            
    print(f"\n📄 Report saved to {REPORT_FILE}")

def run_broadcast_mode(prefixes):
    print("\n📡 Broadcast Mode: Type a prompt, and I'll feed it to EVERY model.")
    prompt_text = input("Enter Prompt: ")
    if not prompt_text: return
    
    prompts = [{"name": "Manual", "text": prompt_text}]
    
    print("\n🚀 Broadcasting...")
    filename = "global_manual_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Global Manual Inference\n")
        f.write(f"Prompt: {prompt_text}\n\n")
        
    for prefix in prefixes:
        results = load_and_run(prefix, prompts, max_tokens=60)
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"## {prefix}\n")
            for res in results:
                f.write(f"- {res}\n")
            f.write("\n")
            
    print(f"\n📄 Report saved to {filename}")

def run_interactive_chat(prefixes):
    print("\n💬 Interactive Chat Mode")
    print("Select a model to talk to:")
    
    for i, p in enumerate(prefixes):
        print(f"{i+1}. {p}")
        
    try:
        choice = int(input("\nSelection: ")) - 1
        if choice < 0 or choice >= len(prefixes):
            print("Invalid selection.")
            return
    except:
        return
        
    prefix = prefixes[choice]
    print(f"\n⏳ Loading {prefix}...")
    
    # Persistent Load
    try:
        config_file = hf_hub_download(HF_REPO, filename=f"{prefix}/config.json" if prefix != "." else "config.json", local_dir=CACHE_DIR)
        with open(config_file, 'r') as cf:
            config = json.load(cf)
            
        model = DGESimpleTransformer(
            vocab_size=config.get("vocab_size", 50257),
            d_model=config.get("d_model", 384),
            n_layer=config.get("n_layer", 12),
            n_head=config.get("n_head", 6),
            max_seq_len=config.get("max_seq_len", 1024)
        )
        
        weights_file = hf_hub_download(HF_REPO, filename=f"{prefix}/weights.pt" if prefix != "." else "weights.pt", local_dir=CACHE_DIR)
        model.load_state_dict(torch.load(weights_file, map_location=DEVICE))
        model.to(DEVICE)
        print("✅ Model Loaded. Type 'exit' to quit.")
        
        while True:
            user_in = input("\nYou: ")
            if user_in.lower() in ['exit', 'quit']:
                break
                
            out = generate_text(model, "gpt2", user_in, max_new_tokens=100)
            print(f"AI: {out}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main_menu():
    prefixes = scan_repository()
    print(f"✅ Found {len(prefixes)} models.")
    
    if not prefixes:
        print("No models found in repo.")
        return

    while True:
        print("\n" + "="*50)
        print("🌍 GLOBAL INFERENCE MENU")
        print("="*50)
        print("1. 🤖 Run Automated Suite (Story/Math/German)")
        print("2. 📡 Broadcast Manual Prompt (Run on ALL models)")
        print("3. 💬 Interactive Chat (Select ONE model)")
        print("4. 🚪 Exit")
        
        choice = input("\nSelect: ").strip()
        
        if choice == '1':
            run_automated_suite(prefixes)
        elif choice == '2':
            run_broadcast_mode(prefixes)
        elif choice == '3':
            run_interactive_chat(prefixes)
        elif choice == '4':
            break
        else:
            print("Invalid Option")

if __name__ == "__main__":
    main_menu()
