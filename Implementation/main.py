
import os
import sys
import torch
import torch.nn.functional as F
from model_manager import ModelManager, Diary
from dge_model import DGESimpleTransformer
import time
import json

TITLE = """
==================================================
      DGE INFRASTRUCTURE V 0.11.0
      Scientific Inference Dashboard
==================================================
"""

class DGEDashboard:
    def __init__(self):
        self.mgr = ModelManager()
        self.active_model = None
        self.active_config = None
        self.active_meta = None
        self.tokenizer = None # TODO: HuggingFace tokenizer integration if needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def _load_tokenizer(self):
        # Fallback to simple char-level or basic GPT2 if available
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("⚠️ Transformers not installed. Using mock tokenizer.")
            self.tokenizer = None

    def list_models_ui(self):
        self.clear_screen()
        print(TITLE)
        print("Scanning for models...\n")
        
        models = self.mgr.list_models()
        if not models:
            print("No models found in models/ directory.")
            input("Press Enter to exit...")
            return
            
        print(f"{'Idx':<5} | {'Family':<20} | {'Stage':<30} | {'Last Log'}")
        print("-" * 100)
        
        for idx, m in enumerate(models):
            last_log = m.get('last_log', '')[:40] + "..." if len(m.get('last_log', '')) > 40 else m.get('last_log', '')
            print(f"{idx+1:<5} | {m['family']:<20} | {m['stage']:<30} | {last_log}")
            
        print("-" * 100)
        choice = input("\nSelect Model Index (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            sys.exit(0)
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                self.load_model_ui(models[idx])
            else:
                input("Invalid index. Press Enter...")
        except ValueError:
            pass

    def load_model_ui(self, meta):
        self.clear_screen()
        print(TITLE)
        print(f"Loading {meta['family']} / {meta['stage']}...")
        
        # Load weights
        try:
            state_dict, config = self.mgr.load_stage(meta['family'], meta['stage'], device=self.device)
            self.active_config = config
            self.active_meta = meta
            
            # Reconstruct model architecture
            if 'router_type' not in config:
                config['router_type'] = 'linear' # Backwards compatibility
                
            self.active_model = DGESimpleTransformer(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layer=config['n_layer'],
                n_head=config['n_head'],
                router_type=config['router_type'],
                initial_gating=config.get('initial_gating', False)
            )
            
            # Load state dict
            # Handle dge_masks? Usually they are buffers, so load_state_dict handles them.
            self.active_model.load_state_dict(state_dict)
            self.active_model.to(self.device)
            self.active_model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"Config: {json.dumps(config, indent=2)}")
            
            # Show Diary Context
            diary_path = os.path.join(meta['path'], 'diary.md')
            if os.path.exists(diary_path):
                print("\n--- DIARY CONTEXT (Last 10 lines) ---")
                with open(diary_path, 'r', encoding='utf-8') as f:
                    print("".join(f.readlines()[-10:]))
                print("-------------------------------------")
                
            input("\nPress Enter to start INFERENCE SHELL...")
            self.inference_shell()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            input("Press Enter to return...")

    def inference_shell(self):
        self._load_tokenizer()
        self.clear_screen()
        print(f"--- INFERENCE SHELL: {self.active_meta['stage']} ---")
        print("Type 'exit' to return.\n")
        
        while True:
            prompt = input("USER > ").strip()
            if prompt.lower() == 'exit':
                break
            if not prompt:
                continue
                
            # Tokenize
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            else:
                # Mock tokenization (char level if numeric)
                # Ensure input consists of digits/spaces matching our vocab size?
                # For now assume mostly numeric tasks
                try:
                    # Parse space separated integers
                    tokens = [int(x) for x in prompt.split()]
                    input_ids = torch.tensor([tokens], device=self.device)
                except ValueError:
                    print("Error: Without tokenizer, input space-separated integers.")
                    continue
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                # Simple generate loop
                generated = input_ids
                for _ in range(20): # Max new tokens
                    logits, _ = self.active_model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == self.tokenizer.eos_token_id if self.tokenizer else -1:
                        break
            
            duration = time.time() - start_time
            
            # Decode
            if self.tokenizer:
                output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                output_text = " ".join(map(str, generated[0].tolist()))
                
            print(f"AI   > {output_text}")
            print(f"[Time: {duration*1000:.1f}ms]")

if __name__ == "__main__":
    dashboard = DGEDashboard()
    while True:
        dashboard.list_models_ui()
