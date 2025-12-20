
import os
import sys
import torch
import torch.nn.functional as F
from utils.model_manager import ModelManager, Diary
from core.model import DGESimpleTransformer
import time
import json
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()

# V 0.16.0: Unified Commander Dashboard
# The central mission control for DGE.

TITLE = """
==================================================
      DGE UNIFIED COMMANDER V 0.16.0
      Local & Cloud Mission Control
==================================================
"""

class DGEDashboard:
    def __init__(self):
        self.mgr = ModelManager()
        self.active_model = None
        self.active_config = None
        self.active_meta = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def _load_tokenizer(self):
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("⚠️ Transformers not installed. Using mock tokenizer.")
            self.tokenizer = None

    def main_menu(self):
        while True:
            self.clear_screen()
            print(TITLE)
            print("Main Dashboard:")
            print("1. [Local] List Models & Inference")
            print("2. [Cloud] RunPod Operations (Deploy/Monitor)")
            print("3. [Remote] Live Pod Chat (Emergence Monitoring)")
            print("4. [Sync]  Pull Results from HuggingFace")
            print("5. [Tour]  Direct Launch: DGE Grand Tour")
            print("q. Exit")
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '1':
                self.list_models_ui()
            elif choice == '2':
                self.cloud_ops_ui()
            elif choice == '3':
                self.remote_chat_ui()
            elif choice == '4':
                self.sync_results_ui()
            elif choice == '5':
                self.launch_grand_tour_ui()
            elif choice == 'q':
                sys.exit(0)

    # --- LOCAL OPERATIONS ---
    def list_models_ui(self):
        self.clear_screen()
        print(TITLE)
        print("--- LOCAL MODELS ---\n")
        
        models = self.mgr.list_models()
        if not models:
            print("No models found in models/ directory.")
            input("\nPress Enter to return...")
            return
            
        print(f"{'Idx':<5} | {'Family':<20} | {'Stage':<30} | {'Last Log'}")
        print("-" * 100)
        
        for idx, m in enumerate(models):
            last_log = m.get('last_log', '')[:40] + "..." if len(m.get('last_log', '')) > 40 else m.get('last_log', '')
            print(f"{idx+1:<5} | {m['family']:<20} | {m['stage']:<30} | {last_log}")
            
        print("-" * 100)
        choice = input("\nSelect Index to LOAD (or 'b' for back): ").strip()
        
        if choice.lower() == 'b':
            return
            
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
        
        try:
            state_dict, config = self.mgr.load_stage(meta['family'], meta['stage'], device=self.device)
            self.active_config = config
            self.active_meta = meta
            
            # Simple reconstructor (v0.11+ compatible)
            self.active_model = DGESimpleTransformer(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layer=config['n_layer'],
                n_head=config['n_head']
            )
            
            # V0.13+ Hierarchical check
            # In V0.14+ we might need add_skill_head calls to match state_dict if it was expanded
            # For now, load_state_dict(..., strict=False) to see if we can chat
            self.active_model.load_state_dict(state_dict, strict=False)
            self.active_model.to(self.device)
            self.active_model.eval()
            
            print("✅ Model loaded successfully!")
            
            # Show Diary
            diary_path = os.path.join(meta['path'], 'diary.md')
            if os.path.exists(diary_path):
                print("\n--- DIARY ---")
                with open(diary_path, 'r', encoding='utf-8') as f:
                    print("".join(f.readlines()[-10:]))
                
            input("\nPress Enter to start INFERENCE SHELL...")
            self.inference_shell()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            input("Press Enter...")

    # --- CLOUD OPERATIONS ---
    def cloud_ops_ui(self):
        import cloud.runpod_manager as runpod_manager
        while True:
            self.clear_screen()
            print(TITLE)
            print("--- RUNPOD MISSION CONTROL ---")
            
            try:
                import time as t
                start = t.time()
                print("⏳ Querying pods...", end='', flush=True)
                pods = runpod_manager.list_pods()
                elapsed = t.time() - start
                print(f" ({elapsed:.1f}s)")
                
                if not pods:
                    print("No active pods.")
                else:
                    for p in pods:
                        print(f"[{p['id']}] {p['gpuTypeId']} - {p['status']} (Uptime: {p['runtime']['uptimeInSeconds']}s)")
            except Exception as e:
                print(f"\n⚠️ RunPod API Error: {e}")

                
            print("\nOptions:")
            print("1. Refresh List")
            print("2. Deploy Single Experiment")
            print("3. Terminate Pod")
            print("b. Back")
            
            choice = input("\nSelect Option: ").strip().lower()
            if choice == 'b': break
            if choice == '1': continue
            if choice == '3':
                pid = input("Enter Pod ID to terminate: ").strip()
                if pid: runpod_manager.terminate_pod(pid); time.sleep(1)
            if choice == '2':
                self.deploy_ui()

    def remote_chat_ui(self):
        import cloud.runpod_manager as runpod_manager
        self.clear_screen()
        print(TITLE)
        print("--- REMOTE LIVE CHAT ---")
        
        try:
            pods = runpod_manager.list_pods()
            active_pods = [p for p in pods if p['status'] == 'RUNNING']
            if not active_pods:
                print("No running pods available for chat.")
                input("\nPress Enter to return...")
                return
        except Exception as e:
            print(f"❌ Error fetching pods: {e}")
            input("Press Enter...")
            return

        print(f"{'Idx':<5} | {'Pod ID':<15} | {'GPU':<25} | {'Address'}")
        print("-" * 70)
        for idx, p in enumerate(active_pods):
            addr = p.get('runtime', {}).get('address', 'Unknown')
            print(f"{idx+1:<5} | {p['id']:<15} | {p['gpuTypeId']:<25} | {addr}")
        
        choice = input("\nSelect Pod to Chat (or 'b' for back): ").strip()
        if choice.lower() == 'b': return

        try:
            p_idx = int(choice) - 1
            if 0 <= p_idx < len(active_pods):
                pod = active_pods[p_idx]
                pod_addr = pod.get('runtime', {}).get('address', '')
                if not pod_addr:
                    print("❌ Pod address not found. Ensure port 5000 is exposed.")
                    input("Press Enter...")
                    return
                
                # Assume RunPod mapping format: the address provided is the Base HTTP URL
                # We need to verify the port. Usually RunPod provides a direct address like [pod_id]-5000.proxy.runpod.net
                # or a direct IP if TCP. For HTTP/5000 it is often a proxy URL.
                
                # Let's use a robust inference loop
                self.run_remote_inference_loop(pod_addr)
            else:
                input("Invalid index. Press Enter...")
        except ValueError:
            pass

    def run_remote_inference_loop(self, base_url):
        # Normalize URL
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        
        # RunPod proxy URLs often look like: https://[pod_id]-5000.proxy.runpod.net
        # If list_pods returns just an IP, we append :5000
        if "http" not in base_url and "." not in base_url: # Likely just an IP
             base_url = f"http://{base_url}:5000"
        elif "proxy.runpod.net" not in base_url and ":" not in base_url[8:]:
             base_url = f"{base_url}:5000"

        self.clear_screen()
        print(f"--- REMOTE SHELL: {base_url} ---")
        print("Testing connection...")
        
        try:
            resp = requests.get(f"{base_url}/status", timeout=5)
            if resp.status_code == 200:
                print("✅ Connection Established. Model is ready on-pod.")
            else:
                print(f"⚠️ Pod responded with {resp.status_code}. Server might be starting.")
        except Exception as e:
            print(f"❌ Could not connect to pod: {e}")
            input("Press Enter...")
            return

        print("Type 'exit' to return.\n")
        while True:
            prompt = input("REMOTE USER > ").strip()
            if prompt.lower() == 'exit': break
            if not prompt: continue
            
            try:
                start_time = time.time()
                resp = requests.post(f"{base_url}/chat", json={"prompt": prompt, "max_tokens": 100}, timeout=30)
                duration = time.time() - start_time
                
                if resp.status_code == 200:
                    result = resp.json()
                    print(f"AI CLOUD     > {result.get('response', 'Empty response')}")
                    print(f"[Time: {duration*1000:.1f}ms]")
                else:
                    print(f"❌ Error from Pod: {resp.text}")
            except Exception as e:
                print(f"❌ Communication Error: {e}")
                
    def deploy_ui(self):
        import cloud.runpod_manager as runpod_manager
        self.clear_screen()
        print("🚀 DEPLOY TO CLOUD")
        print("="*50)
        print("Common Tasks:")
        print("1. Symbol Synergy (run_synergy_experiment)")
        print("2. DGE Grand Tour (4-stage comprehensive test)")
        print("3. Custom Command")
        print("q. Back")
        
        c = input("\nSelect Task: ").strip().lower()
        if c == '1': cmd = "python -m experiments.run_synergy_experiment"
        elif c == '2': cmd = "python -m experiments.run_dge_grand_tour"
        elif c == '3': cmd = input("Bash command: ").strip()
        elif c == 'q': return
        else: return
        
        # Interactive GPU selection (handles confirmation internally)
        result = runpod_manager.select_gpu_interactive()
        if not result or not result[0]:
            print("Deployment cancelled.")
            input("\nPress Enter...")
            return
            
        gpu_id, is_spot, cost = result
        
        # Deploy
        try:
            runpod_manager.deploy_experiment(
                cmd, 
                gpu_type=gpu_id, 
                is_spot=is_spot, 
                price=cost
            )
            input("\n✅ Pod initiated. Press Enter...")
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            input("\nPress Enter...")

    # --- SYNC OPERATIONS ---
    def sync_results_ui(self):
        self.clear_screen()
        print(TITLE)
        print("--- HUGGINGFACE SYNC ---\n")
        repo = input("HuggingFace Repo (default darealSven/dge): ").strip() or "darealSven/dge"
        print(f"Syncing from {repo}...")
        
        try:
            # Download to the current models folder
            dest = os.path.join(os.getcwd(), "models")
            subprocess.run(["huggingface-cli", "download", repo, "--local-dir", dest], check=True)
            print("\n✅ Sync Complete! New models available in Local List.")
        except Exception as e:
            print(f"❌ Sync Failed: {e}")
            
        input("\nPress Enter to return...")

    def launch_grand_tour_ui(self):
        self.clear_screen()
        print("🔥 DGE GRAND TOUR LAUNCHER")
        print("This will deploy the 4-stage systematic suite to RunPod.")
        confirm = input("Confirm Deployment? (y/n): ").strip().lower()
        if confirm == 'y':
            import cloud.runpod_manager as runpod_manager
            runpod_manager.deploy_experiment("python -m experiments.run_dge_grand_tour")
            print("\n✅ Grand Tour is running in the cloud. Check 'Cloud Ops' for status.")
        input("\nPress Enter to return...")

    def inference_shell(self):
        self._load_tokenizer()
        self.clear_screen()
        print(f"--- INFERENCE SHELL: {self.active_meta['stage']} ---")
        print("Type 'exit' to return.\n")
        
        while True:
            prompt = input("USER > ").strip()
            if prompt.lower() == 'exit': break
            if not prompt: continue
            
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            else:
                try:
                    tokens = [int(x) for x in prompt.split()]
                    input_ids = torch.tensor([tokens], device=self.device)
                except ValueError:
                    print("Error: Input space-separated integers.")
                    continue
            
            start_time = time.time()
            with torch.no_grad():
                generated = input_ids
                for _ in range(50):
                    logits, _ = self.active_model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    if self.tokenizer and next_token.item() == self.tokenizer.eos_token_id: break
            
            duration = time.time() - start_time
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True) if self.tokenizer else " ".join(map(str, generated[0].tolist()))
            print(f"AI   > {output_text}")
            print(f"[Time: {duration*1000:.1f}ms]")

if __name__ == "__main__":
    db = DGEDashboard()
    db.main_menu()
