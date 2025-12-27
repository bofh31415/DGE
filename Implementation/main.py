
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

# V 0.17.0: Unified Commander Dashboard
# The central mission control for DGE.

TITLE = """
==================================================
      DGE UNIFIED COMMANDER V 0.17.0
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
            print("🔄 Loading tokenizer... (first run may take 1-2 min)", end='', flush=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(" ✅")
        except ImportError:
            print("\n⚠️ Transformers not installed. Using mock tokenizer.")
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
                pods = runpod_manager.get_pods_with_metrics()
                elapsed = t.time() - start
                print(f" ({elapsed:.1f}s)")
                
                if not pods:
                    print("No active pods.")
                else:
                    for p in pods:
                        runtime = p.get("runtime") or {}
                        gpus = runtime.get("gpus", [])
                        gpu_util = gpus[0].get("gpuUtilPercent", 0) if gpus else 0
                        mem_util = gpus[0].get("memoryUtilPercent", 0) if gpus else 0
                        status = p.get("desiredStatus", "UNKNOWN")
                        gpu_name = p.get("machine", {}).get("gpuDisplayName", "Unknown")
                        
                        # Status indicator
                        if gpu_util > 50:
                            indicator = "🔥 TRAINING"
                        elif gpu_util > 0:
                            indicator = "⚙️ Loading"
                        elif status == "RUNNING":
                            indicator = "⏳ Setup"
                        else:
                            indicator = status
                        
                        print(f"[{p['id']}] {gpu_name} | GPU: {gpu_util}% | {indicator}")
            except Exception as e:
                print(f"\n⚠️ RunPod API Error: {e}")

                
            print("\nOptions:")
            print("1. Refresh List")
            print("2. Deploy Single Experiment")
            print("3. Terminate Pod")
            print("4. Watch Progress (live updates)")
            print("5. Remote Inference (List/Run HF Models)")
            print("6. 🆕 Train TinyStories 75M (~$25-70)")
            print("7. 🧪 Train TinyStories 75M + LAWA (Experimental)")
            print("b. Back")
            
            choice = input("\nSelect Option: ").strip().lower()
            if choice == 'b': break
            if choice == '1': continue
            if choice == '3':
                pid = input("Enter Pod ID to terminate: ").strip()
                if pid: runpod_manager.terminate_pod(pid); time.sleep(1)
            if choice == '4':
                self.watch_pod_progress()
            if choice == '5':
                self.remote_inference_ui()
            if choice == '2':
                self.deploy_ui()
            if choice == '6':
                self.train_tinystories_75m_ui()
            if choice == '7':
                self.train_tinystories_lawa_ui()

    def watch_pod_progress(self):
        """Live monitoring of pod progress with GPU utilization."""
        import cloud.runpod_manager as runpod_manager
        
        print("\n🔭 WATCH MODE - Press Ctrl+C to stop\n")
        refresh_interval = 10  # seconds
        
        try:
            while True:
                try:
                    pods = runpod_manager.get_pods_with_metrics()
                    
                    # Clear line and show timestamp
                    print(f"\r[{time.strftime('%H:%M:%S')}] ", end='')
                    
                    if not pods:
                        print("No active pods.", end='')
                    else:
                        for p in pods:
                            runtime = p.get("runtime") or {}
                            gpus = runtime.get("gpus", [])
                            gpu_util = gpus[0].get("gpuUtilPercent", 0) if gpus else 0
                            mem_util = gpus[0].get("memoryUtilPercent", 0) if gpus else 0
                            status = p.get("desiredStatus", "UNKNOWN")
                            gpu_name = p.get("machine", {}).get("gpuDisplayName", "?")
                            
                            # Status indicator with progress bar
                            if gpu_util > 50:
                                bar = "█" * (gpu_util // 10) + "░" * (10 - gpu_util // 10)
                                indicator = f"🔥 TRAINING [{bar}] {gpu_util}%"
                            elif gpu_util > 0:
                                bar = "█" * (gpu_util // 10) + "░" * (10 - gpu_util // 10)
                                indicator = f"⚙️ Loading [{bar}] {gpu_util}%"
                            elif status == "RUNNING":
                                indicator = "⏳ Setup (pip install...)"
                            else:
                                indicator = status
                            
                            print(f"{gpu_name} | {indicator}", end='')
                    
                    print("", flush=True)
                    time.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\r⚠️ Error: {e}", end='')
                    time.sleep(refresh_interval)
                    
        except KeyboardInterrupt:
            print("\n\n✅ Watch mode stopped.")
        
        input("\nPress Enter...")

    def train_tinystories_75m_ui(self):
        """UI for training TinyStories 75M with GPU cost estimation."""
        import cloud.runpod_manager as runpod_manager
        
        self.clear_screen()
        print(TITLE)
        print("--- TRAIN TINYSTORIES 75M ---\n")
        print("🧠 75M params | ~100K steps | Target loss < 2.0")
        
        print("\n⏳ Checking GPU availability...")
    
        # Check for existing pods first
        active_pods = runpod_manager.list_pods()
        if active_pods:
            print(f"\n⚠️  FOUND {len(active_pods)} ACTIVE PODS:")
            for p in active_pods:
                print(f"   - {p['id']} ({p['status']}) - {p['gpuTypeId']}")
            
            print("\n   If an installation is broken, it is safer to TERMINATE and START FRESH.")
            action = input("   Terminate all active pods before deploying? (y/n) [y]: ").strip().lower()
            if action in ['', 'y']:
                print("   🧨 Terminating pods...")
                for p in active_pods:
                    runpod_manager.terminate_pod(p['id'])
                time.sleep(2) # Wait for cleanup
                
        gpus = runpod_manager.get_available_gpus()
        
        if not gpus:
            print("❌ Could not fetch GPU availability.")
            input("\nPress Enter...")
            return
        
        # Training parameters for cost estimation
        TOTAL_TOKENS = 3.2e9  # ~100K steps × 32 batch × 1024 seq
        BASE_TOKENS_PER_SEC = 15000  # RTX 4090 baseline
        BASE_TFLOPS = 82.6  # RTX 4090 TFLOPS
        
        # Filter available spot GPUs and calculate costs
        available = [g for g in gpus if g.get('communityPrice') and g['communityPrice'] > 0]
        
        for gpu in available:
            tflops = gpu['tflops']
            tokens_per_sec = BASE_TOKENS_PER_SEC * (tflops / BASE_TFLOPS)
            
            # Multi-GPU detection
            name = gpu['name']
            if name.startswith('2x '): tokens_per_sec *= 2
            elif name.startswith('3x '): tokens_per_sec *= 3
            elif name.startswith('4x '): tokens_per_sec *= 4
            elif name.startswith('8x '): tokens_per_sec *= 8
            
            training_hours = TOTAL_TOKENS / tokens_per_sec / 3600
            total_cost = training_hours * gpu['communityPrice']
            gpu['est_hours'] = training_hours
            gpu['est_cost'] = total_cost
        
        # Sort by total cost
        available.sort(key=lambda x: x.get('est_cost', 9999))
        
        # Display top 15
        print("\n" + "="*80)
        print("💰 CHEAPEST GPUs FOR TINYSTORIES 75M (sorted by total cost)")
        print("="*80)
        print(f"{'#':<3} | {'GPU':<22} | {'$/hr':<7} | {'VRAM':<6} | {'Time':<8} | {'Cost':<8}")
        print("-"*80)
        
        for i, gpu in enumerate(available[:15], 1):
            name = gpu['name'][:22]
            price = f"${gpu['communityPrice']:.2f}"
            vram = f"{int(gpu['vram'])} GB"
            hours = f"{gpu['est_hours']:.0f}h"
            cost = f"${gpu['est_cost']:.0f}"
            print(f"{i:<3} | {name:<22} | {price:<7} | {vram:<6} | {hours:<8} | {cost:<8}")
        
        print("-"*80)
        print("💡 Saves to HF every 1000 steps (~10 min). Safe for spot pods!")
        
        # Direct GPU selection from list
        choice = input("\n   Select GPU [1-15] or 'q' to cancel: ").strip().lower()
        if choice == 'q':
            input("\nPress Enter...")
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available[:15]):
                selected = available[idx]
                gpu_id = selected['id']
                price = selected['communityPrice']
                
                print(f"\n   ✅ Selected: {selected['name']} @ ${price:.2f}/hr")
                print(f"      Est. Time: {selected['est_hours']:.0f} hrs | Est. Cost: ${selected['est_cost']:.0f}")
                
                # GPU Count Selection (for multi-GPU DDP)
                gpu_count_input = input("\n   🔢 Number of GPUs (1-8) [1]: ").strip()
                gpu_count = int(gpu_count_input) if gpu_count_input.isdigit() and 1 <= int(gpu_count_input) <= 8 else 1
                
                if gpu_count > 1:
                    print(f"      🚀 Using {gpu_count} GPUs with DistributedDataParallel (DDP)")
                    command = f"torchrun --nproc_per_node={gpu_count} experiments/run_tinystories_75m.py"
                else:
                    command = "python experiments/run_tinystories_75m.py"
                
                # Docker Image Selection
                use_docker = input("\n   🐳 Use optimized DGE Docker image (faster startup)? (y/n) [n]: ").strip().lower()
                image_name = "darealsven/dge-env:latest" if use_docker == 'y' else None
                
                confirm = input("\n   Deploy training? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        runpod_manager.deploy_experiment(
                            command,
                            gpu_type=gpu_id,
                            gpu_count=gpu_count,
                            is_spot=True,
                            price=price,
                            image_name=image_name
                        )
                        print("\n✅ Training deployed! Use 'Watch Progress' to monitor.")
                    except Exception as e:
                        print(f"\n❌ Deployment failed: {e}")
        except ValueError:
            print("   Invalid selection")
        
        input("\nPress Enter...")

    def train_tinystories_lawa_ui(self):
        """UI for training TinyStories 75M with LAWA checkpoint averaging."""
        import cloud.runpod_manager as runpod_manager
        
        self.clear_screen()
        print(TITLE)
        print("--- TRAIN TINYSTORIES 75M + LAWA (Experimental) ---\n")
        print("🧠 75M params | LAWA: Latest Weight Averaging")
        print("📊 Saves both raw + averaged weights every checkpoint")
        print("☁️ Syncs to HuggingFace + Google Drive")
        
        print("\n⏳ Checking GPU availability...")
    
        # Check for existing pods first
        active_pods = runpod_manager.list_pods()
        if active_pods:
            print(f"\n⚠️  FOUND {len(active_pods)} ACTIVE PODS:")
            for p in active_pods:
                print(f"   - {p['id']} ({p['status']}) - {p['gpuTypeId']}")
            
            action = input("   Terminate all active pods before deploying? (y/n) [y]: ").strip().lower()
            if action in ['', 'y']:
                print("   🧨 Terminating pods...")
                for p in active_pods:
                    runpod_manager.terminate_pod(p['id'])
                time.sleep(2)
                
        gpus = runpod_manager.get_available_gpus()
        
        if not gpus:
            print("❌ Could not fetch GPU availability.")
            input("\nPress Enter...")
            return
        
        # Training parameters for cost estimation
        TOTAL_TOKENS = 3.2e9
        BASE_TOKENS_PER_SEC = 15000
        BASE_TFLOPS = 82.6
        
        # Filter available spot GPUs and calculate costs
        available = [g for g in gpus if g.get('communityPrice') and g['communityPrice'] > 0]
        
        for gpu in available:
            tflops = gpu['tflops']
            tokens_per_sec = BASE_TOKENS_PER_SEC * (tflops / BASE_TFLOPS)
            
            name = gpu['name']
            if name.startswith('2x '): tokens_per_sec *= 2
            elif name.startswith('3x '): tokens_per_sec *= 3
            elif name.startswith('4x '): tokens_per_sec *= 4
            elif name.startswith('8x '): tokens_per_sec *= 8
            
            training_hours = TOTAL_TOKENS / tokens_per_sec / 3600
            total_cost = training_hours * gpu['communityPrice']
            gpu['est_hours'] = training_hours
            gpu['est_cost'] = total_cost
        
        available.sort(key=lambda x: x.get('est_cost', 9999))
        
        # Display top 15
        print("\n" + "="*80)
        print("💰 CHEAPEST GPUs FOR TINYSTORIES 75M + LAWA (sorted by total cost)")
        print("="*80)
        print(f"{'#':<3} | {'GPU':<22} | {'$/hr':<7} | {'VRAM':<6} | {'Time':<8} | {'Cost':<8}")
        print("-"*80)
        
        for i, gpu in enumerate(available[:15], 1):
            name = gpu['name'][:22]
            price = f"${gpu['communityPrice']:.2f}"
            vram = f"{int(gpu['vram'])} GB"
            hours = f"{gpu['est_hours']:.0f}h"
            cost = f"${gpu['est_cost']:.0f}"
            print(f"{i:<3} | {name:<22} | {price:<7} | {vram:<6} | {hours:<8} | {cost:<8}")
        
        print("-"*80)
        print("📊 LAWA: Averages last 5 checkpoints for better generalization")
        print("☁️ Syncs to HF + Google Drive every 1000 steps")
        
        choice = input("\n   Select GPU [1-15] or 'q' to cancel: ").strip().lower()
        if choice == 'q':
            input("\nPress Enter...")
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available[:15]):
                selected = available[idx]
                gpu_id = selected['id']
                price = selected['communityPrice']
                
                print(f"\n   ✅ Selected: {selected['name']} @ ${price:.2f}/hr")
                print(f"      Est. Time: {selected['est_hours']:.0f} hrs | Est. Cost: ${selected['est_cost']:.0f}")
                
                # GPU Count Selection
                gpu_count_input = input("\n   🔢 Number of GPUs (1-8) [1]: ").strip()
                gpu_count = int(gpu_count_input) if gpu_count_input.isdigit() and 1 <= int(gpu_count_input) <= 8 else 1
                
                if gpu_count > 1:
                    print(f"      🚀 Using {gpu_count} GPUs with DistributedDataParallel (DDP)")
                    command = f"torchrun --nproc_per_node={gpu_count} experiments/run_tinystories_75m_lawa.py"
                else:
                    command = "python experiments/run_tinystories_75m_lawa.py"
                
                # Docker Image Selection
                use_docker = input("\n   🐳 Use optimized DGE Docker image (faster startup)? (y/n) [n]: ").strip().lower()
                image_name = "darealsven/dge-env:latest" if use_docker == 'y' else None
                
                confirm = input("\n   Deploy LAWA training? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        runpod_manager.deploy_experiment(
                            command,
                            gpu_type=gpu_id,
                            gpu_count=gpu_count,
                            is_spot=True,
                            price=price,
                            image_name=image_name
                        )
                        print("\n✅ LAWA Training deployed! Use 'Watch Progress' to monitor.")
                    except Exception as e:
                        print(f"\n❌ Deployment failed: {e}")
        except ValueError:
            print("   Invalid selection")
        
        input("\nPress Enter...")

    def remote_inference_ui(self):
        """UI for listing HF models and deploying inference server on RunPod."""
        import cloud.runpod_manager as runpod_manager
        from huggingface_hub import HfApi
        import os
        import requests
        import time
        
        self.clear_screen()
        print(TITLE)
        print("--- REMOTE INFERENCE: HuggingFace Models ---\n")
        
        hf_repo = "darealSven/dge"
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"🔍 Scanning {hf_repo}...")
        try:
            api = HfApi(token=hf_token)
            files = api.list_repo_files(hf_repo, token=hf_token)
            
            # Find model directories (contain config.json + weights.pt)
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
            
            if not models:
                print("❌ No valid models found in HF repo.")
                input("\nPress Enter...")
                return
            
            print(f"\n✅ Found {len(models)} models:\n")
            for i, m in enumerate(models):
                display_name = m if m != "." else "(root)"
                print(f"  {i+1}. {display_name}")
            
            print("\nOptions:")
            print("1. 🚀 Deploy Interactive Server (Chat in Terminal)")
            print("2. 📊 Run Automated Suite on ALL models (RunPod)")
            print("b. Back")
            
            choice = input("\nSelect: ").strip().lower()
            
            if choice == 'b':
                return
            elif choice == '1':
                # Deploy server and start interactive chat
                while True:  # Retry loop
                    # Select GPU
                    print("\n" + "="*60)
                    gpu_id, is_spot, cost = runpod_manager.select_gpu_interactive()
                    
                    if not gpu_id:
                        # User cancelled
                        break
                    
                    try:
                        print("\n📦 Deploying inference server...")
                        pod_id = runpod_manager.deploy_experiment(
                            "python experiments/run_global_inference.py --server",
                            gpu_type=gpu_id,
                            is_spot=is_spot,
                            price=cost
                        )
                        
                        # Wait for pod to be running and get address
                        print("\n⏳ Waiting for pod deployment...")
                        time.sleep(15)  # Initial wait for pod startup
                        
                        server_url = None
                        for attempt in range(60):  # Try for ~2 minutes
                            try:
                                pods = runpod_manager.get_pods_with_metrics()
                                our_pod = next((p for p in pods if p['id'] == pod_id), None)
                                
                                if our_pod:
                                    runtime = our_pod.get('runtime')
                                    if runtime:
                                        ports = runtime.get('ports', [])
                                        
                                        # Debug: show what we got
                                        if attempt == 0 and ports:
                                            print(f"   DEBUG: Found {len(ports)} port(s)")
                                        
                                        # Look for any HTTP port mapping
                                        for port_info in ports:
                                            ip = port_info.get('ip')
                                            public_port = port_info.get('publicPort')
                                            private_port = port_info.get('privatePort')
                                            
                                            # Match port 5000 (our Flask server)
                                            if private_port == 5000 and ip and public_port:
                                                server_url = f"http://{ip}:{public_port}"
                                                break
                                        
                                        if server_url:
                                            break
                                
                                if attempt % 5 == 0 and attempt > 0:
                                    print(f"   ... waiting for port {5000} ({attempt*2}s elapsed)")
                                    
                            except Exception as e:
                                if attempt == 0:
                                    print(f"   DEBUG: Error querying pod: {e}")
                            
                            time.sleep(2)
                        
                        if not server_url:
                            print("\n❌ Could not get server address. Check RunPod console.")
                            retry = input("\nTry another GPU? (y/n): ").strip().lower()
                            if retry != 'y':
                                break
                            continue
                        
                        # Test connection - pip install takes 2-3 minutes!
                        print(f"\n🔗 Server: {server_url}")
                        print("   ⏳ Waiting for pip install to complete (~3 min)...")
                        
                        connected = False
                        for attempt in range(60):  # Try for 3 minutes (60 * 3s = 180s)
                            try:
                                resp = requests.get(f"{server_url}/health", timeout=5)
                                if resp.status_code == 200:
                                    print(f"   ✅ Connected after {attempt*3}s!")
                                    connected = True
                                    break
                            except:
                                if attempt % 10 == 0 and attempt > 0:
                                    print(f"   ... still waiting ({attempt*3}s elapsed)")
                                time.sleep(3)
                        
                        if not connected:
                            print("\n⚠️ Server not responding after 3 minutes.")
                            print("   The server might still be starting. Check RunPod logs.")
                            print(f"   Try later: curl {server_url}/health")
                            retry = input("\nTry another GPU? (y/n): ").strip().lower()
                            if retry != 'y':
                                break
                            continue
                        
                        # Interactive chat loop
                        self.remote_chat_with_server(server_url, models)
                        break  # Exit retry loop after successful chat
                        
                    except Exception as e:
                        if "does not have the resources" in str(e):
                            print(f"\n❌ GPU out of capacity: {str(e)}")
                            retry = input("\nTry another GPU? (y/n): ").strip().lower()
                            if retry != 'y':
                                break
                        else:
                            print(f"\n❌ Deployment failed: {e}")
                            retry = input("\nTry again? (y/n): ").strip().lower()
                            if retry != 'y':
                                break
                        
            elif choice == '2':
                confirm = input("\n🚀 Run automated suite on ALL models? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        runpod_manager.deploy_experiment("python experiments/run_global_inference.py --auto")
                        print("\n✅ Deployed! Results will be saved to global_inference_report.md")
                    except Exception as e:
                        print(f"\n❌ Deployment failed: {e}")
                    input("\nPress Enter...")
                        
        except Exception as e:
            print(f"❌ Error accessing HuggingFace: {e}")
            input("\nPress Enter...")

    def remote_chat_with_server(self, server_url, models):
        """Interactive chat with remote inference server."""
        import requests
        
        print("\n" + "="*60)
        print("💬 REMOTE CHAT MODE")
        print("="*60)
        
        # Select model
        print("\nSelect a model to load:")
        for i, m in enumerate(models):
            display_name = m if m != "." else "(root)"
            print(f"  {i+1}. {display_name}")
        
        try:
            choice = int(input("\nModel #: ").strip())
            if choice < 1 or choice > len(models):
                print("❌ Invalid selection.")
                input("\nPress Enter...")
                return
            selected_model = models[choice - 1]
        except:
            print("❌ Invalid input.")
            input("\nPress Enter...")
            return
        
        # Load model
        print(f"\n🔄 Loading {selected_model} on remote server...")
        try:
            resp = requests.post(
                f"{server_url}/load_model",
                json={"prefix": selected_model},
                timeout=120  # Model loading can take time
            )
            if resp.status_code == 200:
                print(f"✅ Model loaded: {selected_model}")
            else:
                print(f"❌ Load failed: {resp.json().get('message', 'Unknown error')}")
                input("\nPress Enter...")
                return
        except Exception as e:
            print(f"❌ Request failed: {e}")
            input("\nPress Enter...")
            return
        
        # Chat loop
        print("\n💡 Type your prompts below. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate
                print("   ⏳ Generating...", end='', flush=True)
                resp = requests.post(
                    f"{server_url}/generate",
                    json={"prompt": user_input, "max_tokens": 100},
                    timeout=60
                )
                print("\r", end='')  # Clear "Generating..." line
                
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get('text', '')
                    print(f"AI:  {text}\n")
                else:
                    print(f"❌ Error: {resp.json().get('message', 'Unknown')}\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Request failed: {e}\n")
        
        input("\nPress Enter to return...")


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
            from huggingface_hub import snapshot_download
            dest = os.path.join(os.getcwd(), "models")
            
            print(f"   Downloading to: {dest}")
            snapshot_download(repo_id=repo, local_dir=dest, local_dir_use_symlinks=False)
            
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
            try:
                runpod_manager.deploy_experiment("python experiments/run_dge_grand_tour.py")
                print("\n✅ Grand Tour is running in the cloud. Check 'Cloud Ops' for status.")
            except Exception as e:
                error_msg = str(e)
                if "does not have the resources" in error_msg:
                    print("\n❌ RTX 4090 is currently OUT OF CAPACITY on RunPod.")
                    print("   Use Cloud Ops → Deploy to select a different GPU.")
                elif "timeout" in error_msg.lower():
                    print(f"\n⏱️ Deployment timed out: {e}")
                else:
                    print(f"\n❌ Deployment failed: {e}")
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

# Backward-compatible alias for tests
DGELab = DGEDashboard

if __name__ == "__main__":
    db = DGEDashboard()
    db.main_menu()
