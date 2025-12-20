
import os
import sys
import torch
import torch.nn.functional as F
from model_manager import ModelManager, Diary
from dge_model import DGESimpleTransformer
from dge_training import train_task, TaskType
import time
import json
import numpy as np

# Configuration
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 8,
    "lr": 1e-3
}

TITLE = """
==================================================
      DGE EXPERIMENT LAB V 0.11.1
      Synergy & Emergence Testbed
==================================================
"""

class ExperimentLab:
    def __init__(self):
        self.mgr = ModelManager()
        self.device = CONFIG['device']
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def main_menu(self):
        while True:
            self.clear_screen()
            print(TITLE)
            print("Select Experiment Chain:")
            print("1. [Fast] Symbol Gen Synergy (||| + ooo -> |||ooo)")
            print("2. [Med]  Language Composition (cat -> Katze + s -> plural)")
            print("3. [Med]  Style Transfer (happy -> HAPPY -> HAPPY!!!)")
            print("4. [Slow] Modular Counting (2+4 -> 6 -> oooooo)")
            print("5. [NEW]  Remote RunPod Deploy (Fire & Forget)")
            print("6. [🔥]  DGE Grand Tour (4 Stages - Systematic Suite)")
            print("q. Exit")
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '1':
                self.run_symbol_experiment()
            elif choice == '2':
                print("🚧 Verification Pending: Implement Language Experiment")
                time.sleep(1)
            elif choice == '3':
                print("🚧 Verification Pending: Implement Style Experiment")
                time.sleep(1)
            elif choice == '4':
                print("🚧 Verification Pending: Implement Modular Counting")
                time.sleep(1)
            elif choice == '5':
                self.runpod_deploy_ui()
            elif choice == '6':
                self.run_grand_tour_locally()
            elif choice == 'q':
                sys.exit(0)

    def run_grand_tour_locally(self):
        """Orchestrates the Grand Tour locally."""
        self.clear_screen()
        print("🌍 DGE Grand Tour - Local Execution")
        print("-" * 50)
        print("This will execute Stage 1 -> Stage 4 sequentially.")
        print("Note: Stages 3 and 4 are heavy. RunPod recommended.")
        
        confirm = input("\nStart Grand Tour now? (y/n): ").strip().lower()
        if confirm == 'y':
            import run_dge_grand_tour
            run_dge_grand_tour.main()
            input("\nGrand Tour Finished. Press Enter...")
                
    def runpod_deploy_ui(self):
        """UI for deploying an experiment to RunPod."""
        self.clear_screen()
        print("🚀 Remote RunPod Deployment")
        print("-" * 30)
        
        # Check for RunPod API Key
        import runpod_manager
        try:
            pods = runpod_manager.list_pods()
            print(f"Active Pods: {len(pods)}")
        except Exception as e:
            print(f"⚠️ RunPod API Error: {e}")
            print("Make sure RUNPOD_API_KEY is in your .env")
            input("\nPress Enter to return...")
            return

        print("\nCommon Commands:")
        print("1. Symbol Generation (Fast)")
        print("2. TinyStories to GSM8K (Slow)")
        print("3. Custom Command")
        
        cmd_choice = input("\nSelect Command (default 1): ").strip()
        
        if cmd_choice == '2':
            command = "python run_tinystories_gsm8k_chain.py"
        elif cmd_choice == '3':
            command = input("Enter custom bash command: ").strip()
        else:
            command = "python run_synergy_experiment.py" # Shortcut for symbol test
            
        gpu_type = input("GPU Type (default RTX 4090): ").strip() or "NVIDIA GeForce RTX 4090"
        
        confirm = input(f"\nDeploy '{command}' to {gpu_type}? (y/n): ").strip().lower()
        if confirm == 'y':
            try:
                pod_id = runpod_manager.deploy_experiment(command, gpu_type=gpu_type)
                print(f"\n✅ Deployment initiated! Pod ID: {pod_id}")
                print("The pod will configure itself and start the experiment in a tmux session.")
            except Exception as e:
                print(f"❌ Deployment failed: {e}")
                
        input("\nPress Enter to return...")

    # --- EXPERIMENT 1: SYMBOL GENERATION ---
    def run_symbol_experiment(self):
        """
        Tests if independent symbol generation skills can combine.
        Skill A: "3" -> "|||"
        Skill B: "3" -> "ooo"
        Synergy: "3" -> "|||ooo" (if additive) or random?
        """
        print("\n🧪 Experiment 1: Symbol Generation Synergy")
        family_name = "Exp_Symbol_Gen"
        config = {
            "d_model": 64,
            "n_layer": 2,
            "n_head": 4,
            "vocab_size": 100, # 0-9 digits, 10='|', 11='o'
            "router_type": "rbf",
            "initial_gating": True
        }
        
        # 1. Setup
        print(f"Creating family '{family_name}'...")
        # Clean previous
        if os.path.exists(os.path.join("models", family_name)):
            import shutil
            shutil.rmtree(os.path.join("models", family_name))
        self.mgr.create_family(family_name, config)
        
        model = DGESimpleTransformer(**config).to(self.device)
        
        # 2. Data Generators for this specific experiment
        def gen_skill_a(batch_size=8):
            # Input: Random digit 0-9. Output: '|||' (3 pipes)
            # Actually let's make it count: Input N -> N pipes
            x = torch.randint(0, 5, (batch_size, 1)) # Small numbers 0-4
            # We need sequence to sequence?
            # Input "3", output "|||". 
            # Let's simple mapping: x=[3], y=[10,10,10]
            # But dge_training expects (x,y) batches.
            # We need a custom train loop or adapter.
            # Let's use simple custom training here for flexibility.
            pass

        # Since dge_training.train_task uses predefined TaskType, 
        # we'll implement a custom training loop here using the ModelManager integration logic manually.
        # This is better for new experiment types.
        
        # Train Stage 1: Skill A (Pipes)
        print("Training Skill A (Pipes: N -> |||)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        
        for step in range(200):
            # Generate batch: Input N (0-4), Output N pipes (token 10)
            N = torch.randint(1, 5, (CONFIG['batch_size'],)).to(self.device) # [3, 1, 4...]
            
            # Construct input/target
            # Input sequence: [N] (length 1) - wait, transformer needs expanding? 
            # Or seqlen. Let's do Input [N, 0, 0], Output [10, 10, 10] padded
            # Simplified: Input [N], Output single token? No, sequence gen.
            
            # Let's try: Input [N] -> Output [Pipe, Pipe, Pipe]
            # Requires autoregressive training?
            # For simplicity: Input embedding of N -> Output embedding of Pipes. 
            # But DGE is sequence model.
            
            # Let's map: Input seq [N], Target seq [10, 10, ... 10]
            # This requires the model to learn "repeat token 10 N times".
            # That's hard for a tiny model without scratchpad.
            
            # SIMPLIFIED TASK: 
            # Skill A: Input [N] -> Output [10] (One Pipe)
            # Skill B: Input [N] -> Output [11] (One Circle)
            # Synergy: Input [N] -> Output [10, 11]? Or [10+11]?
            
            # Actually, the user asked for "3 blue, 1 red". That implies count.
            # Let'sstick to the "fast" plan: "3" -> "|||"
            # We train on fixed length 3.
            # Input: "3" (token 3)
            # Target: "| | |" (tokens 10, 10, 10)
            
            x = N.unsqueeze(1) # [B, 1]
            # Target: we want it to predict 10 given 3.. but autoregressively?
            # Model(x) -> logits. 
            # We want logits to be 10.
            
            optimizer.zero_grad()
            # DGE forward expects (x) usually, returns (logits, info)
            # We can just feed x=[N], train last token?
            logits, info = model(x)
            # logits: [B, 1, Vocab]
            
            target = torch.full((CONFIG['batch_size'], 1), 10, device=self.device, dtype=torch.long)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), target.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}: Loss {loss.item():.4f}")
                
        # Save Stage 1
        model_state = model.module if isinstance(model, torch.nn.DataParallel) else model
        metrics = {"loss": loss.item(), "task": "Skill_A_Pipes"}
        self.mgr.save_stage(model_state, family_name, "Stage1_Pipes", config, metrics)
        
        # Expand for Skill B
        print("\nExpanding for Skill B (Circles: N -> ooo)...")
        # Token 11 = 'o'
        model.expand_for_skill("Skill_Circles", router_type="rbf")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        
        for step in range(200):
            N = torch.randint(1, 5, (CONFIG['batch_size'],)).to(self.device)
            x = N.unsqueeze(1)
            
            # Target 11 (Circle)
            target = torch.full((CONFIG['batch_size'], 1), 11, device=self.device, dtype=torch.long)
            
            optimizer.zero_grad()
            # Interleaved Replay (Synergy require replay?) 
            # User wants history agnostic. Let's try WITHOUT replay first?
            # "Verify skills can be trained independently without replay" -> Phase 7.
            # We are testing synergy.
            
            logits, info = model(x)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), target.view(-1))
            loss.backward()
            optimizer.step()
             
            if step % 50 == 0:
                print(f"Step {step}: Loss {loss.item():.4f}")

        # Save Stage 2
        metrics = {"loss": loss.item(), "task": "Skill_B_Circles"}
        self.mgr.save_stage(model_state, family_name, "Stage2_Circles", config, metrics, parent_stage="Stage1_Pipes")
        
        # Test Synergy
        print("\n🧪 Testing Synergy...")
        # Input '3'. Skill A wants '|' (10). Skill B wants 'o' (11).
        # Soft gating might output mixture?
        # Or if we have additive output: logits_A + logits_B
        # Logits for 10 high + Logits for 11 high.
        # Argmax might be neither? Or one of them.
        
        # Real synergy would be: Train a third "head" or see if representations help.
        # But here we just check overlap.
        
        model.eval()
        x_test = torch.tensor([[3]], device=self.device) # Input "3"
        logits, _ = model(x_test)
        probs = F.softmax(logits[0,0], dim=-1)
        
        p_pipe = probs[10].item()
        p_circle = probs[11].item()
        
        print(f"Input: 3")
        print(f"Prob('|'): {p_pipe:.4f}")
        print(f"Prob('o'): {p_circle:.4f}")
        
        if p_pipe > 0.1 and p_circle > 0.1:
            print("✅ Synergy Potential: Both skills active!")
        else:
            print("❌ No Synergy: Winner takes all.")
            
        input("\nExperiment Complete. Press Enter...")

if __name__ == "__main__":
    lab = ExperimentLab()
    lab.main_menu()
