import torch
import torch.optim as optim
import sys
import os
from dge_model import DGESimpleTransformer
from dge_utils import Quadrant, MoEGatedLinear
from dge_training import train_task, evaluate_task, TaskType
from version import __version__
from dge_logger import DGELogger
import time
import json
import shutil
from dge_utils import DGEAdamW
import argparse

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("==================================================")
    print("          DGE CONCEPT LAB: EXPERIMENTATION        ")
    print("==================================================")

class DGELab:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.model_name = None
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.logger = None
        self.trained_skills = set()
        self.global_step = 0
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _get_model_state(self):
        """Extract comprehensive model state for logging."""
        if self.model is None:
            return {}
            
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Count frozen vs active in DGE layers
        frozen_params = 0
        active_params = 0
        
        for module in self.model.modules():
            if isinstance(module, MoEGatedLinear):
                mask = module.backward_mask
                frozen_params += (mask == 0).sum().item()
                active_params += (mask == 1).sum().item()
        
        return {
            'd_model': self.model.d_model,
            'n_layer': len(self.model.layers),
            'n_head': self.model.layers[0].n_head if self.model.layers else 0,
            'vocab_size': self.model.token_emb.num_embeddings,
            'total_params': total_params,
            'frozen_params': frozen_params,
            'active_params': active_params
        }
        
    def create_model(self):
        print("\n--- Create New Model ---")
        name = input("Enter Model Name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return

        d_model = int(input("Enter d_model (default 64): ") or 64)
        n_layer = int(input("Enter n_layer (default 2): ") or 2)
        n_head = int(input("Enter n_head (default 4): ") or 4)
        vocab_size = int(input("Enter vocab_size (default 1000): ") or 1000)
        
        self.model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
        # CRITICAL FIX: weight_decay=0.0 is required to prevent "decaying" frozen weights (which have 0 grad).
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        self.model_name = name
        self.trained_skills = set() # Reset skills for new model
        self.global_step = 0 # Reset step
        
        # Init Logger
        self.logger = DGELogger(os.path.join(self.models_dir, name))
        
        # Log creation with full model state
        model_state = self._get_model_state()
        self.logger.log_event("CREATED", model_state, step=0)
        
        print(f"Model '{name}' created! d_model={d_model}, layers={n_layer}, vocab={vocab_size}")
        
        # Save initial snapshot
        print("Saving initial snapshot...")
        self.save_model()
        print("Initial snapshot saved.")

    def reset_model(self):
        """
        Resets to a fresh default model for experiments.
        Used by automated experiment chains.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dge_exp_{timestamp}"
        
        # Default experiment configuration
        d_model = 64
        n_layer = 2
        n_head = 4
        vocab_size = 500
        
        self.model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
        self.optimizer = DGEAdamW(self.model.parameters(), lr=1e-3)
        self.model_name = model_name
        self.trained_skills = set()
        self.global_step = 0
        
        # Init Logger
        self.logger = DGELogger(os.path.join(self.models_dir, model_name))
        model_state = self._get_model_state()
        self.logger.log_event("CREATED", model_state, step=0)
        
        print(f"Model '{model_name}' created for experiment.")
        
    def train_menu(self):
        if self.model is None:
            print("❌ No model! Create one first.")
            return

        while True:
            print("\n" + "="*50)
            print("🏋️  TRAINING MENU")
            print("="*50)
            print("1. 📈 Count Up (Next Number)")
            print("2. 📉 Count Down (Previous Number)")
            print("9. 🏛️ Create Foundation Models (Eng/Ger/Math)")
            print("10. 🌍 Run Global Inference (All HF Models)")
            print("21. 🏋️ Neuro-Bodybuilding (Sparsity Tuning)")
            print("23. 🗿 Rosetta Stone Synergy (German Math)")
            print("e. 🧠  Expand Model")
            print("f. 📊  Full Skill Benchmark")
            print("b. 🔙 Back")
            
            choice = input("\nSelect Training Task: ").strip().lower()
            
            if choice == 'b':
                return
            
            vocab = self.model.token_emb.num_embeddings
            
            #Define checkpoint callback
            def checkpoint_fn(step):
                self.save_model(current_step=step)
                self.logger.log_event("CHECKPOINT", {"step": step}, step=step)

            if choice == '1':
                # Identify Replay Tasks (Everything learned EXCEPT current task)
                replay_list = []
                for skill_name in self.trained_skills:
                    if skill_name != TaskType.COUNT_UP.name:
                        replay_list.append(TaskType[skill_name])
                
                print(f"Replay Tasks: {[t.name for t in replay_list]}")

                steps = int(input("Steps (default 500): ") or 500)
                new_step = train_task(self.model, TaskType.COUNT_UP, vocab_size=vocab, steps=steps, 
                                     logger=self.logger, start_step=self.global_step, checkpoint_fn=checkpoint_fn,
                                     optimizer=self.optimizer,
                                     replay_tasks=replay_list)
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_UP.name)
            elif choice == '2':
                # Identify Replay Tasks (Everything learned EXCEPT current task)
                replay_list = []
                for skill_name in self.trained_skills:
                    if skill_name != TaskType.COUNT_DOWN.name:
                        replay_list.append(TaskType[skill_name])
                
                print(f"Replay Tasks: {[t.name for t in replay_list]}")

                steps = int(input("Steps (default 500): ") or 500)
                new_step = train_task(self.model, TaskType.COUNT_DOWN, vocab_size=vocab, steps=steps, 
                                     logger=self.logger, start_step=self.global_step, checkpoint_fn=checkpoint_fn,
                                     optimizer=self.optimizer,
                                     replay_tasks=replay_list)
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_DOWN.name)
            elif choice == '9':
                # Create Foundation Models
                print("\n🏛️ Launching Foundation Models Generator...")
                import subprocess
                subprocess.run(["python", "run_create_foundations.py"])
            elif choice == '10':
                # Global Inference
                print("\n🌍 Launching Global Inference Scan...")
                import subprocess
                subprocess.run(["python", "run_global_inference.py"])
            elif choice == '21':
                # Neuro-Bodybuilding Experiment
                print("\n🏋️ Launching Neuro-Bodybuilding Experiment...")
                import subprocess
                subprocess.run(["python", "run_neuro_bodybuilding.py"])
            elif choice == '23':
                # Rosetta Stone Experiment
                print("\n🗿 Launching Rosetta Stone Experiment...")
                import subprocess
                subprocess.run(["python", "run_rosetta_stone_experiment.py"])
            elif choice == 'e':
                self.expand_menu()
            elif choice == 'f':
                self.benchmark_skills()
            else:
                print("❌ Invalid option")
                continue
                
            self.save_model()

    def expand_menu(self):
        if self.model is None:
            print("❌ No model! Create one first.")
            return
            
        print("\n" + "="*50)
        print("🧠  EXPANSION MENU")
        print("="*50)
        
        try:
            inp = input("Enter added d_model (e.g. 64): ").strip()
            if not inp:
                added = 64
                print("Defaulting to 64.")
            else:
                added = int(inp)
            
            print("\nSelect Quadrant for FROZEN core:")
            print("1. ↖️ Top Left (Standard)")
            print("2. ↗️ Top Right")
            print("3. ↙️ Bottom Left")
            print("4. ↘️ Bottom Right")
            q_choice = input("Choice (default 1): ").strip()
            
            quadrant = Quadrant.TOP_LEFT
            if q_choice == '2': quadrant = Quadrant.TOP_RIGHT
            elif q_choice == '3': quadrant = Quadrant.BOTTOM_LEFT
            elif q_choice == '4': quadrant = Quadrant.BOTTOM_RIGHT
            
            # H4 Update: Calculate absolute new dimension and enforce protections
            new_d_model = self.model.d_model + added
            # V26 (DGE 0.4.0) BEST PRACTICES:
            # - Bigram Router: Contextual gating solves aliasing
            # - Open Init (0.0): Ensures plasticity
            # - No Isolation: Allow Synergy
            # - No Ortho Init: Prevent attenuation
            self.model.expand_model(new_input_dim=new_d_model, 
                                  new_output_dim=self.model.token_emb.num_embeddings,
                                  router_type='bigram', # V26
                                  isolate_cross_terms=False, # V26
                                  use_gradient_rescue=True, 
                                  use_orthogonal_init=False, # V26
                                  router_init_bias=0.0, # V26
                                  gating_threshold=0.0 # V26
                                  )
            
            # Log expansion with updated model state
            if self.logger:
                model_state = self._get_model_state()
                model_state['added_width'] = added
                model_state['quadrant'] = str(quadrant)
                self.logger.log_event("EXPANDED", model_state, step=self.global_step)
            
            # Re-init optimizer for new parameters
            # CRITICAL FIX: weight_decay=0.0 is required to prevent "decaying" frozen weights.
            router_params = []
            default_params = []
            for name, param in self.model.named_parameters():
                if 'router' in name or 'gate' in name:
                    router_params.append(param)
                else:
                    default_params.append(param)
                    
            self.optimizer = DGEAdamW([
                {'params': default_params, 'lr': 1e-3},
                {'params': router_params, 'lr': 1e-4} 
            ], weight_decay=0.0)
            print("✅ Model expanded successfully.")
            
        except ValueError:
            print("❌ Invalid input.")

    def inference_menu(self):
        if self.model is None:
            print("❌ No model!")
            return
            
        print("\n" + "="*50)
        print("🎲  INFERENCE MENU")
        print("="*50)
        print("Enter comma-separated integers (e.g. 1,2,3)")
        print("Type 'b' to go back.")
        
        while True:
            text = input("\nInput > ").strip()
            if text.lower() == 'b':
                break
                
            try:
                tokens = [int(x.strip()) for x in text.split(',')]
                x = torch.tensor([tokens], dtype=torch.long)
                
                self.model.eval()
                with torch.no_grad():
                    logits, _ = self.model(x)
                    # logits: [1, T, vocab]
                    last_logits = logits[0, -1, :]
                    pred = last_logits.argmax().item()
                    
                print(f"🔮 Prediction: {pred}")
                
                # Simple check for counting
                if len(tokens) > 0 and pred == tokens[-1] + 1:
                    print("✅ (Correctly counted +1)")
                else:
                    print("⚠️ (Did not count +1)")
                    
                self.model.train()
                
            except ValueError:
                print("❌ Invalid input. Please enter integers.")

    def save_model(self, create_snapshot=True, current_step=None):
        if self.model is None:
            print("No model to save!")
            return
            
        save_dir = os.path.join(self.models_dir, self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine step to log
        step_to_log = current_step if current_step is not None else self.global_step
        
        # Prepare config
        config = {
            'd_model': self.model.d_model,
            'n_layer': len(self.model.layers),
            'n_head': self.model.layers[0].n_head, 
            'model_name': self.model_name,
            'vocab_size': self.model.token_emb.num_embeddings,
            'trained_skills': list(self.trained_skills),
            'global_step': step_to_log # Use the accurate step
        }
        
        # Save main/latest files
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'weights.pt'))
        
        # Create versioned snapshot
        if create_snapshot:
            snapshots_dir = os.path.join(save_dir, 'snapshots')
            os.makedirs(snapshots_dir, exist_ok=True)
            
            snapshot_name = f"step_{step_to_log:06d}"
            snapshot_path = os.path.join(snapshots_dir, snapshot_name)
            os.makedirs(snapshot_path, exist_ok=True)
            
            # Save snapshot files
            with open(os.path.join(snapshot_path, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            torch.save(self.model.state_dict(), os.path.join(snapshot_path, 'weights.pt'))
            
            print(f"Snapshot saved: {snapshot_name}")
        
        if self.logger:
            self.logger.log_event("SAVED", {"path": save_dir, "snapshot": create_snapshot}, step=self.global_step)
            
        print(f"Model saved to {save_dir}")

    def load_menu(self):
        print("\n" + "="*50)
        print("📂  LOAD MODEL")
        print("="*50)
        
        if not os.path.exists(self.models_dir):
            print("❌ No models directory found.")
            return

        models = [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
        if not models:
            print("❌ No models found.")
            return
            
        print("Available Models:")
        valid_models = []
        for i, m in enumerate(models):
            # Check if config.json exists to ensure it's a valid model directory
            if os.path.exists(os.path.join(self.models_dir, m, 'config.json')):
                print(f"{len(valid_models)+1}. {m}")
                valid_models.append(m)
            
        if not valid_models:
            print("❌ No valid models found (missing config.json).")
            return

        while True:
            choice = input("\nSelect model to load (number) or 'b' to go back: ").strip().lower()
            if choice == 'b':
                return
            
            try:
                choice = int(choice)
                if 1 <= choice <= len(valid_models):
                    break
                else:
                    print("❌ Invalid number.")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'b'.")

        name = valid_models[choice-1]
        self.load_model_by_name(name)

    def load_model_by_name(self, name):
        """
        Loads the model state from disk.
        Args:
            name (str): Model name (directory name)
        """
        load_dir = os.path.join(self.models_dir, name)
        
        # Load Config
        try:
            with open(os.path.join(load_dir, 'config.json'), 'r') as f:
                config = json.load(f)
                
            print(f"Loading model '{name}' with config: {config}")
            
            # Reconstruct Model
            vocab_size = config.get('vocab_size', 1000)
            target_d_model = config['d_model']
            
            # V20 Fix: Handle Legacy Models w/ varying context lengths
            # Check 'max_seq_len' first, then 'context_length', then default 128
            max_seq_len = config.get('max_seq_len', config.get('context_length', 128))
            
            # 1. Attempt Base Construction
            self.model = DGESimpleTransformer(
                vocab_size=vocab_size,
                d_model=target_d_model,
                n_layer=config['n_layer'],
                n_head=config['n_head'],
                max_seq_len=max_seq_len
            )
            
            # Load Weights with Auto-Recovery for Expanded Models
            try:
                self.model.load_state_dict(torch.load(os.path.join(load_dir, 'weights.pt')))
            except RuntimeError as e:
                err_str = str(e)
                # Check for Expansion Mismatch (e.g. 64 -> 96)
                if "size mismatch" in err_str and "ln1.norms" in err_str:
                    print("⚠️  Detected Expanded Model Architecture mismatch. Attempting Reconstructive Surgery...")
                    
                    # Heuristic for Seed Fund Experiment (64 -> 96)
                    # If target is 96, we assume base was 64 (d_model=64, n_head=4 -> head_dim=16)
                    # Wait, Seed Fund used d=64, n_head=4 -> head_dim=16. 
                    # 96 / 16 = 6 heads.
                    
                    base_d_model = 64
                    base_n_head = 4
                    
                    # If this is TinyStories (384 -> 1024), we handle that too
                    if target_d_model == 1024:
                        base_d_model = 384
                        base_n_head = 6
                    
                    print(f"   Re-initializing as Base Model (d={base_d_model})...")
                    self.model = DGESimpleTransformer(
                        vocab_size=vocab_size,
                        d_model=base_d_model,
                        n_layer=config['n_layer'],
                        n_head=base_n_head,
                        max_seq_len=max_seq_len
                    )
                    
                    # Apply Expansion
                    print(f"   Re-applying Expansion ({base_d_model} -> {target_d_model})...")
                    # We assume standard expansion parameters for now
                    self.model.expand_model(
                        new_input_dim=target_d_model,
                        new_output_dim=0, # Assuming no vocab expansion
                        router_type='mlp', # MATCHED: Checkpoint uses MLP router (router.0.weight)
                        use_gradient_rescue=True,
                        isolate_cross_terms=False 
                    )
                    
                    # Retry Load
                    print("   Retrying State Load...")
                    self.model.load_state_dict(torch.load(os.path.join(load_dir, 'weights.pt')))
                    print("✅  Reconstruction Successful.")
                else:
                    raise e

            self.model_name = name
            self.trained_skills = set(config.get('trained_skills', []))
            self.global_step = config.get('global_step', 0)
            
            self.logger = DGELogger(load_dir)
            
            # Log load with current model state
            model_state = self._get_model_state()
            self.logger.log_event("LOADED", model_state, step=self.global_step)
            
            # Re-init optimizer
            # CRITICAL FIX: weight_decay=0.0 is used here too for consistency
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0)
            print("[OK] Model loaded successfully.")
            
        except Exception as e:
            print(f"[Error] loading model: {e}")
            import traceback
            traceback.print_exc()

    def inspect_model(self):
        if self.model is None:
            print("No model!")
            return
            
        print("\n" + "="*50)
        print("           MODEL INSPECTION")
        print("="*50)
        
        # Get comprehensive state
        state = self._get_model_state()
        
        print("\n[Architecture]")
        print(f"  d_model:    {state['d_model']}")
        print(f"  n_layer:    {state['n_layer']}")
        print(f"  n_head:     {state['n_head']}")
        print(f"  vocab_size: {state['vocab_size']}")
        
        print("\n[Parameters]")
        print(f"  Total:      {state['total_params']:,}")
        print(f"  Frozen:     {state['frozen_params']:,}")
        print(f"  Active:     {state['active_params']:,}")
        
        if state['total_params'] > 0:
            frozen_pct = (state['frozen_params'] / state['total_params']) * 100
            print(f"  Frozen:     {frozen_pct:.1f}%")
        
        # Gate statistics (from first layer for reference)
        print("\n[Gate Statistics - Layer 0]")
        layer0 = self.model.layers[0]
        gate_row = layer0.w_o.gate_row
        gate_col = layer0.w_o.gate_col
        
        print(f"  w_o shape:      {layer0.w_o.weight.shape}")
        print(f"  Gate Mean (Row): {gate_row.mean().item():.3f}")
        print(f"  Gate Mean (Col): {gate_col.mean().item():.3f}")
        
        print("\n[Training State]")
        print(f"  Global Step:     {self.global_step}")
        print(f"  Learned Skills:  {', '.join(self.trained_skills) if self.trained_skills else 'None'}")
        print("="*50 + "\n")

    def benchmark_skills(self):
        if self.model is None:
            print("No model!")
            return
            
        print("\n--- Full Skill Benchmark ---")
        vocab = self.model.token_emb.num_embeddings
        
        mode = input("Benchmark (A)ll tasks or (L)earned tasks only? [default: A]: ").strip().upper()
        
        target_tasks = []
        if mode == 'L':
            # Filter TaskTypes that are in trained_skills
            for task in TaskType:
                if task.name in self.trained_skills:
                    target_tasks.append(task)
            if not target_tasks:
                print("No skills learned yet.")
                return
        else:
            # All Tasks
            target_tasks = list(TaskType)
        
        results = {}
        for task in target_tasks:
            acc = evaluate_task(self.model, task, vocab_size=vocab)
            status = "(Learned)" if task.name in self.trained_skills else "(Baseline)"
            print(f"Task: {task.name:<15} {status:<12} Accuracy: {acc:.2f}%")
            results[task.name] = acc
        
        if self.logger:
            self.logger.log_event("BENCHMARK", results, step=self.global_step)

    def _run_experiment_by_choice(self, choice: str):
        """
        Run an experiment by its menu choice string.
        Used by CLI mode (-x flag) for non-interactive execution.
        """
        choice = choice.strip().lower()
        
        # Remap simplified choices to working experiments
        choice_map = {'1': '18', '2': '19', '3': '20'}
        choice = choice_map.get(choice, choice)
        
        if choice == '18':
            import run_seed_fund_experiment
            print("\n🟢 Running Directed Synergy (V26 - Working Config)...")
            run_seed_fund_experiment.run_seed_fund_experiment()
        elif choice == '19':
            import run_replay_ratio_experiment
            print("\n📉 Running Variable Replay Ratio Sensitivity Experiment...")
            run_replay_ratio_experiment.run_experiment()
        elif choice == '20':
            self._run_tinystories_training()
        elif choice == '17':
            self.reset_model()
            vocab_size = 500
            
            # Train Skill A
            print("Training Skill A (COUNT_UP)...")
            new_step = train_task(
                self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
            )
            self.global_step = new_step
            self.trained_skills.add(TaskType.COUNT_UP.name)
            
            # Expand
            print("Expanding model...")
            current_d_model = self.model.d_model
            self.model.expand_model(new_input_dim=current_d_model + 64, 
                              new_output_dim=self.model.token_emb.num_embeddings, 
                              router_type='mlp',
                              isolate_cross_terms=True,
                              use_gradient_rescue=True,
                              use_orthogonal_init=True)
            
            # Split parameters for Differential LR
            router_params = []
            default_params = []
            for name, param in self.model.named_parameters():
                if 'router' in name or 'gate' in name:
                    router_params.append(param)
                else:
                    default_params.append(param)
                    
            self.optimizer = DGEAdamW([
                {'params': default_params, 'lr': 1e-3},
                {'params': router_params, 'lr': 1e-4}
            ], weight_decay=0.0)
            
            # Train Skill B with probing
            print("Training Skill B (COUNT_DOWN) with Skill A probing...")
            new_step = train_task(
                self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                probe_task=TaskType.COUNT_UP, sparsity_lambda=0.05
            )
            self.global_step = new_step
            self.trained_skills.add(TaskType.COUNT_DOWN.name)
            self.save_model("dge_single_exp_custom")
        elif choice == '17':
            import run_v21_rbf
            print("\n🧪 Running V21 RBF Verification...")
            run_v21_rbf.run_experiment()
        elif choice == '18':
            import run_seed_fund_experiment
            print("\n🟢 Running Directed Synergy Verification (V26 Success)...")
            run_seed_fund_experiment.run_seed_fund_experiment()
        elif choice == '19':
            import run_replay_ratio_experiment
            print("\n📉 Running Variable Replay Ratio Sensitivity Experiment...")
            run_replay_ratio_experiment.run_experiment()
        elif choice == '20':
            self._run_tinystories_training()
        elif choice == '21':
            import run_tinystories_gsm8k_chain
            print("\n🔬 Running TinyStories → GSM8K Chain (Paper Experiment)...")
            run_tinystories_gsm8k_chain.run_experiment()
        else:
            print(f"❌ Unknown experiment choice: '{choice}'")
            print("Available experiments: 1, 2, 3, 4 (or legacy: 17, 18, 19, 20, 21)")
            print("  1/18: Directed Synergy (V26 - Working Config)")
            print("  2/19: Variable Replay Ratio Sensitivity")
            print("  3/20: TinyStories Training")
            print("  4/21: TinyStories → GSM8K Chain (Paper Experiment)")


    def experiment_menu(self):
        while True:
            print("\n" + "="*50)
            print("🧪  EXPERIMENTS MENU")
            print("="*50)
            print("1. 🟢 Directed Synergy (V26 - Working Config)")
            print("2. 📉 Variable Replay Ratio Sensitivity (0.01% - 50%)")
            print("3. 📖 Train on TinyStories (HuggingFace)")
            print("4. 🔬 TinyStories → GSM8K Chain (Paper Experiment)")
            print("5. 💪 Neuro-Bodybuilding (Sparsity Tuning)")
            print("6. 🔤 Dynamic Vocabulary Expansion")
            print("b. 🔙 Back")
            print("q. 🚪 Exit")
            
            choice = input("\nSelect Option: ").strip().lower()
            # Remap simplified menu choices to working experiments
            choice_map = {'1': '18', '2': '19', '3': '20', '4': '21', '5': '22', '6': '23'}
            choice = choice_map.get(choice, choice)
            
            if choice == '18':
                import run_seed_fund_experiment
                print("\n🟢 Running Directed Synergy (V26 - Working Config)...")
                run_seed_fund_experiment.run_seed_fund_experiment()
            elif choice == '19':
                import run_replay_ratio_experiment
                print("\n📉 Running Variable Replay Ratio Sensitivity Experiment...")
                run_replay_ratio_experiment.run_experiment()
            elif choice == '20':
                self._run_tinystories_training()
            elif choice == '21':
                import run_tinystories_gsm8k_chain
                print("\n🔬 Running TinyStories → GSM8K Chain (Paper Experiment)...")
                run_tinystories_gsm8k_chain.run_experiment()
            elif choice == '22':
                import run_neuro_bodybuilding
                print("\n💪 Running Neuro-Bodybuilding (Sparsity Tuning)...")
                run_neuro_bodybuilding.run_experiment()
            elif choice == '23':
                import run_vocab_expansion_experiment
                print("\n🔤 Running Dynamic Vocabulary Expansion...")
                run_vocab_expansion_experiment.run_experiment()
            elif choice == 'b':
                break
            elif choice == 'q':
                print("👋 Goodbye!")
                sys.exit(0)
            elif choice == '2':
                print("\\n⚙️ STARTING EXPERIMENT CHAIN V3 (Gradient Rescue & Orthogonality)...")
                vocab_size = 500
                
                # --- EXP 1: H2 Gradient Rescue ---
                print("\\n🔬 EXP 1: H2 Gradient Rescue (Transparent Gate)")
                self.reset_model()
                
                # Train Skill A
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                
                # Expand with Rescue
                print("  Expanding model with Gradient Rescue...")
                current_d_model = self.model.d_model
                self.model.expand_model(
                    new_input_dim=current_d_model + 32, # Conservative width
                    new_output_dim=self.model.token_emb.num_embeddings, 
                    router_type='mlp', # Using MLP as base since it had the starvation issue
                    use_gradient_rescue=True 
                )
                
                # Standard LR split
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                self.optimizer = DGEAdamW([{'params': default_params, 'lr': 1e-3}, {'params': router_params, 'lr': 1e-4}], weight_decay=0.0)
                
                # Train Skill B
                print("  Training Skill B (COUNT_DOWN)...")
                train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
                )
                self.save_model("dge_exp_v3_h2_rescue")
                
                # --- EXP 2: H3 Orthogonality ---
                print("\\n🔬 EXP 2: H3 Orthogonal Initialization")
                self.reset_model()
                
                # Train Skill A
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                
                # Expand with Ortho
                print("  Expanding model with Orthogonal Init...")
                current_d_model = self.model.d_model
                self.model.expand_model(
                    new_input_dim=current_d_model + 32, 
                    new_output_dim=self.model.token_emb.num_embeddings, 
                    router_type='mlp',
                    use_orthogonal_init=True,
                    use_gradient_rescue=True # Combining both for best shot? Or isolated?
                    # Let's isolate first? No, user wants solution. Optimization implies combining best practices.
                    # Actually, H3 (Ortho) might mitigate need for Rescue if it aligns better?
                    # But Rescue fixes the fundamental deadlock mechanism. 
                    # Let's run H3 WITH Rescue to be safe, or just H3?
                    # User asked for Orthogonality to avoid tinkering.
                    # Let's run BOTH for H3.
                )
                
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                self.optimizer = DGEAdamW([{'params': default_params, 'lr': 1e-3}, {'params': router_params, 'lr': 1e-4}], weight_decay=0.0)
                
                print("  Training Skill B (COUNT_DOWN)...")
                train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
                )
                self.save_model("dge_exp_v3_h3_ortho")
                
                # --- EXP 3: H4 Block Diagonal Isolation ---
                print("\\n🔬 EXP 3: H4 Block Diagonal Isolation")
                self.reset_model()
                
                # Train Skill A
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                
                # Expand with ISO
                print("  Expanding model with Block Diagonal Isolation...")
                current_d_model = self.model.d_model
                self.model.expand_model(
                    new_input_dim=current_d_model + 32, 
                    new_output_dim=self.model.token_emb.num_embeddings, 
                    router_type='mlp',
                    use_orthogonal_init=True,
                    use_gradient_rescue=True,
                    isolate_cross_terms=True # <--- H4: The key flag
                )
                
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                self.optimizer = DGEAdamW([{'params': default_params, 'lr': 1e-3}, {'params': router_params, 'lr': 1e-4}], weight_decay=0.0)
                
                print("  Training Skill B (COUNT_DOWN)...")
                train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
                )
                self.save_model("dge_exp_v3_h4_iso")
                
            elif choice == '3':
                # Experiment Chain V2 (Legacy Option 10)
                print("\\n🧪 STARTING EXPERIMENT CHAIN V2 (Noise Init Hypotheses)...")
                vocab_size = 500
                
                # Experiment 1: MLP Router (Capacity)
                print("\\n🔬 EXP 1: V 0.3.0 (MLP Router) - Testing H1 (Linear Inseparability)")
                self.reset_model()
                
                # Initial Training Skill A
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_UP.name)
                
                # Expand with MLP
                print("  Expanding model with MLP router...")
                current_d_model = self.model.d_model
                self.model.expand_model(new_input_dim=current_d_model + 16, new_output_dim=self.model.token_emb.num_embeddings, router_type='mlp')
                
                # Differential LR
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                
                self.optimizer = DGEAdamW([
                    {'params': default_params, 'lr': 1e-3},
                    {'params': router_params, 'lr': 1e-4}
                ], weight_decay=0.0)
                
                # Train Skill B
                print("  Training Skill B (COUNT_DOWN) with Skill A probing...")
                new_step = train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_DOWN.name)
                self.save_model("dge_val_0_3_0_mlp")

                # Experiment 2: Strong Sparsity (Constraint)
                print("\\n🔬 EXP 2: V 0.3.1 (Lambda 0.2) - Testing H2 (Weak Constraint)")
                self.reset_model()
                
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_UP.name)
                
                print("  Expanding model...")
                current_d_model = self.model.d_model
                self.model.expand_model(new_input_dim=current_d_model + 16, new_output_dim=self.model.token_emb.num_embeddings, router_type='linear')
                
                # Differential LR
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                
                self.optimizer = DGEAdamW([
                    {'params': default_params, 'lr': 1e-3},
                    {'params': router_params, 'lr': 1e-4}
                ], weight_decay=0.0)
                
                print("  Training Skill B (COUNT_DOWN) with STRONG sparsity...")
                new_step = train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.20
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_DOWN.name)
                self.save_model("dge_val_0_3_1_lambda_0_2")

                # Experiment 3: Low Router LR (Sensitivity)
                print("\\n🔬 EXP 3: V 0.3.2 (Router LR 5e-5) - Testing H3 (Overshoot)")
                self.reset_model()
                
                print("  Training Skill A (COUNT_UP)...")
                new_step = train_task(
                    self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_UP.name)
                
                print("  Expanding model...")
                current_d_model = self.model.d_model
                self.model.expand_model(new_input_dim=current_d_model + 16, new_output_dim=self.model.token_emb.num_embeddings, router_type='linear')
                
                # Differential LR (LOW router LR)
                router_params = [p for n, p in self.model.named_parameters() if 'router' in n or 'gate' in n]
                default_params = [p for n, p in self.model.named_parameters() if 'router' not in n and 'gate' not in n]
                
                self.optimizer = DGEAdamW([
                    {'params': default_params, 'lr': 1e-3},
                    {'params': router_params, 'lr': 5e-5}  # Halved router LR
                ], weight_decay=0.0)
                
                print("  Training Skill B (COUNT_DOWN) with LOW router LR...")
                new_step = train_task(
                    self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=500,
                    logger=self.logger, start_step=self.global_step, optimizer=self.optimizer,
                    probe_task_type=TaskType.COUNT_UP, sparsity_lambda=0.05
                )
                self.global_step = new_step
                self.trained_skills.add(TaskType.COUNT_DOWN.name)
                self.save_model("dge_val_0_3_2_low_lr")
                
                print("\\n✅ EXPERIMENT CHAIN V2 COMPLETE!")
            elif choice == '17':
                import run_v21_rbf
                print("\n🧪 Running V21 RBF Verification...")
                run_v21_rbf.run_experiment()
            elif choice == '18':
                import run_seed_fund_experiment
                print("\n🟢 Running Directed Synergy Verification (V26 Success)...")
                run_seed_fund_experiment.run_seed_fund_experiment()
            elif choice == '19':
                import run_replay_ratio_experiment
                print("\n📉 Running Variable Replay Ratio Sensitivity Experiment...")
                run_replay_ratio_experiment.run_experiment()
            elif choice == '20':
                # TinyStories Training
                self._run_tinystories_training()
            elif choice == 'b':
                break
            elif choice == 'q':
                print("👋 Exiting... Goodbye!")
                exit()
            else:
                print("❌ Invalid option.")

    def run_dge_validation_chain(self):
        """
        Automated validation pipeline for DGE Architecture.
        Cycle:
        1. Create fresh model
        2. Train Task A (Count Up)
        3. Verify A
        4. Expand Model (Identity Check)
        5. Verify A (Must be preserved)
        6. Train Task B (Count Down)
        7. Verify A + B
        """
        import time
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dge_val_{timestamp}"
        
        print("\n" + "*"*60)
        print(f" STARTING DGE VALIDATION CHAIN: {model_name} ")
        print("*"*60)
        
        # --- Step 1: Genesis ---
        print("\n[Phase 1] Genesis: Creating Model...")
        self.model_name = model_name
        d_model = 64
        n_layer = 2
        n_head = 4
        vocab_size = 500  # Smaller vocab for faster convergence on simple tasks
        
        self.model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
        self.model = DGESimpleTransformer(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
        
        # --- Differential Learning Rates (V 0.2.3) ---
        # Router needs to open SLOWLY to prevent lazy drift.
        router_params = []
        default_params = []
        
        for name, param in self.model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            else:
                default_params.append(param)
                
        self.optimizer = DGEAdamW([
            {'params': default_params, 'lr': 1e-3},
            {'params': router_params, 'lr': 1e-4} # 0.1x Learning Rate for Stability
        ], weight_decay=0.0) # Weight decay handled internally by DGEAdamW if needed/configured
        self.trained_skills = set()
        self.global_step = 0
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Init Logger & Snapshot
        self.logger = DGELogger(os.path.join(self.models_dir, model_name))
        model_state = self._get_model_state()
        self.logger.log_event("CREATED", model_state, step=0)
        self.save_model()
        print("Model created and initial snapshot saved.")
        
        # --- Step 2: Learn Skill A ---
        print("\n[Phase 2] Training Skill A: COUNT_UP (500 steps)...")
        # Callback for periodic saves
        def checkpoint_fn(step):
            self.save_model(current_step=step)
            
        steps_a = 500
        new_step = train_task(
            self.model, TaskType.COUNT_UP, vocab_size=vocab_size, steps=steps_a, 
            logger=self.logger, start_step=self.global_step, checkpoint_fn=checkpoint_fn,
            optimizer=self.optimizer # Use the DGEAdamW instance with groups
        )
        self.global_step = new_step
        self.trained_skills.add(TaskType.COUNT_UP.name)
        
        # --- Step 3: Verify A ---
        print("\n[Phase 3] Verifying Skill A...")
        acc_a = evaluate_task(self.model, TaskType.COUNT_UP, vocab_size=vocab_size)
        print(f"Skill A Accuracy: {acc_a:.2f}%")
        if acc_a < 90.0:
            print("WARNING: Skill A not learned sufficiently. Aborting chain?")
            # We continue for science, but warn
        
        # --- Step 4: Expansion ---
        print("\n[Phase 4] Expanding Model (+64 width)...")
        current_d_model = self.model.d_model
        self.model.expand_model(new_input_dim=current_d_model + 32, 
                              new_output_dim=self.model.token_emb.num_embeddings, 
                              router_type='mlp',
                              isolate_cross_terms=True,
                              use_gradient_rescue=True,
                              use_orthogonal_init=True)
        
        # Log Expansion
        if self.logger:
            model_state = self._get_model_state()
            model_state['added_width'] = 32 # Updated to reflect the new expansion
            model_state['quadrant'] = "TOP_LEFT" # This might be inferred or removed if not applicable to new signature
            self.logger.log_event("EXPANDED", model_state, step=self.global_step)
            
        # Re-init optimizer
        # Re-init optimizer (Parameters changed) - MUST PRESERVE DIFFERENTIAL LRs
        router_params = []
        default_params = []
        for name, param in self.model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            else:
                default_params.append(param)
                
        self.optimizer = DGEAdamW([
            {'params': default_params, 'lr': 1e-3},
            {'params': router_params, 'lr': 1e-4} 
        ], weight_decay=0.0)
        
        # --- Step 5: Identity Check ---
        print("\n[Phase 5] Identity Verification (Checking Skill A post-expansion)...")
        acc_a_post = evaluate_task(self.model, TaskType.COUNT_UP, vocab_size=vocab_size)
        print(f"Skill A Accuracy (Post-Expand): {acc_a_post:.2f}%")
        
        diff = acc_a - acc_a_post
        if abs(diff) < 1.0:
            print("SUCCESS: Identity Preserved! (Delta < 1%)")
        else:
            print(f"FAILURE: Significant drift detected! Delta: {diff:.2f}%")
            
        # --- Step 6: Learn Skill B ---
        print("\n[Phase 6] Training Skill B: COUNT_DOWN (500 steps)...")
        steps_b = 500
        new_step = train_task(
            self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size, steps=steps_b, 
            logger=self.logger, start_step=self.global_step, checkpoint_fn=checkpoint_fn,
            optimizer=self.optimizer, # Use Differential LRs
            probe_task_type=TaskType.COUNT_UP, # Forensic Logging: Spy on Skill A
            sparsity_lambda=0.05
        )
        self.global_step = new_step
        self.trained_skills.add(TaskType.COUNT_DOWN.name)
        
        # --- Step 7: Final Verification ---
        print("\n[Phase 7] Final Verification (A + B)...")
        final_acc_a = evaluate_task(self.model, TaskType.COUNT_UP, vocab_size=vocab_size)
        final_acc_b = evaluate_task(self.model, TaskType.COUNT_DOWN, vocab_size=vocab_size)
        
        print("-" * 40)
        print(f"FINAL RESULTS for {model_name}")
        print("-" * 40)
        print(f"Skill A (Count Up):   {final_acc_a:.2f}%  (Init: {acc_a:.2f}%)")
        print(f"Skill B (Count Down): {final_acc_b:.2f}%")
        print("-" * 40)
        
        # Log Final Benchmark
        self.logger.log_event("BENCHMARK_CHAIN_FINAL", {
            "COUNT_UP": final_acc_a,
            "COUNT_DOWN": final_acc_b,
            "ID_PRESERVATION_DELTA": diff
        }, step=self.global_step)
        
        self.save_model()
        print("Validation Chain Complete.")

    def _run_tinystories_training(self):
        """
        Train a fresh DGE model on TinyStories dataset from HuggingFace.
        """
        from data import load_tinystories
        from dge_training import train_dataset
        from replay_buffer import ReplayBuffer
        
        print("\n" + "="*60)
        print("📖 TINYSTORIES TRAINING (HuggingFace)")
        print("="*60)
        
        # Get user inputs
        model_name = input("Model Name (default 'tinystories_dge'): ").strip() or "tinystories_dge"
        max_samples = int(input("Max Samples (default 10000, 0=all): ").strip() or 10000)
        epochs = int(input("Epochs (default 1): ").strip() or 1)
        batch_size = int(input("Batch Size (default 32): ").strip() or 32)
        seq_len = int(input("Sequence Length (default 128): ").strip() or 128)
        
        # Model config - use GPT-2's vocab size for compatibility
        d_model = int(input("d_model (default 256): ").strip() or 256)
        n_layer = int(input("n_layer (default 4): ").strip() or 4)
        n_head = int(input("n_head (default 8): ").strip() or 8)
        vocab_size = 50257  # GPT-2 vocab size
        
        print(f"\n🔧 Creating model: {model_name}")
        print(f"   Config: d={d_model}, layers={n_layer}, heads={n_head}, vocab={vocab_size}")
        
        # Create fresh model
        self.model = DGESimpleTransformer(
            vocab_size=vocab_size, 
            d_model=d_model, 
            n_layer=n_layer, 
            n_head=n_head
        )
        self.model_name = model_name
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.trained_skills = set()
        self.global_step = 0
        
        # Init Logger
        self.logger = DGELogger(os.path.join(self.models_dir, model_name))
        self.logger.log_event("CREATED", self._get_model_state(), step=0)
        
        # Save initial checkpoint
        self.save_model()
        
        # Load TinyStories
        print(f"\n📚 Loading TinyStories dataset...")
        dataloader = load_tinystories(
            split='train',
            max_samples=max_samples if max_samples > 0 else None,
            seq_len=seq_len,
            batch_size=batch_size,
            tokenizer_name='gpt2',
            vocab_size=vocab_size
        )
        
        # Create replay buffer for future expansion
        replay_buffer = ReplayBuffer(max_size=5000, task_name="tinystories")
        
        # Define checkpoint callback
        def checkpoint_fn(step):
            self.save_model(current_step=step)
        
        # Train!
        print(f"\n🏋️ Starting training: {epochs} epoch(s)...")
        new_step = train_dataset(
            model=self.model,
            dataloader=dataloader,
            epochs=epochs,
            optimizer=self.optimizer,
            logger=self.logger,
            start_step=self.global_step,
            checkpoint_fn=checkpoint_fn,
            checkpoint_interval=500,
            replay_buffer=replay_buffer,
            replay_ratio=0.0,  # No replay on first task
            task_name="tinystories",
            auto_populate_buffer=True
        )
        
        self.global_step = new_step
        self.trained_skills.add("TINYSTORIES")
        
        # Save replay buffer for future use
        buffer_path = os.path.join(self.models_dir, model_name, 'replay_buffer')
        replay_buffer.save(buffer_path)
        
        # Final save
        self.save_model()
        
        print("\n" + "="*60)
        print(f"✅ TinyStories training complete!")
        print(f"   Model: {model_name}")
        print(f"   Steps: {self.global_step}")
        print(f"   Replay Buffer: {len(replay_buffer)} samples saved")
        print("="*60)

    def _dataset_menu(self):
        """
        Dataset management menu for downloading and importing datasets.
        """
        from data import (list_local_datasets, download_hf_dataset, 
                         import_text_file, DATA_STORE_PATH)
        
        while True:
            print("\n" + "="*50)
            print("🗂️  DATASET MANAGER")
            print("="*50)
            print(f"📁 Data Store: {DATA_STORE_PATH}")
            
            # List existing datasets
            datasets = list_local_datasets()
            if datasets:
                print(f"\n📂 Local Datasets ({len(datasets)}):")
                for ds in datasets:
                    print(f"   • {ds['name']}: {ds['samples']} samples ({ds['type']})")
            else:
                print("\n📂 No local datasets yet.")
            
            print("\n--- Actions ---")
            print("1. ⬇️  Download from HuggingFace")
            print("2. 📄 Import Text File")
            print("3. 🔄 Refresh List")
            print("b. 🔙 Back")
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '1':
                print("\n--- Download from HuggingFace ---")
                print("Examples: roneneldan/TinyStories, wikitext, openai/gsm8k")
                dataset_name = input("Dataset Name: ").strip()
                if not dataset_name:
                    print("❌ No dataset name provided.")
                    continue
                
                # First check for configs
                config_name = None
                from data import get_dataset_configs, get_dataset_splits
                configs = get_dataset_configs(dataset_name)
                if len(configs) > 1:
                    print(f"\n📋 This dataset has {len(configs)} configs:")
                    for i, cfg in enumerate(configs):
                        print(f"   {i+1}. {cfg}")
                    cfg_choice = input("Select config number: ").strip()
                    try:
                        cfg_idx = int(cfg_choice) - 1
                        if 0 <= cfg_idx < len(configs):
                            config_name = configs[cfg_idx]
                        else:
                            print("❌ Invalid selection.")
                            continue
                    except ValueError:
                        print("❌ Invalid selection.")
                        continue
                elif len(configs) == 1:
                    config_name = configs[0]
                
                # Now show available splits with sizes
                splits = get_dataset_splits(dataset_name, config_name)
                print(f"\n📊 Available splits:")
                
                # Try to get split sizes (may take a moment)
                try:
                    from datasets import load_dataset_builder
                    if config_name:
                        builder = load_dataset_builder(dataset_name, config_name)
                    else:
                        builder = load_dataset_builder(dataset_name)
                    split_info = builder.info.splits
                    for s in splits:
                        if s in split_info:
                            print(f"   • {s}: {split_info[s].num_examples:,} samples")
                        else:
                            print(f"   • {s}: (size unknown)")
                except Exception:
                    for s in splits:
                        print(f"   • {s}")
                
                # Offer options: single split, all, or auto-split
                if len(splits) == 1:
                    print(f"\n⚠️ Only '{splits[0]}' split available.")
                    auto_split = input("Auto-split into 80% train / 20% test? (y/n, default y): ").strip().lower()
                    if auto_split != 'n':
                        selected_splits = [splits[0]]
                        do_auto_split = True
                    else:
                        selected_splits = [splits[0]]
                        do_auto_split = False
                else:
                    do_auto_split = False
                    print(f"\nOptions: Enter split name, 'all' for all splits, or comma-separated list")
                    split_input = input(f"Select split(s) (default 'train'): ").strip() or 'train'
                    
                    if split_input.lower() == 'all':
                        selected_splits = splits
                        print(f"📦 Will download all {len(splits)} splits")
                    elif ',' in split_input:
                        selected_splits = [s.strip() for s in split_input.split(',') if s.strip() in splits]
                        if not selected_splits:
                            print(f"⚠️ No valid splits found, using 'train'")
                            selected_splits = ['train'] if 'train' in splits else [splits[0]]
                    else:
                        if split_input in splits:
                            selected_splits = [split_input]
                        else:
                            print(f"⚠️ '{split_input}' not found, using 'train'")
                            selected_splits = ['train'] if 'train' in splits else [splits[0]]
                
                max_samples_str = input("Max Samples (default 10000, 0=all): ").strip() or '10000'
                max_samples = int(max_samples_str) if max_samples_str != '0' else None
                text_field = input("Text Field (default 'text'): ").strip() or 'text'
                local_name = input("Local Name (default auto): ").strip() or None
                
                try:
                    from data import download_hf_dataset
                    
                    if do_auto_split:
                        # Download and auto-split 80/20
                        print("📊 Downloading and auto-splitting 80/20...")
                        base_name = local_name or dataset_name.replace('/', '_')
                        if config_name:
                            base_name += f"_{config_name}"
                        
                        path = download_hf_dataset(
                            dataset_name=dataset_name,
                            split=selected_splits[0],
                            max_samples=max_samples,
                            text_field=text_field,
                            local_name=f"{base_name}_full",
                            force_download=True,
                            config_name=config_name
                        )
                        
                        # Now split the downloaded data
                        import json
                        from data import DATA_STORE_PATH
                        full_path = os.path.join(DATA_STORE_PATH, f"{base_name}_full", "texts.jsonl")
                        
                        if os.path.exists(full_path):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                all_texts = [json.loads(line) for line in f]
                            
                            # Shuffle and split
                            import random
                            random.shuffle(all_texts)
                            split_idx = int(len(all_texts) * 0.8)
                            train_texts = all_texts[:split_idx]
                            test_texts = all_texts[split_idx:]
                            
                            # Save train split
                            train_dir = os.path.join(DATA_STORE_PATH, f"{base_name}_train")
                            os.makedirs(train_dir, exist_ok=True)
                            with open(os.path.join(train_dir, "texts.jsonl"), 'w', encoding='utf-8') as f:
                                for t in train_texts:
                                    f.write(json.dumps(t) + '\n')
                            with open(os.path.join(train_dir, "metadata.json"), 'w') as f:
                                json.dump({'type': 'auto_split', 'split': 'train', 'num_samples': len(train_texts), 'source': dataset_name}, f, indent=2)
                            
                            # Save test split
                            test_dir = os.path.join(DATA_STORE_PATH, f"{base_name}_test")
                            os.makedirs(test_dir, exist_ok=True)
                            with open(os.path.join(test_dir, "texts.jsonl"), 'w', encoding='utf-8') as f:
                                for t in test_texts:
                                    f.write(json.dumps(t) + '\n')
                            with open(os.path.join(test_dir, "metadata.json"), 'w') as f:
                                json.dump({'type': 'auto_split', 'split': 'test', 'num_samples': len(test_texts), 'source': dataset_name}, f, indent=2)
                            
                            print(f"✅ Created {base_name}_train ({len(train_texts)} samples)")
                            print(f"✅ Created {base_name}_test ({len(test_texts)} samples)")
                        else:
                            print(f"⚠️ Could not find downloaded data to split.")
                    else:
                        # Normal download - loop through selected splits
                        base_name = local_name or dataset_name.replace('/', '_')
                        if config_name:
                            base_name += f"_{config_name}"
                        
                        for split in selected_splits:
                            # Add split name to local name if downloading multiple
                            if len(selected_splits) > 1:
                                split_local_name = f"{base_name}_{split}"
                            else:
                                split_local_name = local_name  # User's choice or auto
                            
                            print(f"\n⬇️ Downloading split: {split}...")
                            path = download_hf_dataset(
                                dataset_name=dataset_name,
                                split=split,
                                max_samples=max_samples,
                                text_field=text_field,
                                local_name=split_local_name,
                                force_download=True,
                                config_name=config_name
                            )
                            print(f"✅ Downloaded to: {path}")
                        
                        if len(selected_splits) > 1:
                            print(f"\n📦 Downloaded {len(selected_splits)} splits successfully!")
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    
            elif choice == '2':
                print("\n--- Import Text File ---")
                filepath = input("File Path: ").strip()
                if not filepath or not os.path.exists(filepath):
                    print("❌ File not found.")
                    continue
                
                local_name = input("Local Name (default: filename): ").strip() or None
                chunk_str = input("Chunk Size (default: split by paragraphs): ").strip()
                chunk_size = int(chunk_str) if chunk_str else None
                
                try:
                    path = import_text_file(
                        filepath=filepath,
                        local_name=local_name,
                        chunk_size=chunk_size
                    )
                    print(f"✅ Imported to: {path}")
                except Exception as e:
                    print(f"❌ Import failed: {e}")
                    
            elif choice == '3':
                print("🔄 Refreshing...")
                # Loop will refresh automatically
                
            elif choice == 'b':
                break
            else:
                print("❌ Invalid option.")

    def run(self):
        print(f"\nWelcome to DGE Lab {__version__} (Forensic & Checkpoint Edition)")
        while True:
            print("\n" + "="*50)
            print("🤖  DGE MAIN MENU")
            print("="*50)
            if self.model_name:
                print(f"🔹 Current Model: {self.model_name}")
            else:
                print("🔸 No Active Model")
                
            print("\n--- Core Actions ---")
            print("0. 🆕 Create New Model")
            print("l. 📂 Load Model")
            print("t. 🏋️  Train Model")
            print("e. 🧠  Expand Model")
            print("s. 💾  Save Model")
            
            print("\n--- Tools & Analysis ---")
            print("i. 🔍  Inspect Model")
            print("r. 🎲  Run Inference")
            print("f. 📊  Full Skill Benchmark")
            print("d. 🗂️  Dataset Manager")
            print("x. 🧪  Experiments (Automated)")
            
            print("\n--- System ---")
            print("q. 🚪  Exit") # Kept existing 'q' for consistency
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '0':
                self.create_model()
            elif choice == 'l':
                self.load_menu()
            elif choice == 't':
                self.train_menu()
            elif choice == 'e':
                self.expand_menu()
            elif choice == 'i':
                self.inspect_model()
            elif choice == 'r':
                self.inference_menu()
            elif choice == 's':
                self.save_model()
            elif choice == 'f':
                self.benchmark_skills()
            elif choice == 'x':
                self.experiment_menu()
            elif choice == 'd':
                self._dataset_menu()
            elif choice == 'q':
                print("👋 Exiting... Goodbye!")
                break
            else:
                print("❌ Invalid option.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGE Lab V" + __version__)
    parser.add_argument('-x', '--experiment', type=str, metavar='N',
                        help='Run experiment N directly and exit (e.g., -x 18 for Directed Synergy)')
    args = parser.parse_args()
    
    lab = DGELab()
    
    if args.experiment:
        # Direct experiment execution mode
        exp_choice = args.experiment.strip().lower()
        print(f"\n🚀 CLI Mode: Running Experiment '{exp_choice}'...")
        lab._run_experiment_by_choice(exp_choice)
        print("\n✅ Experiment complete. Exiting.")
    else:
        # Interactive mode
        lab.run()
