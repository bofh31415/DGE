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
            print("b. 🔙 Back")
            
            choice = input("\nSelect Training Task: ").strip().lower()
            
            if choice == 'b':
                return
            
            vocab = self.model.token_emb.num_embeddings
            
            #Define checkpoint callback
            def checkpoint_fn(step):
                self.save_model()
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
            added = int(input("Enter added d_model (e.g. 64): "))
            
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

    def save_model(self, create_snapshot=True):
        if self.model is None:
            print("No model to save!")
            return
            
        save_dir = os.path.join(self.models_dir, self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare config
        config = {
            'd_model': self.model.d_model,
            'n_layer': len(self.model.layers),
            'n_head': self.model.layers[0].n_head, 
            'model_name': self.model_name,
            'vocab_size': self.model.token_emb.num_embeddings,
            'trained_skills': list(self.trained_skills),
            'global_step': self.global_step
        }
        
        # Save main/latest files
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'weights.pt'))
        
        # Create versioned snapshot
        if create_snapshot:
            snapshots_dir = os.path.join(save_dir, 'snapshots')
            os.makedirs(snapshots_dir, exist_ok=True)
            
            snapshot_name = f"step_{self.global_step:06d}"
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
            vocab_size = config.get('vocab_size', 1000) # Default for legacy models
            self.model = DGESimpleTransformer(
                vocab_size=vocab_size,
                d_model=config['d_model'],
                n_layer=config['n_layer'],
                n_head=config['n_head']
            )
            # Load Weights
            self.model.load_state_dict(torch.load(os.path.join(load_dir, 'weights.pt')))
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
            
        except (ValueError, FileNotFoundError) as e:
            print(f"[Error] loading model: {e}")

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
        
        if choice == '1':
            self.run_dge_validation_chain()
        elif choice == '2':
            # Single Experiment (Legacy Option 8)
            print("\n🧪 RUNNING SINGLE EXPERIMENT (V 0.2.0 Spec)...")
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
        else:
            print(f"❌ Unknown experiment choice: '{choice}'")
            print("Available experiments: 1, 2, 17, 18")
            print("  1: DGE Validation Chain (V 0.2.x Baseline)")
            print("  2: Single Experiment (V 0.2.0 Spec)")
            print("  17: Verify V21 RBF (Imprint + Rescue V4)")
            print("  18: Directed Synergy (Phase 2 Seed Fund)")

    def experiment_menu(self):
        while True:
            print("\n" + "="*50)
            print("🧪  EXPERIMENTS MENU")
            print("="*50)
            print("1. 🔗 Run DGE Validation Chain (V 0.2.x Baseline)")
            print("2. 🧪 Run Single Experiment (V 0.2.0 Spec)")
            print("3. ⛓️  Run Experiment Chain V2 (V 0.3.x - Noise Init Hypotheses)")
            print("4. ⚙️  Run Experiment Chain V3 (V 0.3.x - Gradient Rescue & Orthogonality)")
            print("17. 🧪 Verify V21 RBF (Imprint + Rescue V4)")
            print("18. 🟢 Directed Synergy (Phase 2 Seed Fund)")
            print("b. 🔙 Back")
            print("q. 🚪 Exit")
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '1':
                self.run_dge_validation_chain()
            elif choice == '2':
                # Single Experiment (Legacy Option 8)
                print("\\n🧪 RUNNING SINGLE EXPERIMENT (V 0.2.0 Spec)...")
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
            elif choice == '4':
                # Experiment Chain V3
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
            self.save_model()
            
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
            print("2. 📂 Load Model")
            print("t. 🏋️  Train Model")
            print("e. 🧠  Expand Model")
            print("s. 💾  Save Model")
            
            print("\n--- Tools & Analysis ---")
            print("i. 🔍  Inspect Model")
            print("r. 🎲  Run Inference")
            print("f. 📊  Full Skill Benchmark")
            print("x. 🧪  Experiments (Automated)")
            
            print("\n--- System ---")
            print("q. 🚪  Exit") # Kept existing 'q' for consistency
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == '0':
                self.create_model()
            elif choice == '2':
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
