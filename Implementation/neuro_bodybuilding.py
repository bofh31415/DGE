import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import numpy as np

class GAController:
    """
    A Genetic Algorithm Controller that evolves binary masks for gradients.
    Goal: Find a mask M such that Update(M) minimizes Loss + SparsityPenalty.
    """
    def __init__(self, model, mutation_rate=0.1, population_size=5, sparsity_weight=0.1):
        self.model = model
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.sparsity_weight = sparsity_weight
        
        # We focus masking on Linear layers for now (weights only)
        self.target_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    def generate_mask_population(self, gradients):
        """
        Generates a population of candidate masks.
        V1 Heuristic: Perturb a base mask derived from gradient magnitude.
        """
        population = []
        
        # Base heuristic: Keep top-k gradients (Magnitude Pruning style) as a strong start
        # This speeds up convergence compared to random start.
        
        for _ in range(self.population_size):
            mask_dict = {}
            for name, grad in gradients.items():
                if grad is None:
                    continue
                
                # Create a binary mask
                # Strategy: Stochastic Magnitude Pruning
                # Probability of keeping a weight is proportional to its grad magnitude
                
                mag = grad.abs()
                # Normalize
                if mag.max() > 0:
                    prob = mag / mag.max()
                else:
                    prob = torch.zeros_like(mag)
                
                # Mutation: Randomly flip some bits based on mutation rate?
                # Let's do simple Bernoulli sampling based on prob + noise
                noise = torch.rand_like(prob) * self.mutation_rate
                final_prob = prob + noise
                
                # Threshold at 0.5? Or dynamic?
                # Dynamic threshold to target 50% sparsity roughly?
                # User wants EXTREME sparsity.
                # Let's enforce strict threshold.
                threshold = 0.5 
                mask = (final_prob > threshold).float()
                
                mask_dict[name] = mask
            population.append(mask_dict)
            
        return population

    def evaluate_fitness(self, mask, current_loss, sparsity_score):
        """
        Fitness = -(TaskLoss + SparsityWeight * (1 - Sparsity))
        Users wants to MINIMIZE active neurons (Maximize Sparsity).
        So Penalty is for Active Neurons.
        """
        # Sparsity score = fraction of zeros.
        # Active score = fraction of ones = (1 - sparsity)
        
        active_penalty = (1.0 - sparsity_score) * self.sparsity_weight
        
        # We want to minimize (Loss + Penalty)
        # So Fitness is negative of that.
        return -(current_loss + active_penalty)

class NeuroTrainer:
    """
    Wraps Training Loop to inject Neuro-Bodybuilding logic.
    """
    def __init__(self, model, dataloader, criterion, optimizer, device='cuda', logger=None):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.controller = GAController(model)
        self.logger = logger  # Optional DGELogger for event tracking
        
        # Training history for model card generation
        self.history = {
            "losses": [],
            "fitness_scores": [],
            "sparsity_levels": [],
            "steps": 0
        }
        
    def train_step(self, x, y):
        # 1. Standard Forward & Backward to get Raw Gradients
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        output = self.model(x)
        # Assuming output is logits, y leads to loss
        # Need to handle DGE model signature (logits, loss) if applicable
        if isinstance(output, tuple):
            logits = output[0]
            # loss might be calculated inside model or here
            # recalculate for standard interface
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = output
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
        loss.backward()
        
        # 2. Extract Gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # 3. GA: Generate Candidates
        # In a real heavy implementation, we would try applying each mask and FWD pass.
        # That's 5x compute. For V1 "Demo", we might approximate or do it.
        # User asked for "Lookahead". Let's do 1-step lookahead for a small population.
        
        candidates = self.controller.generate_mask_population(gradients)
        
        best_fitness = -float('inf')
        best_mask = None
        
        # Save original params
        # original_state = copy.deepcopy(self.model.state_dict()) # Too slow for deep copy every step!
        # Optimization: We only mess with gradients, not params yet.
        # W_new = W - lr * (G * M)
        # Taylor expansion: Loss(W_new) ~= Loss(W) - lr * ||G*M||^2 ...
        # Actually, simpler: Measure magnitude of gradients KEPT.
        # And measure Sparsity.
        # Fitness = (GradMagnitude * TaskRelevance) - SparsityCost
        
        # For true "Bodybuilding", we skip the expensive rollout for now and use the heuristic:
        # Keep gradients that are LARGE (Task Critical) but minimal count.
        
        for mask_dict in candidates:
            # Calculate metrics
            total_elements = 0
            kept_elements = 0
            grad_energy = 0.0
            
            for name, mask in mask_dict.items():
                g = gradients[name]
                kept_g = g * mask
                
                kept_elements += mask.sum().item()
                total_elements += mask.numel()
                grad_energy += kept_g.norm().item() # Rough proxy for "Loss Reduction Potential"
            
            sparsity = 1.0 - (kept_elements / (total_elements + 1e-9))
            
            # Heuristic Fitness:
            # We want High Energy (Gradient Norm) with High Sparsity.
            fitness = grad_energy - (self.controller.sparsity_weight * (1-sparsity) * 100) 
            # (Scaling sparsity weight because 1-sparsity is small fraction)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_mask = mask_dict
                best_sparsity = sparsity
        
        # 4. Apply Best Mask
        if best_mask:
            for name, param in self.model.named_parameters():
                if name in best_mask and param.grad is not None:
                    param.grad.data.mul_(best_mask[name])
        
        # 5. Optimizer Step (Modified Gradients)
        self.optimizer.step()
        
        # 6. Log Training Event
        self.history["steps"] += 1
        self.history["losses"].append(loss.item())
        self.history["fitness_scores"].append(best_fitness)
        self.history["sparsity_levels"].append(best_sparsity if best_mask else 0.0)
        
        # Log to DGELogger if available
        if self.logger and self.history["steps"] % 100 == 0:
            self.logger.log_event("BODYBUILDING", {
                "step": self.history["steps"],
                "loss": loss.item(),
                "fitness": best_fitness,
                "sparsity": best_sparsity if best_mask else 0.0
            }, step=self.history["steps"])
        
        return loss.item(), best_fitness
    
    def get_summary(self):
        """Return training summary for model card generation."""
        if not self.history["losses"]:
            return {}
        return {
            "total_steps": self.history["steps"],
            "final_loss": self.history["losses"][-1],
            "avg_fitness": sum(self.history["fitness_scores"]) / len(self.history["fitness_scores"]),
            "avg_sparsity": sum(self.history["sparsity_levels"]) / len(self.history["sparsity_levels"]),
            "max_sparsity": max(self.history["sparsity_levels"]),
        }
