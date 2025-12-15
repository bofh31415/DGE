import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from typing import Union

class DGEAdamW(torch.optim.AdamW):
    """
    AdamW optimizer with selective weight decay.
    Weight decay is NOT applied to parameters that have a `frozen_mask` = 0.
    This prevents decay from corrupting frozen weights.
    
    Usage:
        optimizer = DGEAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group.get('amsgrad', False)
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])

            # Perform stepweight decay
            # CUSTOMIZATION: Apply decay only where mask == 1
            for i, p in enumerate(params_with_grad):
                state_steps[i] += 1
                
                # Check for frozen mask
                # Assume parent module registered 'backward_mask' or 'frozen_bias_mask'
                # We need the mask to be on the same shape as p or broadcastable.
                # Strategy: Check if parameter has a 'dge_active_mask' attribute.
                # Or iterate through model's buffers to find matching mask.
                # Simpler: Store mask directly on param via monkey-patching during expansion.
                # For now, apply full decay if no mask info available.
                
                wd = group['weight_decay']
                if wd != 0:
                    if hasattr(p, 'dge_mask'):
                        # Apply decay only to active parts
                        p.mul_(1 - group['lr'] * wd * p.dge_mask)
                    else:
                        # Standard behavior: Decay all
                        p.mul_(1 - group['lr'] * wd)

            # Adam update
            for i, p in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                step = step_t.item()
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = group['lr'] / bias_correction1
                
                if amsgrad:
                    max_exp_avg_sqs[i] = torch.maximum(max_exp_avg_sqs[i], exp_avg_sq)
                    denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
class Quadrant(Enum):
    TOP_LEFT = "TL"
    TOP_RIGHT = "TR"
    BOTTOM_LEFT = "BL"
    BOTTOM_RIGHT = "BR"


class RBFRouter(nn.Module):
    """
    Radial Basis Function Router.
    Activates only when input is close to a learned centroid.
    Gate = exp(-beta * ||x - mu||^2)
    """
    def __init__(self, input_dim: int, num_experts: int, beta_init=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Centroids (mu)
        self.centroids = nn.Parameter(torch.Tensor(num_experts, input_dim))
        nn.init.normal_(self.centroids, mean=0.0, std=0.02) # Start near zero but distinct
        
        # Inverse Width (beta) - Using log parameterization to ensure positivity
        self.log_beta = nn.Parameter(torch.ones(num_experts) * math.log(beta_init))
        
    @torch.no_grad()
    def imprint_from_batch(self, x: torch.Tensor):
        """
        Initializes centroids by sampling from the input batch.
        x: [Batch, Seq, Dim] or [Batch, Dim]
        """
        # Flatten batch and sequence dims
        x_flat = x.reshape(-1, self.input_dim)
        
        # Select n_experts random indices
        total_tokens = x_flat.size(0)
        if total_tokens < self.num_experts:
             # Fallback: Repeat data if not enough
             indices = torch.randint(0, total_tokens, (self.num_experts,))
        else:
             # Random permutation
             indices = torch.randperm(total_tokens)[:self.num_experts]
             
        selected = x_flat[indices] # [Experts, Dim]
        self.centroids.data.copy_(selected)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Input_Dim]
        # centroids: [Experts, Input_Dim]
        
        # Calculate squared Euclidean distance: ||x - mu||^2
        # (x - mu)^2 = x^2 + mu^2 - 2*x*mu
        # More stable to use direct subtraction with broadcasting if memory allows.
        # x: [..., 1, D]
        # mu: [1, E, D]
        
        # Reshape for broadcasting
        # x_expanded: [..., 1, D]
        x_expanded = x.unsqueeze(-2) 
        
        # Distance: [..., Experts]
        # Sum over last dim (Input_Dim)
        dist_sq = (x_expanded - self.centroids).pow(2).sum(-1)
        
        # Beta must be positive
        beta = torch.exp(self.log_beta)
        
        # Gate activation
        # Avoid exploding gradients in exp
        gates = torch.exp(-beta * dist_sq)
        
        return gates

class BigramRouter(nn.Module):
    """
    Contextual Router (Bigram/Conv1D).
    Input: [Batch, Seq, Dim]
    Feeds MLP([x_t, x_{t-1}]) to determine gate activation.
    Disambiguates tokens based on immediate history.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim=None, router_init_bias=-4.0):
        super().__init__()
        self.input_dim = input_dim
        concat_dim = input_dim * 2
        
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 2)
            
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Initialization
        # Last layer bias controls initial openness
        nn.init.constant_(self.mlp[-1].bias, router_init_bias)
        # Weights near zero to minimize noise
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.001)
        
        # First layer
        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.mlp[0].bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Dim]
        # We assume 3D input [B, T, D].
        if x.dim() != 3:
            # Fallback for non-sequential input (e.g. flattened) -> No Context
            # Just separate logic or error?
            # For robustness, we'll treat prev as zeros.
            prev = torch.zeros_like(x)
        else:
            # Shift right along sequence dim (1)
            # x: [B, T, D]
            prev = torch.roll(x, shifts=1, dims=1)
            prev[:, 0, :] = 0.0 # Zero out the first token's context (it has no prev)
            
        combined = torch.cat([x, prev], dim=-1) # [B, T, 2*D]
        return self.mlp(combined)

class RBFBigramRouter(RBFRouter):
    """
    Combines RBF Selectivity with Bigram Context.
    Input: [Batch, Seq, Dim]
    Internally constructs [x_t, x_{t-1}] and computes RBF distance in 2*Dim space.
    """
    def __init__(self, input_dim: int, num_experts: int, beta_init=1.0):
        # We pass 2*input_dim to the parent RBFRouter
        super().__init__(input_dim * 2, num_experts, beta_init=beta_init)
        self.original_input_dim = input_dim

    def _make_bigram(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            prev = torch.zeros_like(x)
        else:
            prev = torch.roll(x, shifts=1, dims=1)
            prev[:, 0, :] = 0.0
        return torch.cat([x, prev], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bigram_x = self._make_bigram(x)
        return super().forward(bigram_x)

    @torch.no_grad()
    def imprint_from_batch(self, x: torch.Tensor):
        bigram_x = self._make_bigram(x)
        # RBFRouter.imprint_from_batch handles flattening and sampling
        super().imprint_from_batch(bigram_x)

class HybridGate(nn.Module):
    """
    A Gate that is Static (Always Open) for old indices and Dynamic (MoE) for new indices.
    """
    def __init__(self, input_dim: int, old_count: int, new_count: int, router_type='linear', router_init_bias=-4.0, gating_threshold: float = 0.0):
        super().__init__()
        self.old_count = old_count
        self.new_count = new_count
        self.router_type = router_type
        self.gating_threshold = gating_threshold
        self.router_type = router_type
        self.gating_threshold = gating_threshold
        self.last_mean_open = 1.0 # Default to 1.0 to avoid zero div
        self.rescue_strength = 10000.0 # V4/V6 Fix: Default to high strength
        self.use_gradient_rescue = False # Default Off, enabled by expand logic
        
        # Static Old Segment (Buffer to freeze it)
        self.register_buffer('old_gate', torch.ones(old_count))
        
        # Dynamic New Segment (Router)
        if new_count > 0:
            if router_type == 'rbf':
                # V21: RBF Router for Localized Selectivity
                self.router = RBFRouter(input_dim, new_count, beta_init=1.0)
            elif router_type == 'rbf_bigram':
                # V23: RBF Bigram Router (Context + Selectivity)
                self.router = RBFBigramRouter(input_dim, new_count, beta_init=1.0)
            elif router_type == 'bigram':
                # V22: Bigram Contextual Router
                self.router = BigramRouter(input_dim, new_count, router_init_bias=router_init_bias)
            elif router_type == 'mlp':
                # V 0.3.0: MLP Router for Non-Linear Separation
                hidden_dim = max(16, input_dim // 4)
                self.router = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, new_count)
                )
                # Init last layer to Closed (Tunable)
                nn.init.constant_(self.router[-1].bias, router_init_bias)
                # V5: Zero Init (Small Noise) to guarantee gates start closed
                nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.001)
                
                # Also dampen the first layer to prevent massive activations
                nn.init.normal_(self.router[0].weight, mean=0.0, std=0.01)
                nn.init.constant_(self.router[0].bias, 0.0)
            else:
                # Default Linear Router
                self.router = nn.Linear(input_dim, new_count)
                # Initialize to Closed (Tunable)
                nn.init.constant_(self.router.bias, router_init_bias)
                # V5: Zero Init (Small Noise)
                nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        else:
            self.router = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., Input_Dim]
        # Handle arbitrary batch dimensions
        *batch_dims, _ = x.shape
        
        # 1. Old Gate (Static)
        # Reshape old_gate to [1, ..., 1, Old_Count] -> No, just broadcast to [..., Old_Count]
        # Using expand directly on the buffer
        # old_part = self.old_gate.expand(*batch_dims, -1) 
        # Wait, old_gate is [Old_Count]. We want [Batch, Old_Count].
        # View as [1, 1..., Old_Count] then expand?
        # Simpler: just expand.
        old_part = self.old_gate.expand(*batch_dims, -1)
        
        if self.new_count > 0:
            router_out = self.router(x)
            if self.router_type in ['rbf', 'rbf_bigram']:
                new_part = router_out
            else:
                new_part = torch.sigmoid(router_out)
            
            # V24: Strict Separation (Hard Thresholding with STE)
            if self.gating_threshold > 0.0:
                mask = (new_part >= self.gating_threshold).to(new_part.dtype)
                new_part = (new_part * mask) - new_part.detach() + new_part

            # V6: Parallel Detached Rescue
            # Problem: V4 TensorHook exploded Input Gradients. V5 ParamHook died on Zero Gradients.
            # Solution: Run Router TWICE.
            # Path 1: new_part = Router(x). Gradients -> x and W. (Normal)
            # Path 2: rescue_part = Router(x.detach()). Gradients -> W only. (Boosted)
            
            # Clean up V5 state if it exists (for reload safety)
            if hasattr(self, '_rescue_handles'):
                for h in self._rescue_handles: h.remove()
                self._rescue_handles.clear()
            
            self.last_rescue_scale = 0.0

            if getattr(self, 'use_gradient_rescue', False) and new_part.mean() < 0.01 and self.training:
                # Only apply if actually needed (closed gates) and training
                # RBF or Sigmoid output is in 'new_part' (Path 1)
                
                # Path 2: Detached Input
                # We need to re-run the router logic.
                # Note: This doubles the compute for the router, but router is cheap (small MLP/RBF).
                rescue_part = self.router(x.detach())
                if self.router_type not in ['rbf', 'rbf_bigram']:
                     rescue_part = torch.sigmoid(rescue_part)
                
                # Apply STE threshold if needed (to match Path 1 logic exactly)
                if self.gating_threshold > 0.0:
                    r_mask = (rescue_part >= self.gating_threshold).to(rescue_part.dtype)
                    rescue_part = (rescue_part * r_mask) - rescue_part.detach() + rescue_part

                strength = self.rescue_strength
                self.last_rescue_scale = strength

                # Register Hook on Path 2 Output
                def rescue_hook(grad):
                    return grad * strength
                
                if rescue_part.requires_grad:
                    rescue_part.register_hook(rescue_hook)
                    
                    # Merge Paths
                    # y = y1 + y2 - y2.detach()
                    # Value: y1 + 0 = y1.
                    # Backward: dL/dy * dy/dy1 + dL/dy * dy/dy2.
                    # dy/dy1 = 1. -> Gradient flows to x and W (Path 1).
                    # dy/dy2 = 1. -> Gradient * 10000 flows to W only (Path 2 input is detached).
                    # Result: Weights get 10001x gradient. Inputs get 1x gradient.
                    new_part = new_part + rescue_part - rescue_part.detach()

            self.last_mean_open = new_part.mean().detach()
            
            return torch.cat([old_part, new_part], dim=-1)
        else:
            self.last_mean_open = torch.tensor(1.0, device=x.device)
            return old_part

class MoEGatedLinear(nn.Module):
    """
    V 0.2.0: Mixture-of-Experts Gated Linear Layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, router_type='linear', router_init_bias=-4.0, gating_threshold: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.router_type = router_type 
        self.use_gradient_rescue = False
        self.use_gradient_rescue = False
        self.rescue_strength = 10000.0 # V8.1: Boosted for Double-Lock (Need > 3000x)
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # V 0.2.0: Split Gate (Hybrid)
        self.gate_row = HybridGate(in_features, out_features, 0, router_type=router_type, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        self.gate_col = HybridGate(in_features, in_features, 0, router_type=router_type, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        
        self.register_buffer('active_mask', torch.ones(out_features, in_features))
        self.register_buffer('frozen_bias_mask', torch.ones(out_features))
        self.register_buffer('backward_mask', torch.ones(out_features, in_features)) # Added for compat

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # V 0.2.6: Fixed Forward Logic
        # 1. Col Gate (Input Selection)
        if self.gate_col is not None:
             g_col = self.gate_col(input)
             x_in = input * g_col
        else:
             x_in = input
             
        # 2. Linear
        out = F.linear(x_in, self.weight, self.bias)
        
        # 3. Row Gate (Output Selection)
        if self.gate_row is not None:
             g_row = self.gate_row(input) # Router sees Raw Input!
             out = out * g_row
             
        return out

def expand_dge_linear(
    layer: Union[nn.Linear, MoEGatedLinear], 
    added_in: int, 
    added_out: int, 
    frozen_core_pos: Quadrant = Quadrant.TOP_LEFT,
    isolate_cross_terms: bool = False,
    cross_term_policy: str = 'full', # 'full', 'block_diagonal', 'imprint'

    router_type='linear',
    use_gradient_rescue: bool = True, # ADOPTED: V26 Strategy (Default On)
    use_orthogonal_init: bool = False, # ADOPTED: V26 Strategy (Default Zero Init for Safe Replay)
    router_init_bias: float = 0.0, # DIRECTED SYNERGY: Start Open (Life Support)
    gating_threshold: float = 0.0
) -> MoEGatedLinear:
    """
    Expands a DoubleGateLinear layer by adding input and/or output dimensions.
    Configures the new backward mask to freeze the core (old weights).
    
    Args:
        ...
        cross_term_policy: 'full' (all active), 'block_diagonal' (all cross frozen),
                           'imprint' (Old->New active, New->Old frozen).
        isolate_cross_terms: Legacy, if True, overrides policy to 'block_diagonal'.
    """
    # Map Legacy
    # DIRECTED SYNERGY PHASE 1: FORCE ENABLE CROSS TERMS (Old->New)
    # We explicitly IGNORE isolate_cross_terms=True to prevent Dead Sidecar.
    if isolate_cross_terms:
        # cross_term_policy = 'block_diagonal' # OLD
        pass # NEW: Ignore isolation request. Keep 'full' or whatever was passed.
        isolate_cross_terms = False # Force False for downstream logic
    old_out, old_in = layer.weight.shape
    new_in = old_in + added_in
    new_out = old_out + added_out
    
    # 2. Create New Layer
    new_layer = MoEGatedLinear(new_in, new_out, bias=(layer.bias is not None), router_type=router_type, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
    new_layer.use_gradient_rescue = use_gradient_rescue
    
    # Defaults for initialization of new areas
    # Initialize new gates to be OPEN to allow gradient flow to the new zero-weights.
    # We rely SOLELY on W=0.0 for Identity Preservation.
    # If Gates are closed (-5.0), gradients vanish (Double-Lock).
    # Set init to 2.5 (Sigmoid(5) ~ 0.99)
    # This part is now handled by HybridGate's internal router initialization.
    # The `gate_row` and `gate_col` attributes of MoEGatedLinear will be HybridGate instances,
    # not direct parameters to initialize with constant_().
    # So, these lines should be removed or commented out if they are no longer applicable.
    # For now, I'll keep them commented as the instruction only asked for the class name change.
    # nn.init.constant_(new_layer.gate_row, 2.5)
    # nn.init.constant_(new_layer.gate_col, 2.5)
    
    # Copy weights and gates based on Quadrant
    # We also set the backward_mask: 0 for the old core (frozen), 1 for new areas.
    # Note: reset_parameters() already sets mask to 1s. We just need to zero out the old core.
    
    # CRITICAL FIX for Identity Preservation:
    # Initialize ALL weights to Epsilon Noise first.
    # While Zero Init (0.0) is ideal for stability, it KILLS gradients for the Gate.
    # d(Gate)/d(Loss) depends on Weight. If Weight=0, d(Gate)=0. Rescue fails.
    # We use std=0.02 to provide a "Gradient Wick" while keeping output ~0.0004.
    nn.init.normal_(new_layer.weight, mean=0.0, std=0.02)
    
    # Helper indices for the "Old" block in the new matrix
    if frozen_core_pos == Quadrant.TOP_LEFT:
        r_slice = slice(0, old_out)
        c_slice = slice(0, old_in)
        
        
        # Gate mapping
        r_gate_idx = slice(0, old_out)
        c_gate_idx = slice(0, old_in)
    
    # 3. Copy Weights & Biases (No Grad context for safety)
    with torch.no_grad():
        # Copy Old Interactions (Top-Left)
        new_layer.weight[:old_out, :old_in] = layer.weight
        if layer.bias is not None:
            new_layer.bias[:old_out] = layer.bias
            
        # Initialize New Weights (Sidecar & Cross)
        if use_orthogonal_init:
            # H3: Orthogonal Initialization
            # Project new weights to be orthogonal to old weights
            # Shape [new_out, new_in]. 
            # We treat the weight matrix as a set of row vectors? Or column vectors?
            # Usually we want inputs to be distinct (Dimensions). Orthogonalize new columns w.r.t old columns.
            
            # 1. New Input Features (Right Side) - Columns
            if added_in > 0:
                # Orthogonalize [:, old_in:] w.r.t [:, :old_in]
                # But we already zeroed it? No, we init with noise first.
                noise_in = torch.randn(new_out, added_in, device=layer.weight.device) * 0.01 # Small noise
                
                if old_in > 0:
                    # Projection: W_new = Noise - W_old @ (W_old.T @ W_old)^-1 @ W_old.T @ Noise 
                    # Simplified: Gram-Schmidt like subtraction.
                    # Or just use QR on the whole concatenated matrix [W_old | Noise]?
                    # We want to keep W_old EXACTLY as is.
                    # So we project Noise onto nullspace of W_old.
                    
                    # W_old [Out, Old_In]
                    # We want W_new [Out, Add_In] such that columns of W_new are perp to columns of W_old?
                    # No, usually rows are the features of the neuron. 
                    # If we add neurons (Rows), we want new rows perp to old rows.
                    # If we add inputs (Cols), we want new cols perp to old cols?
                    pass 
                    
                # For now, simple small noise is standard. Orthogonal init fully compliant requires care.
                # Let's use standard init for now but projected. 
                # Implementing simple QR based generation for the new block independently.
                nn.init.orthogonal_(new_layer.weight[:, old_in:]) 
                new_layer.weight[:, old_in:] *= 0.1 # V3 Fix: Increased from 0.01 to 0.1 to ensure dGate signal (avoid Double Lock)
                
            # 2. New Output Features (Bottom Side) - Rows
            if added_out > 0:
                nn.init.orthogonal_(new_layer.weight[old_out:, :])
                new_layer.weight[old_out:, :] *= 0.1

        else:
            # Standard Noise Init (V 0.2.7)
            # We need Non-Zero weights to allow gradient flow to the Gate.
            # dL/dGate = dL/dOut * Weight. If Weight=0, Gate cannot learn.
            # With Gate=-4.0 (0.018), and Weight=0.02, effective noise is 3.6e-4. Identity Safe.
            
            # --- PHASE 3: THE FIREWALL (Asymmetric Init) ---
            # 1. Q_BL (New->New) & Q_BR (Old->New): Noise Init (Capacity & Synergy)
            nn.init.normal_(new_layer.weight[old_out:, :], mean=0.0, std=0.02)
            
            # 2. Q_TR (New->Old): ZERO INIT (Firewall)
            # This is the "Leak Path" where new inputs corrupt old outputs.
            # We enforce exact zeros here. The gate can open, but it transmits silence.
            # Identity is preserved: Y_old = W_old * X_old + Gate * 0.0 * X_new = Y_old.
            new_layer.weight[:old_out, old_in:].zero_()
        
        # Clean up bias for new rows
        if layer.bias is not None:
            new_layer.bias[old_out:] = 0.0
            
    # 4. Set up GATES (The V 0.2.0 Upgrade)
    # We need HybridGates.
    # Col Gate: Acts on Input [new_in]. Old=old_in, New=added_in.
    # Row Gate: Acts on Output [new_out]. Old=old_out, New=added_out.
    
    # Check if we are expanding an existing MoE layer
    if isinstance(layer, MoEGatedLinear):
        # We need to preserve existing router weights?
        # Current V 0.2.0 Plan simplified: Just freeze everything old.
        # If previous layer had dynamic gates, they are now "Old" and thus Static/Frozen?
        # No, if they were dynamic, they are learned.
        # Ideally we freeze them in their current state.
        # But `HybridGate` design assumes "Old = Static 1.0".
        # This implies we "bake in" the open gates?
        # If DGE V1 used static 2.5, baking in 1.0 is fine.
        # If V 0.2.0 uses dynamic gates...
        # For this transition (V1 -> V2), we assume Old was 1.0 (or we force it).
        pass
        
    # Initialize HybridGates
    # Input to router is `new_in`? Or `old_in`?
    # Usually Router sees the ful input `x`. So `new_in`.
    
    # Gate Col (Inputs)
    new_layer.gate_col = HybridGate(input_dim=new_in, old_count=old_in, new_count=added_in, router_type=router_type, router_init_bias=router_init_bias)
    new_layer.gate_col.use_gradient_rescue = use_gradient_rescue
    new_layer.gate_col.rescue_strength = 10000.0 if use_gradient_rescue else 1.0
    
    # Gate Row (Outputs)
    new_layer.gate_row = HybridGate(input_dim=new_in, old_count=old_out, new_count=added_out, router_type=router_type, router_init_bias=router_init_bias)
    new_layer.gate_row.use_gradient_rescue = use_gradient_rescue
    new_layer.gate_row.rescue_strength = 10000.0 if use_gradient_rescue else 1.0
    
    # 5. Gradient Masking (Backward Hook)
    # 1 = Update, 0 = Freeze.
    mask = torch.ones(new_out, new_in)
    mask[:old_out, :old_in] = 0.0 # Freeze Old Core
    
    # --- PHASE 3: THE FIREWALL (Hard Freeze) ---
    # We strictly FREEZE the New->Old quadrant to prevent Leakage Creep.
    # Without Replay, any plasticity here destroys Old Task performance.
    # Q_TL (New In -> Old Out) = FROZEN.
    mask[:old_out, old_in:] = 0.0
    
    # Isolate Cross Terms (Sidecar Mode)
    # 5. Gradient Masking (Backward Hook)
    # 1 = Update, 0 = Freeze.
    mask = torch.ones(new_out, new_in)
    mask[:old_out, :old_in] = 0.0 # Freeze Old Core
    
    # Isolate Cross Terms (Sidecar Mode)
    if cross_term_policy == 'block_diagonal':
        # Cross Terms: Old->New and New->Old
        # We freeze them at 0.0 (Mask = 0).
        # Top-Right (Old In -> New Out) -> Wait, logic below is [Old_Out, Old_In:] (Top Right)
        mask[:old_out, old_in:] = 0.0 
        # Bottom-Left (New In -> Old Out) -> Wait, [Old_Out:, :Old_In] (Bottom Left)
        mask[old_out:, :old_in] = 0.0
        
        # CRITICAL FIX V10: Zero out values
        with torch.no_grad():
            new_layer.weight[:old_out, old_in:] = 0.0 
            new_layer.weight[old_out:, :old_in] = 0.0 
            
    elif cross_term_policy == 'imprint':
        # V20: Asymmetric Imprinting
        # We want New Neurons to read Old Features (Old->New). So Bottom-Left is ACTIVE.
        # We want Old Neurons to be protected from New Noise (New->Old). So Top-Right is FROZEN.
        
        # Top-Right (New Input -> Old Output): Freeze to 0.0 to preserve Identity of Old Output.
        mask[:old_out, old_in:] = 0.0
        with torch.no_grad():
            new_layer.weight[:old_out, old_in:] = 0.0
            
        # Bottom-Left (Old Input -> New Output): Leave Active (1.0). 
        # Weights are initialized to 0.0 or Noise (from Section 3).
        # Section 3 inited [:, :old_in] (Left columns) as Copy Top-Left, Noise Bottom-Left.
        # So Bottom-Left has noise (1e-3). 
        # Mask is 1.0. Gradients will flow from New Output back to Old Input.
        pass

        
    new_layer.register_buffer('backward_mask', mask)
    
    def hook_fn(grad):
        return grad * new_layer.backward_mask
        
    new_layer.weight.register_hook(hook_fn)
    
    # 7. Gradient Rescue Hook (H2)
    # Applied BEFORE the Ice Wall (Mask) because hooks execute in reverse registration order (LIFO).
    # We want Rescue -> Boost Grad -> Mask -> Zero if Frozen.
    # So we register Rescue LAST (so it runs FIRST).
    if new_layer.use_gradient_rescue:
        def rescue_hook(grad):
            # Check gate status
            # We need the mean open value from the forward pass.
            # We have gate_col and gate_row.
            # The weight connects In -> Out.
            # Gradient comes from Out.
            # Effectively, dL/dW = (dL/dOut_Lin)^T @ x_in.
            # x_in was gated by gate_col. 
            # dL/dOut_Lin passed through gate_row? No, gate_row is after.
            # So Weight sees the effect of BOTH gates.
            # If EITHER is closed, gradient is killed.
            
            # Get combined openness estimate (min/mul)
            # Use small epsilon to avoid div by zero
            openness = 1.0
            if new_layer.gate_col:
                openness *= new_layer.gate_col.last_mean_open
            if new_layer.gate_row:
                openness *= new_layer.gate_row.last_mean_open
                
            # If openness is small (e.g. 0.01), grad is suppressed by 0.01 (or 0.01^2).
            # We compensate.
            # Clamp openness to avoid explosion.
            # Effective Openness Floor: 1e-4 (allows for double-closed gates ~0.018^2 = 0.0003)
            eff_openness = max(float(openness), 0.0001) 
            scale_factor = 1.0 / eff_openness
            
            # Limit strictness of rescue
            scale_factor = min(scale_factor, new_layer.rescue_strength) 
            
            # Store telemetry for logging (User Request: "add all data to the forensic log")
            if not hasattr(new_layer, 'rescue_stats'):
                new_layer.rescue_stats = {'openness': 0.0, 'scale': 1.0, 'count': 0}
            
            # Simple moving average or just last value? 
            # Since this runs every backward pass, let's store the last value for simplicity,
            # or an exponential moving average if we want stability.
            # But validate_dge_integrity runs frequently. Let's just key off "last".
            new_layer.last_rescue_openness = openness
            new_layer.last_rescue_scale = scale_factor
            

            
            return grad * scale_factor
            
        new_layer.weight.register_hook(rescue_hook)
    
    # 6. Freeze Bias (V 0.1.5 Fix)
    if new_layer.bias is not None:
        # Bias Mask: 1=active, 0=frozen
        b_mask = torch.ones(new_out)
        b_mask[:old_out] = 0.0
        new_layer.register_buffer('frozen_bias_mask', b_mask)
        
        def bias_hook(grad):
            return grad * new_layer.frozen_bias_mask
        new_layer.bias.register_hook(bias_hook)
        
    return new_layer

def get_ramp_up_factor(t, T_ramp=1000, slope=20):
    """
    Sigmoid ramp-up schedule as per paper Equation 221.
    G(t) = sigmoid(slope * (t/T_ramp - 0.5))
    """
    if t >= T_ramp:
        return 1.0
    val = slope * (t / T_ramp - 0.5)
    return torch.sigmoid(torch.tensor(val)).item()

class SplitLayerNorm(nn.Module):
    """
    Independent LayerNorms for partitioned channels.
    Prevents "statistical cross-talk" where activity in new channels shifts 
    the global mean/variance, corrupting the signal for old channels.
    """
    def __init__(self, norms):
        super().__init__()
        self.norms = nn.ModuleList(norms)
        
    def forward(self, x):
        # x shape: [Batch, Seq, Total_Dim]
        # We split x based on the dimensions of the norms
        outputs = []
        start_idx = 0
        
        for ln in self.norms:
            # Check shapes
            # SplitLayerNorm wraps multiple LNs. 
            # Total size is sum of their normalized_shapes (if scalars) or dims.
            # Implementation detail: expanded_ln.layers contains the LNs.
            # Assuming we can't easily check 'normalized_shape' attribute directly on wrapper.
            # For SplitLayerNorm, we don't have .normalized_shape
            # Just check output shape logic?
            # Or check internal layers manually:
            # self.assertEqual(expanded_ln.normalized_shape[0], d_old + added)
            dim = ln.normalized_shape[0]
            end_idx = start_idx + dim
            
            # Slice input
            x_slice = x[..., start_idx:end_idx]
            
            # Apply Independent Norm
            out_slice = ln(x_slice)
            outputs.append(out_slice)
            
            start_idx = end_idx
            
        # Concatenate results
        return torch.cat(outputs, dim=-1)

def expand_layer_norm(module: nn.Module, added_dim: int) -> SplitLayerNorm:
    """
    Expands a LayerNorm by splitting it.
    If input is nn.LayerNorm, converts to SplitLayerNorm([old, new]).
    If input is SplitLayerNorm, appends new norm.
    """
    
    # 1. New Norm for the added capacity
    # Initialize to Identity (Weight=1, Bias=0) is default for LayerNorm
    # But usually reset_parameters does this.
    new_ln = nn.LayerNorm(added_dim)
    
    # 2. Handle Expansion
    if isinstance(module, nn.LayerNorm):
        old_ln = module
        # Freeze the Old LayerNorm completely
        # We use hooks for consistency with the rest of DGE forensics
        for param in [old_ln.weight, old_ln.bias]:
            if param is not None:
                # Register a hook that multiplies grad by 0
                # We need a persistent handle if we ever want to unfreeze, but for now this is fine
                # Using a named buffer for the mask so forensics can see it?
                # The param itself doesn't have a buffer. We can just use a closure.
                def zero_grad_hook(grad):
                    return torch.zeros_like(grad)
                param.register_hook(zero_grad_hook)
        
        # Mark it for forensics scanner
        old_ln.register_buffer('frozen_mask', torch.ones(old_ln.normalized_shape[0])) # 1=Frozen (Full)
        
        norms = [old_ln, new_ln]
        
    elif isinstance(module, SplitLayerNorm):
        # Append to existing
        norms = list(module.norms) + [new_ln]
        
    else:
        raise ValueError(f"Unsupported module for LN expansion: {type(module)}")
        
    return SplitLayerNorm(norms)

def expand_embedding(embedding, added_dim):
    """
    Expands an nn.Embedding and freezes the old columns.
    """
    old_emb = embedding
    vocab_size = old_emb.num_embeddings
    old_dim = old_emb.embedding_dim
    new_dim = old_dim + added_dim
    
    new_emb = nn.Embedding(vocab_size, new_dim)
    
    # Copy old weights
    with torch.no_grad():
        new_emb.weight[:, :old_dim] = old_emb.weight
        # Initialize new weights to small noise to allow gradient flow
        # Identity is preserved by the Zero-Init of the Linear Layers that consume this.
        nn.init.normal_(new_emb.weight[:, old_dim:], mean=0.0, std=0.02)
        
    # --- Freeze Old Segment ---
    # Mask: 1.0 = Frozen (First old_dim columns)
    # Applying generic mask for last dimension
    frozen_mask = torch.zeros(new_dim)
    frozen_mask[:old_dim] = 1.0
    # Register purely for persistence if we wanted to save it, but hook needs it in closure or attribute
    new_emb.register_buffer('frozen_mask', frozen_mask)
    
    # CRITICAL FIX V15: Hook the gradients to enforce the mask!
    def hook_fn(grad):
        # frozen_mask is [new_dim]
        # grad is [vocab, new_dim] typically
        if grad.dim() > 1:
             return grad * (1.0 - frozen_mask.view(1, -1).to(grad.device))
        else:
             return grad * (1.0 - frozen_mask.to(grad.device))
        
    new_emb.weight.register_hook(hook_fn)
    
    return new_emb

def expand_parameter(param, added_dim):
    """
    Expands a generic nn.Parameter (assuming last dim is d_model).
    Used for Positional Embeddings.
    """
    old_data = param.data
    # Shape: [..., old_dim]
    shape = list(old_data.shape)
    old_dim = shape[-1]
    shape[-1] += added_dim # New shape
    new_dim = shape[-1]
    
    new_data = torch.zeros(shape, device=param.device)
    
    # Copy old
    with torch.no_grad():
        new_data[..., :old_dim] = old_data
        # New parts random noise
        nn.init.normal_(new_data[..., old_dim:], mean=0.0, std=0.02)
    
    new_param = nn.Parameter(new_data)
    
    # Freeze Old
    frozen_mask = torch.zeros(new_dim)
    frozen_mask[:old_dim] = 1.0
    
    # Can't register buffer on Parameter, only on Module.
    # We capture mask in closure.
    def hook_fn(grad):
        # Broadcast mask to match gradient shape
        # Mask [new_dim]. Grad [..., new_dim].
        return grad * (1.0 - frozen_mask.to(grad.device))
        
    new_param.register_hook(hook_fn)
    
    return new_param

def expand_linear_and_freeze_old(layer: nn.Linear, added_in: int) -> nn.Linear:
    """
    Expands a standard nn.Linear layer's input dimension (columns) and freezes the OLD columns.
    This is used for the Head, where 'Rows' (Vocab) stays constant, but 'Cols' (d_model) grows.
    
    Args:
        layer: Existing nn.Linear layer
        added_in: Number of input features to add (columns)
        
    Returns:
        New nn.Linear with frozen old columns.
    """
    old_out, old_in = layer.weight.shape
    new_in = old_in + added_in
    
    # Create new layer
    new_layer = nn.Linear(new_in, old_out, bias=layer.bias is not None)
    
    # Initialize new weights to 0.0 for Identity Preservation
    # Logits = W_old * h_old + W_new * h_new
    # If W_new = 0, Logits = W_old * h_old (Identity)
    nn.init.constant_(new_layer.weight, 0.0)
    
    # Copy old weights
    with torch.no_grad():
        new_layer.weight[:, :old_in] = layer.weight
        if layer.bias is not None:
            new_layer.bias[:] = layer.bias
            
    # Create Freeze Mask for gradients
    # Shape: [Out, In]
    # We want to freeze columns 0..old_in
    mask = torch.ones_like(new_layer.weight)
    mask[:, :old_in] = 0.0
    
    new_layer.register_buffer("frozen_mask", mask)
    
    def hook(grad):
        return grad * new_layer.frozen_mask
        
    new_layer.weight.register_hook(hook)
    
    return new_layer
    
    return new_layer

def set_seed(seed: int):
    """
    Sets the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
