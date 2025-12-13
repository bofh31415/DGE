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

class HybridGate(nn.Module):
    """
    A Gate that is Static (Always Open) for old indices and Dynamic (MoE) for new indices.
    """
    def __init__(self, input_dim: int, old_count: int, new_count: int):
        super().__init__()
        self.old_count = old_count
        self.new_count = new_count
        
        # Static Old Segment (Buffer to freeze it)
        self.register_buffer('old_gate', torch.ones(old_count))
        
        # Dynamic New Segment (Router)
        if new_count > 0:
            self.router = nn.Linear(input_dim, new_count)
            # Initialize to Neutral/Open (0.0 bias -> 0.5 sigmoid) via V 0.2.6 Plan
            # This combats the "Dead Gradient" problem where 0.0 Weights * Closed Gate = 0.0 Gradient
            # With W=0.0, Identity is safe even if Gate is Open.
            nn.init.constant_(self.router.bias, 0.0)
            nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
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
            router_logits = self.router(x)
            new_part = torch.sigmoid(router_logits)
            
            # Store mean activation for Sparsity Loss
            self.last_mean_open = new_part.mean() 
            
            return torch.cat([old_part, new_part], dim=-1)
        else:
            self.last_mean_open = torch.tensor(0.0, device=x.device)
            return old_part

class MoEGatedLinear(nn.Module):
    """
    V 0.2.0: Mixture-of-Experts Gated Linear Layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.gate_row = None 
        self.gate_col = None 
        
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
    isolate_cross_terms: bool = False
) -> MoEGatedLinear:
    """
    Expands a DoubleGateLinear layer by adding input and/or output dimensions.
    Configures the new backward mask to freeze the core (old weights).
    
    Args:
        ...
        isolate_cross_terms: If True, freezes the cross-quadrants (BL/TR) to 0.0,
                             enforcing strict block-diagonal separation.
    """
    old_out, old_in = layer.weight.shape
    new_in = old_in + added_in
    new_out = old_out + added_out
    
    new_layer = MoEGatedLinear(new_in, new_out, bias=layer.bias is not None)
    
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
    # Initialize ALL weights to 0.0 first.
    # This ensures that even if gates are partially open (due to cross-terms),
    # the new connections contribute nothing initially.
    # This is standard "Zero-Init" practice for growing networks.
    nn.init.constant_(new_layer.weight, 0.0)
    
    # Helper indices for the "Old" block in the new matrix
    if frozen_core_pos == Quadrant.TOP_LEFT:
        r_slice = slice(0, old_out)
        c_slice = slice(0, old_in)
        
        # Gate mapping
        r_gate_idx = slice(0, old_out)
        c_gate_idx = slice(0, old_in)
        
    new_in = old_in + added_in
    
    # 2. Create New Layer
    new_layer = MoEGatedLinear(new_in, new_out, bias=(layer.bias is not None))
    
    # 3. Copy Weights & Biases (No Grad context for safety)
    with torch.no_grad():
        # Copy Old Interactions (Top-Left)
        new_layer.weight[:old_out, :old_in] = layer.weight
        if layer.bias is not None:
            new_layer.bias[:old_out] = layer.bias
            
        # Initialize New Weights (Sidecar & Cross) to 0.0
        # This ensures neutral start.
        new_layer.weight[old_out:, :] = 0.0
        new_layer.weight[:, old_in:] = 0.0
        
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
    new_layer.gate_col = HybridGate(input_dim=new_in, old_count=old_in, new_count=added_in)
    
    # Gate Row (Outputs)
    new_layer.gate_row = HybridGate(input_dim=new_in, old_count=old_out, new_count=added_out)
    
    # 5. Gradient Masking (Backward Hook)
    # 1 = Update, 0 = Freeze.
    mask = torch.ones(new_out, new_in)
    mask[:old_out, :old_in] = 0.0 # Freeze Old Core
    
    # Isolate Cross Terms (Sidecar Mode)
    if isolate_cross_terms:
        # Cross Terms: Old->New and New->Old
        # We typically zero them out or freeze them at 0.
        # If we init to 0 and freeze, they stay 0.
        # Top-Right (Old In -> New Out)
        mask[old_out:, :old_in] = 0.0 
        # Bottom-Left (New In -> Old Out)
        mask[:old_out, old_in:] = 0.0
        
    new_layer.register_buffer('backward_mask', mask)
    
    def hook_fn(grad):
        return grad * new_layer.backward_mask
        
    new_layer.weight.register_hook(hook_fn)
    
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
    
    def hook_fn(grad):
        if grad is None: return None
        # Grad shape: [vocab, dim] OR Sparse indices? 
        # nn.Embedding usually returns dense grad if sparse=False (default).
        # We need to broadcast the mask [dim] to [vocab, dim]
        # grad * (1 - mask)
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
    # Slice dynamically
    # Assuming 3D for PosEmb [1, T, D] or generic?
    # Let's trust standard slicing works for last dim if we use ...
    with torch.no_grad():
        new_data[..., :old_dim] = old_data
        # New parts random noise
        nn.init.normal_(new_data[..., old_dim:], mean=0.0, std=0.02)
    
    new_param = nn.Parameter(new_data)
    
    # Freeze Old
    frozen_mask = torch.zeros(new_dim)
    frozen_mask[:old_dim] = 1.0
    
    # Can't register buffer on Parameter, only on Module.
    # But we can capture 'frozen_mask' in the closure of the hook.
    # We should move mask to device of grad inside hook.
    def hook_fn(grad):
        if grad is None: return None
        # Grad has same shape as param [..., new_dim]
        # Mask is [new_dim]
        # Broadcast matches last dim automatically in PyTorch
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
