import torch
import torch.nn as nn
from enum import Enum
from dge_linear import DoubleGateLinear

class Quadrant(Enum):
    TOP_LEFT = "TL"
    TOP_RIGHT = "TR"
    BOTTOM_LEFT = "BL"
    BOTTOM_RIGHT = "BR"

def expand_dge_linear(
    layer: DoubleGateLinear, 
    added_in: int, 
    added_out: int, 
    frozen_core_pos: Quadrant = Quadrant.TOP_LEFT,
    isolate_cross_terms: bool = False
) -> DoubleGateLinear:
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
    
    new_layer = DoubleGateLinear(new_in, new_out, bias=layer.bias is not None)
    
    # Defaults for initialization of new areas
    # Initialize new gates to be OPEN to allow gradient flow to the new zero-weights.
    # We rely SOLELY on W=0.0 for Identity Preservation.
    # If Gates are closed (-5.0), gradients vanish (Double-Lock).
    # Set init to 2.5 (Sigmoid(5) ~ 0.99)
    nn.init.constant_(new_layer.gate_row, 2.5)
    nn.init.constant_(new_layer.gate_col, 2.5)
    
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
        
    elif frozen_core_pos == Quadrant.TOP_RIGHT:
        # Old core is top-right: rows 0..old_out, cols added_in..new_in
        r_slice = slice(0, old_out)
        c_slice = slice(added_in, new_in)
        
        r_gate_idx = slice(0, old_out)
        c_gate_idx = slice(added_in, new_in)

    elif frozen_core_pos == Quadrant.BOTTOM_LEFT:
        # Old core is bottom-left: rows added_out..new_out, cols 0..old_in
        r_slice = slice(added_out, new_out)
        c_slice = slice(0, old_in)
        
        r_gate_idx = slice(added_out, new_out)
        c_gate_idx = slice(0, old_in)
        
    elif frozen_core_pos == Quadrant.BOTTOM_RIGHT:
        # Old core is bottom-right
        r_slice = slice(added_out, new_out)
        c_slice = slice(added_in, new_in)
        
        r_gate_idx = slice(added_out, new_out)
        c_gate_idx = slice(added_in, new_in)
        
    else:
        raise ValueError(f"Unsupported quadrant: {frozen_core_pos}")

    # --- Copy Weights ---
    with torch.no_grad():
        new_layer.weight[r_slice, c_slice] = layer.weight
        
        # --- Copy Bias ---
        if layer.bias is not None:
             # If old core is at top (TL or TR), logic is simple? 
             # Wait, bias depends only on Output dimension (Rows).
             # If expansion adds rows, we need to know where the old rows went.
             # TL/TR -> Old rows are at top [0:old_out]
             # BL/BR -> Old rows are at bottom [added_out:new_out]
             
             if frozen_core_pos in [Quadrant.TOP_LEFT, Quadrant.TOP_RIGHT]:
                 bias_slice = slice(0, old_out)
             else:
                 bias_slice = slice(added_out, new_out)
                 
             new_layer.bias[bias_slice] = layer.bias

        # --- Copy Gates ---
        # Gate Row corresponds to Output Rows
        # Gate Col corresponds to Input Cols
        if frozen_core_pos in [Quadrant.TOP_LEFT, Quadrant.TOP_RIGHT]:
             new_layer.gate_row[0:old_out] = layer.gate_row
        else:
             new_layer.gate_row[added_out:new_out] = layer.gate_row
             
        if frozen_core_pos in [Quadrant.TOP_LEFT, Quadrant.BOTTOM_LEFT]:
             new_layer.gate_col[:, 0:old_in] = layer.gate_col
        else:
             new_layer.gate_col[:, added_in:new_in] = layer.gate_col
             
        # --- Freeze Core ---
        # Set backward mask to 0 for the old core region
        new_layer.backward_mask[r_slice, c_slice] = 0.0
        
        if frozen_core_pos == Quadrant.TOP_LEFT:
            # Cross terms: TR (rows 0:old, cols old:new) and BL (rows old:new, cols 0:old)
            if isolate_cross_terms:
                 new_layer.backward_mask[0:old_out, old_in:new_in] = 0.0 # TR
                 new_layer.backward_mask[old_out:new_out, 0:old_in] = 0.0 # BL
                 
    return new_layerk.
        
        # Freeze Gate Row segments
        if frozen_core_pos in [Quadrant.TOP_LEFT, Quadrant.TOP_RIGHT]:
             # Old Out Rows are at top [0:old_out]
             new_layer.gate_row_mask[0:old_out] = 0.0
        else:
             # Old Out Rows are at bottom [added_out:new_out]
             new_layer.gate_row_mask[added_out:new_out] = 0.0
             
        # Freeze Gate Col segments
        if frozen_core_pos in [Quadrant.TOP_LEFT, Quadrant.BOTTOM_LEFT]:
             # Old In Cols are at left [0:old_in]
             new_layer.gate_col_mask[:, 0:old_in] = 0.0
        else:
             # Old In Cols are at right [added_in:new_in]
             new_layer.gate_col_mask[:, added_in:new_in] = 0.0
        
        # IMPORTANT: The paper says "Old knowledge ($G_{fwd}$ open) while protecting it ($G_{bwd}$ closed)".
        # So we ensure the GATES for the old region are preserved (Open).
        # We did this by copying gate_row/gate_col.
        # But we must ensure the new parts of the gate vectors (which form the cross product for the new quadrants)
        # start closed. This was done by the init to -5.0.
        
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

def expand_layer_norm(ln: nn.LayerNorm, added_dim: int) -> nn.LayerNorm:
    """
    Expands a LayerNorm module, preserving learned affine parameters (weight/bias).
    """
    old_dim = ln.normalized_shape[0]
    new_dim = old_dim + added_dim
    
    new_ln = nn.LayerNorm(new_dim, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
    
    if ln.elementwise_affine:
        with torch.no_grad():
            # Copy old params
            new_ln.weight[:old_dim] = ln.weight
            new_ln.bias[:old_dim] = ln.bias
            
            # Initialize new params
            # Weight -> 1.0 (Identity scaling for new features)
            new_ln.weight[old_dim:] = 1.0 
            # Bias -> 0.0
            new_ln.bias[old_dim:] = 0.0
            
            # --- Freeze Old Segment ---
            # Register a backward mask buffer: 0.0 = Frozen, 1.0 = Active
            mask = torch.ones(new_dim)
            mask[:old_dim] = 0.0
            new_ln.register_buffer('frozen_mask', 1.0 - mask) # Storing 'frozen_mask' where 1=Frozen to match my validation logic? 
            # Wait, let's stick to the validation logic I just wrote: frozen_w = grad * frozen_mask.
            # So if frozen_mask is 1, it counts as leakage.
            # So frozen_mask should be 1 where we want to FREEZE (indices < old_dim).
            # And the hook should multiply by (1 - frozen_mask).
            
            # 1. Store Mask (1=Frozen, 0=Active)
            frozen_mask = torch.zeros(new_dim)
            frozen_mask[:old_dim] = 1.0
            new_ln.register_buffer('frozen_mask', frozen_mask)
            
            # 2. Register Hooks
            def hook_fn(grad):
                if grad is None: return None
                return grad * (1.0 - frozen_mask.to(grad.device))
                
            new_ln.weight.register_hook(hook_fn)
            new_ln.bias.register_hook(hook_fn)
            
    return new_ln

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
