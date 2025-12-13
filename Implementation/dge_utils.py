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
    frozen_core_pos: Quadrant = Quadrant.TOP_LEFT
) -> DoubleGateLinear:
    """
    Expands a DoubleGateLinear layer by adding input and/or output dimensions.
    Configures the new backward mask to freeze the core (old weights) according 
    to frozen_core_pos, while keeping new areas trainable.
    
    Args:
        layer: Existing DoubleGateLinear layer
        added_in: Number of input features to add
        added_out: Number of output features to add
        frozen_core_pos: Position of the old frozen weights in the new matrix
        
    Returns:
        New expanded DoubleGateLinear layer
    """
    old_out, old_in = layer.weight.shape
    new_in = old_in + added_in
    new_out = old_out + added_out
    
    new_layer = DoubleGateLinear(new_in, new_out, bias=layer.bias is not None)
    
    # Defaults for initialization of new areas
    # Initialize new gates significantly close to decrease visibility initially
    # Sigmoid(-5) ~= 0.006. 
    # We want G_fwd(new) ~ 0. So row+col should be << 0.
    # Set init to -5.0. -5 + -5 = -10 => Sigmoid(-10) is very small.
    nn.init.constant_(new_layer.gate_row, -5.0)
    nn.init.constant_(new_layer.gate_col, -5.0)
    
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
