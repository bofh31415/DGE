import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleGateLinear(nn.Module):
    """
    Double Gate Extension (DGE) Linear Layer.
    
    Implements a linear layer where the effective weight is modulated by independent
    forward gates (row and column wise outer sum), and gradient updates are 
    controlled by a backward mask.
    
    Paper: "Double Gate Extension (DGE): Decoupling Inference and Training for Perfect Continual Learning"
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Main weight matrix: [out_features, in_features]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Forward Gates Parameters
        # Defined as vectors. mask = sigmoid(gate_row + gate_col)
        # Dimensions: gate_row is [out_features, 1], gate_col is [1, in_features]
        self.gate_row = nn.Parameter(torch.Tensor(out_features, 1))
        self.gate_col = nn.Parameter(torch.Tensor(1, in_features))
        
        # Backward Mask (G_bwd)
        # Non-learnable buffer. 1 = trainable, 0 = frozen.
        self.register_buffer('backward_mask', torch.ones(out_features, in_features))
        
        self.reset_parameters()
        
        # Hook handle
        self._hook_handle = None
        self._register_gradient_hook()

    def reset_parameters(self):
        # Initialize weights using Kaiming Uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gates to be open by default (approx 1 after sigmoid)
        # Sigmoid(5) ~= 0.993
        # We split the contribution: 2.5 + 2.5 = 5.0
        nn.init.constant_(self.gate_row, 2.5)
        nn.init.constant_(self.gate_col, 2.5)
        
        # Reset backward mask to all ones
        self.backward_mask.fill_(1.0)

    def _register_gradient_hook(self):
        # Registers a hook on the weight parameter to mask gradients
        if self._hook_handle is not None:
            self._hook_handle.remove()
            
        def hook(grad):
            if grad is None:
                return None
            return grad * self.backward_mask
            
        self._hook_handle = self.weight.register_hook(hook)

    def get_forward_mask(self):
        """Computes G_fwd = sigmoid(gate_row + gate_col)"""
        # Broadcasting handles the outer sum
        return torch.sigmoid(self.gate_row + self.gate_col)

    def forward(self, input):
        # Calculate effective weight
        g_fwd = self.get_forward_mask()
        w_eff = self.weight * g_fwd
        
        return F.linear(input, w_eff, self.bias)

    def set_backward_mask(self, mask):
        """Sets the backward mask. mask should be shape [out_features, in_features]"""
        assert mask.shape == self.weight.shape
        self.backward_mask.copy_(mask)

    def lock_existing_weights(self):
        """Convenience method to set G_bwd to 0 for all current weights."""
        self.backward_mask.fill_(0.0)

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')
