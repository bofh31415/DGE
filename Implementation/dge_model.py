import torch
import torch.nn as nn
from dge_linear import DoubleGateLinear
from dge_utils import expand_dge_linear, expand_layer_norm, expand_embedding, expand_parameter, Quadrant

class DGEBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Simple Attention projections
        # For simplicity in this lab, we use one large linear for QKV
        # and one for Output.
        self.w_qkv = DoubleGateLinear(d_model, 3 * d_model)
        self.w_o = DoubleGateLinear(d_model, d_model)
        
        # MLP
        # d_model -> 4*d_model -> d_model
        self.w_mlp_in = DoubleGateLinear(d_model, 4 * d_model)
        self.w_mlp_out = DoubleGateLinear(4 * d_model, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [batch, seq, d_model]
        
        # Attn
        resid = x
        x = self.ln1(x)
        B, T, C = x.size()
        
        # QKV
        qkv = self.w_qkv(x) # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled Dot Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = torch.softmax(att, dim=-1)
        y = att @ v # [B, n_head, T, head_dim]
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.w_o(y)
        x = resid + y
        
        # MLP
        resid = x
        x = self.ln2(x)
        x = self.w_mlp_in(x)
        x = self.act(x)
        x = self.w_mlp_out(x)
        x = resid + x
        
        return x

    def expand(self, added_width, new_n_head, quadrant):
        """
        Expand the block width by added_width. Usually d_model -> d_model + added_width.
        """
        # Expand QKV: In +added, Out +3*added
        # Note: Q,K,V are interleaved. DoubleGateLinear is unaware of this structure.
        # This is a simplification. For rigorous RoPE preservation, we'd need structured expansion.
        # For the Lab, we will accept a simple expansion where we trust the linear layer resizing.
        # But wait - if we just expand 3*d_model output, it might mix Q/K/V quadrants messy.
        # For a proper LAB demo of DGE mechanics, standard linear expansion is fine to show GATING.
        # We will keep it simple.
        
        # V 0.1.3: Enable Strict Isolation (Sidecar) to prevent Interference
        iso = True
        
        self.w_qkv = expand_dge_linear(self.w_qkv, added_in=added_width, added_out=3*added_width, frozen_core_pos=quadrant, isolate_cross_terms=iso)
        self.w_o = expand_dge_linear(self.w_o, added_in=added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=iso) 
        
        self.w_mlp_in = expand_dge_linear(self.w_mlp_in, added_in=added_width, added_out=4*added_width, frozen_core_pos=quadrant, isolate_cross_terms=iso)
        self.w_mlp_out = expand_dge_linear(self.w_mlp_out, added_in=4*added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=iso)
        
        self.d_model += added_width
        self.n_head = new_n_head # Update head count
        
        # CRITICAL FIX: Preserve LayerNorm statistics!
        self.ln1 = expand_layer_norm(self.ln1, added_width)
        self.ln2 = expand_layer_norm(self.ln2, added_width)

class DGESimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_layer=2, n_head=4):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 128, d_model)) # Fixed max seq len
        self.layers = nn.ModuleList([DGEBlock(d_model, n_head) for _ in range(n_layer)])
        self.lm_head = DoubleGateLinear(d_model, vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :] # Broadcasting handled
        x = tok_emb + pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def expand_model(self, added_d_model, quadrant=Quadrant.TOP_LEFT):
        """Expands the entire model width."""
        current_head_dim = self.d_model // self.layers[0].n_head
        new_d_model = self.d_model + added_d_model
        new_n_head = new_d_model // current_head_dim
        
        print(f"Expanding Model: {self.d_model} -> {new_d_model} (Heads: {self.layers[0].n_head} -> {new_n_head})")
        
        # Expand Embedding
        self.token_emb = expand_embedding(self.token_emb, added_d_model)

        # Resize Pos Emb
        self.pos_emb = expand_parameter(self.pos_emb, added_d_model)

        # Expand Layers
        for i, layer in enumerate(self.layers):
            print(f"Expanding Layer {i}...")
            layer.expand(added_d_model, new_n_head, quadrant)
            
        # Expand Head
        self.lm_head = expand_dge_linear(self.lm_head, added_in=added_d_model, added_out=0, frozen_core_pos=quadrant)
        
        self.d_model = new_d_model
        print("Expansion Complete.")

    def validate_dge_integrity(self):
        """
        Calculates forensic metrics for DGE validity.
        Specifically checks if any gradients exist on 'frozen' weights.
        """
        frozen_sq_sum = 0.0
        active_sq_sum = 0.0
        
        for name, module in self.named_modules():
            if isinstance(module, DoubleGateLinear):
                if module.weight.grad is None:
                    continue
                    
                # Mask: 0 = Frozen, 1 = Active
                # Invert mask for frozen
                frozen_mask = (module.backward_mask == 0)
                
                # Grads on frozen weights
                frozen_grads = module.weight.grad * frozen_mask
                frozen_sq_sum += frozen_grads.norm().item() ** 2
                
                # Grads on active weights
                active_grads = module.weight.grad * module.backward_mask
                active_sq_sum += active_grads.norm().item() ** 2

                # Gate Grads
                # Check Row Gate drift
                if module.gate_row.grad is not None:
                    frozen_gate_row = module.gate_row.grad * (1 - module.gate_row_mask)
                    frozen_sq_sum += frozen_gate_row.norm().item() ** 2

                # Check Col Gate drift
                if module.gate_col.grad is not None:
                    frozen_gate_col = module.gate_col.grad * (1 - module.gate_col_mask)
                    frozen_sq_sum += frozen_gate_col.norm().item() ** 2

            elif isinstance(module, nn.LayerNorm):
                # Check for frozen LayerNorm segments
                # We expect expand_layer_norm to register a 'frozen_mask' buffer if partial freezing is active
                if hasattr(module, 'frozen_mask'):
                     if module.weight.grad is not None:
                         frozen_w = module.weight.grad * module.frozen_mask
                         frozen_sq_sum += frozen_w.norm().item() ** 2
                     if module.bias.grad is not None:
                         frozen_b = module.bias.grad * module.frozen_mask
                         frozen_sq_sum += frozen_b.norm().item() ** 2
                
        return {
            'frozen_grad_norm': frozen_sq_sum ** 0.5,
            'active_grad_norm': active_sq_sum ** 0.5,
            # We could add specific gate metrics if needed, but bundling into frozen_grad_norm 
            # is enough to trigger the "Wait, something frozen is moving" alarm.
        }
