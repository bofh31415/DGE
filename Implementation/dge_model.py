import torch
import torch.nn as nn
from dge_utils import expand_dge_linear, expand_layer_norm, expand_embedding, expand_parameter, expand_linear_and_freeze_old, MoEGatedLinear, HybridGate, Quadrant

class DGEBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Simple Attention projections
        # For simplicity in this lab, we use one large linear for QKV
        # and one for Output.
        self.w_qkv = MoEGatedLinear(d_model, 3 * d_model)
        self.w_o = MoEGatedLinear(d_model, d_model)
        
        # MLP
        # d_model -> 4*d_model -> d_model
        self.w_mlp_in = MoEGatedLinear(d_model, 4 * d_model)
        self.w_mlp_out = MoEGatedLinear(4 * d_model, d_model)
        
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
        self.lm_head = MoEGatedLinear(d_model, vocab_size, bias=False)
        
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
            
        sparsity_penalty = 0.0
        # Collect Gate Sparsity Loss
        for module in self.modules():
            if isinstance(module, MoEGatedLinear):
                # Add row gate sparsity (Output side)
                if module.gate_row is not None and hasattr(module.gate_row, 'last_mean_open'):
                    sparsity_penalty += module.gate_row.last_mean_open
                # Add col gate sparsity (Input side)
                if module.gate_col is not None and hasattr(module.gate_col, 'last_mean_open'):
                    sparsity_penalty += module.gate_col.last_mean_open
        
        # Lambda for sparsity: 0.05 (Reverted to baseline for V 0.2.3, relying on Low LR)
        total_loss = loss + (0.05 * sparsity_penalty) if loss is not None else (0.05 * sparsity_penalty)
        
        return logits, total_loss

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
        Calculates detailed forensic metrics for DGE validity.
        """
        metrics = {
            'frozen_grad_norm': 0.0, # Aggregate
            'active_grad_norm': 0.0,
            'frozen_weight_grad': 0.0,
            'frozen_bias_grad': 0.0,
            'frozen_head_grad': 0.0,
            'gate_grad_norm': 0.0,
            'max_gate_mag': 0.0
        }
        
        frozen_sq_sum = 0.0
        active_sq_sum = 0.0
        
        frozen_weight_sq = 0.0
        frozen_bias_sq = 0.0
        frozen_head_sq = 0.0
        gate_grad_sq = 0.0
        max_gate_val = 0.0
        
        for name, module in self.named_modules():
            # --- MoEGatedLinear Checks ---
            if isinstance(module, MoEGatedLinear):
                # Max Gate Magnitude (Forward check) - Now via HybridGate
                with torch.no_grad():
                    # In MoE, gates are HybridGate modules. Check router bias.
                    if module.gate_row is not None and hasattr(module.gate_row, 'router'):
                        if module.gate_row.router is not None:
                            max_gate_val = max(max_gate_val, module.gate_row.router.bias.abs().max().item())
                    if module.gate_col is not None and hasattr(module.gate_col, 'router'):
                        if module.gate_col.router is not None:
                            max_gate_val = max(max_gate_val, module.gate_col.router.bias.abs().max().item())

                if module.weight.grad is None:
                    continue
                    
                # 1. Weights
                # Mask: 0 = Frozen, 1 = Active
                frozen_mask = (module.backward_mask == 0)
                
                # Grads on frozen weights
                frozen_grads = module.weight.grad * frozen_mask
                if frozen_grads.numel() > 0:
                    norm = frozen_grads.norm().item()
                    frozen_weight_sq += norm ** 2
                    frozen_sq_sum += norm ** 2
                
                # Grads on active weights
                active_grads = module.weight.grad * module.backward_mask
                active_sq_sum += active_grads.norm().item() ** 2

                # 2. Gates (HybridGate)
                # Router weights are trainable; old_gate is buffer (no grad).
                if module.gate_row is not None and hasattr(module.gate_row, 'router'):
                    if module.gate_row.router is not None and module.gate_row.router.weight.grad is not None:
                        gate_grad_sq += module.gate_row.router.weight.grad.norm().item() ** 2
                        
                if module.gate_col is not None and hasattr(module.gate_col, 'router'):
                    if module.gate_col.router is not None and module.gate_col.router.weight.grad is not None:
                        gate_grad_sq += module.gate_col.router.weight.grad.norm().item() ** 2
                    
                # 3. Bias (Critical V 0.1.5 Check)
                if module.bias is not None and module.bias.grad is not None:
                    if hasattr(module, 'frozen_bias_mask'):
                        # frozen_bias_mask: 0.0 = Frozen, 1.0 = Active (Wait, I used 0 for mask in utils)
                        # Let's verify utils:
                        # bias_mask = torch.ones(new_out)
                        # bias_mask[0:old_out] = 0.0
                        # hook: grad * mask
                        # So frozen area is where mask == 0.0
                        
                        # So we check grad * (1.0 - mask)
                        frozen_b = module.bias.grad * (1.0 - module.frozen_bias_mask)
                        norm = frozen_b.norm().item()
                        frozen_bias_sq += norm ** 2
                        frozen_sq_sum += norm ** 2
            
            # --- MoEGatedLinear Checks ---
            elif isinstance(module, MoEGatedLinear):
                # 1. Weights
                # Check Frozen Weights (Old Core) -> Should correspond to backward_mask == 0
                if module.weight.grad is not None:
                    # frozen = grad * (1 - mask)
                    frozen_grads = module.weight.grad * (1 - module.backward_mask)
                    norm = frozen_grads.norm().item()
                    frozen_weight_sq += norm ** 2
                    frozen_sq_sum += norm ** 2
                
                    # Grads on active weights
                    active_grads = module.weight.grad * module.backward_mask
                    active_sq_sum += active_grads.norm().item() ** 2

                # 2. Gates (HybridGate Analysis)
                # Gate Row
                if module.gate_row is not None and hasattr(module.gate_row, 'router'):
                    if module.gate_row.router is not None:
                        # Dynamic Router Grads -> Active
                        if module.gate_row.router.weight.grad is not None:
                            gate_grad_sq += module.gate_row.router.weight.grad.norm().item() ** 2
                        
                        # Max Magnitude check (on logits or sigmoid?)
                        # Checking logits bias to see if staying closed
                        max_gate_val = max(max_gate_val, module.gate_row.router.bias.max().item())
                        
                # Gate Col
                if module.gate_col is not None and hasattr(module.gate_col, 'router'):
                    if module.gate_col.router is not None:
                        if module.gate_col.router.weight.grad is not None:
                            gate_grad_sq += module.gate_col.router.weight.grad.norm().item() ** 2
                            
                # Note: We can't easily check "Frozen Gate" leakage because 'old_gate' is a buffer, not a param.
                # It has no grad. So it's "Perfectly Frozen" by definition of PyTorch.
                # The only risk is if we accidentally made it a parameter. 
                # dge_utils uses register_buffer. So it is safe.
                    
                # 3. Bias (Critical V 0.1.5 Check)
                if module.bias is not None and module.bias.grad is not None:
                    if hasattr(module, 'frozen_bias_mask'):
                        # frozen_bias_mask: 0.0 = Frozen, 1.0 = Active (Wait, I used 0 for mask in utils)
                        # Let's verify utils:
                        # bias_mask = torch.ones(new_out)
                        # bias_mask[0:old_out] = 0.0
                        # hook: grad * mask
                        # So frozen area is where mask == 0.0
                        
                        # So we check grad * (1.0 - mask)
                        frozen_b = module.bias.grad * (1.0 - module.frozen_bias_mask)
                        norm = frozen_b.norm().item()
                        frozen_bias_sq += norm ** 2
                        frozen_sq_sum += norm ** 2
            
            # --- LayerNorm Checks ---
            elif isinstance(module, nn.LayerNorm):
                # Standard LayerNorms might be expanded.
                # In expand_layer_norm: frozen_mask has 1.0 for FROZEN.
                # hook returns grad * (1.0 - frozen_mask).
                # So we verify if grad exists where frozen_mask == 1.0
                if hasattr(module, 'frozen_mask'):
                     if module.weight.grad is not None:
                         # frozen_mask is 1 where frozen.
                         frozen_w = module.weight.grad * module.frozen_mask
                         norm = frozen_w.norm().item()
                         frozen_weight_sq += norm ** 2
                         frozen_sq_sum += norm ** 2
                         
                     if module.bias.grad is not None:
                         frozen_b = module.bias.grad * module.frozen_mask
                         norm = frozen_b.norm().item()
                         frozen_bias_sq += norm ** 2
                         frozen_sq_sum += norm ** 2
            
            # --- Head Checks (nn.Linear) ---
            elif isinstance(module, nn.Linear):
                # Check for expanded head with frozen columns
                # In expand_linear_and_freeze_old:
                # frozen_mask (buffer): 0.0 = Frozen cols, 1.0 = Active cols.
                if hasattr(module, 'frozen_mask'):
                    if module.weight.grad is not None:
                        # Frozen where mask == 0
                        frozen_h = module.weight.grad * (1.0 - module.frozen_mask)
                        norm = frozen_h.norm().item()
                        frozen_head_sq += norm ** 2
                        frozen_sq_sum += norm ** 2
                        
                        # Active norm for head
                        active_h = module.weight.grad * module.frozen_mask
                        active_sq_sum += active_h.norm().item() ** 2

        metrics['frozen_grad_norm'] = frozen_sq_sum ** 0.5
        metrics['active_grad_norm'] = active_sq_sum ** 0.5
        
        metrics['frozen_weight_grad'] = frozen_weight_sq ** 0.5
        metrics['frozen_bias_grad'] = frozen_bias_sq ** 0.5
        metrics['frozen_head_grad'] = frozen_head_sq ** 0.5
        
        metrics['gate_grad_norm'] = gate_grad_sq ** 0.5
        metrics['max_gate_mag'] = max_gate_val
        
        return metrics
