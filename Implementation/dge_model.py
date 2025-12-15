import torch
import torch.nn as nn
from dge_utils import expand_dge_linear, expand_layer_norm, expand_embedding, expand_parameter, expand_linear_and_freeze_old, MoEGatedLinear, HybridGate, Quadrant

class DGEBlock(nn.Module):
    def __init__(self, d_model, n_head, router_type='linear'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.router_type = router_type # Store router_type for potential future use or consistency
        
        # Simple Attention projections
        # V12: Separate Q, K, V layers to fix "Structural Shuffle" bug during expansion
        self.w_q = MoEGatedLinear(d_model, d_model)
        self.w_k = MoEGatedLinear(d_model, d_model)
        self.w_v = MoEGatedLinear(d_model, d_model)
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
        # V12: Separate Forward
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
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

    def expand(self, added_width, new_n_head, quadrant, router_type='linear', use_gradient_rescue=True, use_orthogonal_init=False, isolate_cross_terms=False, cross_term_policy='full', router_init_bias=-4.0, gating_threshold: float = 0.0):
        """
        Expand the block width by added_width. Usually d_model -> d_model + added_width.
        """
        # Expand QKV: In +added, Out +added (each)
        # V12: Separate Expansion
        self.w_q = expand_dge_linear(self.w_q, added_in=added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        self.w_k = expand_dge_linear(self.w_k, added_in=added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        self.w_v = expand_dge_linear(self.w_v, added_in=added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        
        self.w_o = expand_dge_linear(self.w_o, added_in=added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold) 
        
        self.w_mlp_in = expand_dge_linear(self.w_mlp_in, added_in=added_width, added_out=4*added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        self.w_mlp_out = expand_dge_linear(self.w_mlp_out, added_in=4*added_width, added_out=added_width, frozen_core_pos=quadrant, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_type=router_type, use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, router_init_bias=router_init_bias, gating_threshold=gating_threshold)
        
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
        
    def forward(self, idx, targets=None, sparsity_lambda=0.05):
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
        
        # Lambda for sparsity
        total_loss = loss + (sparsity_lambda * sparsity_penalty) if loss is not None else (sparsity_lambda * sparsity_penalty)
        
        return logits, total_loss

    def expand_model(self, new_input_dim, new_output_dim, router_type='linear', use_gradient_rescue=True, use_orthogonal_init=False, isolate_cross_terms=False, cross_term_policy='full', router_init_bias=-4.0, gating_threshold: float = 0.0):
        """
        Expands the model capacity using DGE.
        Returns a new DGEModel instance (or mutates self).
        """
        added_d_model = new_input_dim - self.d_model
        # The instruction provided a malformed print statement and incomplete expansion logic.
        # I will attempt to reconstruct the intent based on the new signature and the original expand_model.
        # Assuming new_input_dim is the new d_model, and new_output_dim is not directly used here
        # but might imply a change in vocab_size for the lm_head.
        # The original `expand_model` used `added_d_model` and `quadrant`.
        # The new signature uses `new_input_dim`, `new_output_dim`, `router_type`.
        # This implies a significant refactor of the expansion logic.

        # Given the instruction, it seems to be a partial update.
        # I will try to integrate the new signature and the provided lines,
        # while keeping the original logic as much as possible,
        # and fixing the obvious syntax errors in the provided snippet.

        # Original logic:
        # current_head_dim = self.d_model // self.layers[0].n_head
        # new_d_model = self.d_model + added_d_model
        # new_n_head = new_d_model // current_head_dim

        # The instruction's print statement:
        # print(f"Expanding DGE Model: {self.input_dim}x{self.output_dim} -> {new_input_dim}x{new_output_dim} (Heads: {self.layers[0].n_head} -> {new_n_head})")
        # This is syntactically incorrect and refers to non-existent attributes (self.input_dim, self.output_dim).
        # I will use the new_input_dim as the new d_model and infer added_d_model.
        
        if added_d_model < 0:
            raise ValueError("New input dimension must be greater than or equal to current d_model for expansion.")

        current_head_dim = self.d_model // self.layers[0].n_head
        new_n_head = new_input_dim // current_head_dim # Assuming head_dim remains constant
        
        print(f"Expanding DGE Model: d_model {self.d_model} -> {new_input_dim} (Heads: {self.layers[0].n_head} -> {new_n_head}) (Router: {router_type}) [Rescue: {use_gradient_rescue}, Ortho: {use_orthogonal_init}, BlockDiag: {isolate_cross_terms}, Policy: {cross_term_policy}, Threshold: {gating_threshold}]")
        
        # Expand Embedding
        self.token_emb = expand_embedding(self.token_emb, added_d_model)

        # Resize Pos Emb
        self.pos_emb = expand_parameter(self.pos_emb, added_d_model)

        # Expand Layers
        for i, layer in enumerate(self.layers):
            print(f"Expanding Layer {i}...")
            # DGEBlock.expand needs update too?
            # It calls expand_dge_linear internally.
            # We need to update DGEBlock.expand signature as well.
            layer.expand(added_d_model, new_n_head, quadrant=Quadrant.TOP_LEFT, router_type=router_type, 
                         use_gradient_rescue=use_gradient_rescue, use_orthogonal_init=use_orthogonal_init, isolate_cross_terms=isolate_cross_terms, cross_term_policy=cross_term_policy, router_init_bias=router_init_bias, gating_threshold=gating_threshold) 

        # Expand Head
        # The instruction snippet had `ge_linear(self.lm_head, added_in=added_d_model, added_out=0, frozen_core_pos=quadrant)`
        # which is a partial line. I will reconstruct it based on the original and new parameters.
        # The original `lm_head` expansion used `added_d_model` for `added_in` and `quadrant`.
        # `new_output_dim` could be used for `added_out` if vocab_size is changing.
        # For now, I'll assume `added_out` is 0 as in the original, unless `new_output_dim` implies a change.
        # If `new_output_dim` is meant to change the vocab_size, then `added_out` should be `new_output_dim - self.lm_head.out_features`.
        # Given the instruction, it's ambiguous. I'll stick to the original `added_out=0` for `lm_head` for now,
        # as the primary expansion seems to be `d_model`.
        # Expand LM Head
        self.lm_head = expand_dge_linear(
            self.lm_head, 
            added_in=added_d_model, 
            added_out=0, 
            frozen_core_pos=Quadrant.TOP_LEFT, 
            isolate_cross_terms=isolate_cross_terms,
            cross_term_policy='full', # CRITICAL: Head MUST see new features!
            router_type=router_type,
            use_gradient_rescue=use_gradient_rescue,
            use_orthogonal_init=use_orthogonal_init,
            router_init_bias=router_init_bias,
            gating_threshold=gating_threshold
        )
        
        self.d_model = new_input_dim # Update d_model to the new value
        print("Expansion Complete.")

    def validate_dge_integrity(self):
        """
        Calculates detailed forensic metrics for DGE validity.
        """
        frozen_sq_sum = 0.0
        active_sq_sum = 0.0
        
        frozen_weight_sq = 0.0
        frozen_bias_sq = 0.0
        frozen_head_sq = 0.0
        gate_grad_sq = 0.0
        
        # Aggregators for detailed logs
        max_gate_val = 0.0 # Standard ABS max
        
        # New Forensic Stats
        gate_bias_vals = []
        router_weight_norms = []
        
        # Rescue Telemetry (H2)
        rescue_openness_vals = []
        rescue_scale_vals = []
        

        for name, module in self.named_modules():

            # --- MoEGatedLinear Checks ---
            # --- MoEGatedLinear Checks ---
            # Robust check for MoEGatedLinear
            is_moe = isinstance(module, MoEGatedLinear) 
            if not is_moe and module.__class__.__name__ == 'MoEGatedLinear':
                 is_moe = True

            
            if is_moe:

                
                # Max Gate Magnitude (Forward check) - Now via HybridGate
                with torch.no_grad():
                    # In MoE, gates are HybridGate modules. Check router bias.
                    # Helper to collect bias
                    def collect_router_stats(gate_module):
                        if gate_module is not None and hasattr(gate_module, 'router') and gate_module.router is not None:
                            # Handle MLP (Sequential) or Linear
                            router_layer = gate_module.router
                            if isinstance(router_layer, nn.Sequential):
                                # For stats, we care about the FINAL decision layer
                                router_layer = router_layer[-1]
                            
                            if hasattr(router_layer, 'bias') and router_layer.bias is not None:
                                bias = router_layer.bias
                                # Max Mag
                                nonlocal max_gate_val
                                max_gate_val = max(max_gate_val, bias.abs().max().item())
                                
                                # Signed Stats
                                gate_bias_vals.extend(bias.flatten().tolist())
                                
                            if hasattr(router_layer, 'weight') and router_layer.weight is not None:
                                # Weight Norm
                                router_weight_norms.append(router_layer.weight.norm().item())

                    collect_router_stats(module.gate_row)
                    collect_router_stats(module.gate_col)

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
                def check_router_grad(gate_module):
                    grad_sq = 0.0
                    if gate_module is not None and hasattr(gate_module, 'router') and gate_module.router is not None:
                         # Handle MLP (Sequential) or Linear
                        router_layer = gate_module.router
                        if isinstance(router_layer, nn.Sequential):
                            # Check ALL layers or just the last?
                            # Usually we want aggregated gradient norm for the whole router.
                            # Summing sq norms of all trainable params in router.
                            for param in router_layer.parameters():
                                if param.grad is not None:
                                    grad_sq += param.grad.norm().item() ** 2
                        else:
                            # Linear case
                            if hasattr(router_layer, 'weight') and router_layer.weight.grad is not None:
                                grad_sq += router_layer.weight.grad.norm().item() ** 2
                            if hasattr(router_layer, 'bias') and router_layer.bias is not None and router_layer.bias.grad is not None:
                                grad_sq += router_layer.bias.grad.norm().item() ** 2
                    return grad_sq

                gate_grad_sq += check_router_grad(module.gate_row)
                gate_grad_sq += check_router_grad(module.gate_col)
                    
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
            
                # --- Moved Rescue Logic Here ---
                # 2. Rescue Metrics (H2 Telemetry)
                if hasattr(module, 'last_rescue_openness'):
                    val = module.last_rescue_openness
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    rescue_openness_vals.append(val)
                
                if hasattr(module, 'last_rescue_scale'):
                    val = module.last_rescue_scale
                    if isinstance(val, torch.Tensor):
                         val = val.item()
                    rescue_scale_vals.append(val)
            

                            
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

        frozen_grad_norm = frozen_sq_sum ** 0.5
        active_grad_norm = active_sq_sum ** 0.5
        gate_grad_norm = gate_grad_sq ** 0.5
        
        # Process Forensic Aggregates
        bias_max = max(gate_bias_vals) if gate_bias_vals else 0.0
        bias_min = min(gate_bias_vals) if gate_bias_vals else 0.0
        bias_mean = sum(gate_bias_vals) / len(gate_bias_vals) if gate_bias_vals else 0.0
        bias_mean = sum(gate_bias_vals) / len(gate_bias_vals) if gate_bias_vals else 0.0
        weight_norm_avg = sum(router_weight_norms) / len(router_weight_norms) if router_weight_norms else 0.0
        
        # Rescue Stats
        rescue_open_mean = sum(rescue_openness_vals) / len(rescue_openness_vals) if rescue_openness_vals else 0.0
        rescue_open_min = min(rescue_openness_vals) if rescue_openness_vals else 0.0
        rescue_scale_mean = sum(rescue_scale_vals) / len(rescue_scale_vals) if rescue_scale_vals else 0.0
        rescue_scale_max = max(rescue_scale_vals) if rescue_scale_vals else 0.0
        
        return {
            "Frozen_Grad_Norm": frozen_grad_norm,
            "Active_Grad_Norm": active_grad_norm, # Found Bug: Was missing!
            "Gate_Grad_Norm": gate_grad_norm,
            "Max_Gate_Mag": max_gate_val,
            
            # Forensic Details
            "Gate_Bias_Max": bias_max,
            "Gate_Bias_Min": bias_min, 
            "Gate_Bias_Mean": bias_mean,
            "Gate_Bias_Min": bias_min, 
            "Gate_Bias_Mean": bias_mean,
            "Router_Weight_Norm": weight_norm_avg,
            
            # Rescue / H2 Telemetry
            "Rescue_Openness_Mean": rescue_open_mean,
            "Rescue_Openness_Min": rescue_open_min,
            "Rescue_Scale_Mean": rescue_scale_mean,
            "Rescue_Scale_Max": rescue_scale_max,
            
            # Legacy/Debug info (optional, keeping for completeness if needed by Logger)
            "frozen_weight_grad": frozen_weight_sq ** 0.5,
            "frozen_bias_grad": frozen_bias_sq ** 0.5
        }
