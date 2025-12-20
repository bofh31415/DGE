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



    def forward(self, x, base_only=False, synergy_mode='gated'):
        # x: [batch, seq, d_model]
        # V 0.12.0: Added synergy_mode propagation
        
        # Attn
        resid = x
        x = self.ln1(x)
        B, T, C = x.size()
        
        # QKV
        q = self.w_q(x, base_only=base_only, synergy_mode=synergy_mode)
        k = self.w_k(x, base_only=base_only, synergy_mode=synergy_mode)
        v = self.w_v(x, base_only=base_only, synergy_mode=synergy_mode)
        
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
        y = self.w_o(y, base_only=base_only, synergy_mode=synergy_mode)
        x = resid + y
        
        # MLP
        resid = x
        x = self.ln2(x)
        x = self.w_mlp_in(x, base_only=base_only, synergy_mode=synergy_mode)
        x = self.act(x)
        x = self.w_mlp_out(x, base_only=base_only, synergy_mode=synergy_mode)
        x = resid + x
        
        return x

    def expand(self, added_width, new_n_head, quadrant, router_type='bigram', use_gradient_rescue=True, use_orthogonal_init=False, isolate_cross_terms=False, cross_term_policy='full', router_init_bias=0.0, gating_threshold: float = 0.0):
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
    def __init__(self, vocab_size=1000, d_model=64, n_layer=2, n_head=4, max_seq_len=1024, initial_gating=False, router_type='bigram', synergy_mode='gated'):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.initial_gating = initial_gating  # V 0.9.3: Store state
        self.router_type = router_type
        self.synergy_mode = synergy_mode  # V 0.12.0: 'gated' or 'additive'

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([DGEBlock(d_model, n_head) for _ in range(n_layer)])
        
        # Initialize Base Head with Gating if requested
        self.lm_head = MoEGatedLinear(d_model, vocab_size, bias=False, initial_gating=initial_gating, router_type=router_type)
        
    def forward(self, idx, targets=None, sparsity_lambda=0.05, base_only=False):
        B, T = idx.size()
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :] # Broadcasting handled
        x = tok_emb + pos_emb
        
        # Base Only Mode (Router 0 Only)
        # Sets a thread-local or global context, OR passes flags down?
        # DGEBlock doesn't take 'base_only' arg in forward.
        # But DGEBlock calls self.w_q(x), which is MoEGatedLinear.
        # MoEGatedLinear needs to know to use only Index 0.
        # We can implement this by temporarily monkey-patching the Gate's behavior 
        # OR better: Add a context manager or attribute to the model that layers check.
        # Let's use a temporary attribute on the model that gates can check if they have reference?
        # No, gates don't have ref to model.
        # 
        # Alternative: We pass `base_only=base_only` to layers.
        # But nn.Sequential / ModuleList makes standard calls `layer(x)`.
        # We need to loop.
        
        for layer in self.layers:
            # We need to hack this slightly or update DGEBlock.forward signature.
            # Let's assume we can update DGEBlock.forward to accept **kwargs or explicit arg.
            # But wait, MoEGatedLinear needs the signal.
            # DGEBlock calls `self.w_q(x)`.
            # 
            # Solution: We set a flag on the module before forward and unset after?
            # A Context Manager approach on the model is cleaner but implementation heavy.
            # 
            # Simple approach: If base_only is True, we zero out the input to expansion experts?
            # Or force routers to output [1.0, 0.0, ...]
            # 
            # The cleanest structure-compliant way is:
            # Pass `base_only` to layer.forward().
            # Update DGEBlock.forward() to accept `base_only`.
            # Update DGEBlock to pass `base_only` to MoEGatedLinear.forward().
            # Update MoEGatedLinear.forward() to accept `base_only`.
            # 
            # This requires updating 3 classes.
            # Let's do it for robustness.
            x = layer(x, base_only=base_only, synergy_mode=self.synergy_mode)
            
        logits = self.lm_head(x, base_only=base_only, synergy_mode=self.synergy_mode)
        
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

    def expand_model(self, new_input_dim, new_output_dim, router_type='bigram', use_gradient_rescue=True, use_orthogonal_init=False, isolate_cross_terms=False, cross_term_policy='full', router_init_bias=0.0, gating_threshold: float = 0.0):
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
            cross_term_policy=cross_term_policy, # V 0.9.4: Use passed policy for retention
            router_type=router_type,
            use_gradient_rescue=use_gradient_rescue,
            use_orthogonal_init=use_orthogonal_init,
            router_init_bias=router_init_bias,
            gating_threshold=gating_threshold
        )
        
        self.d_model = new_input_dim # Update d_model to the new value
        print("Expansion Complete.")

    # =========================================================================
    # SKILL MANAGEMENT API (V 0.9.6: Router0 Always IDK Architecture)
    # =========================================================================
    
    def expand_for_skill(self, skill_name: str, expansion_delta: int = 64, router_type: str = 'rbf') -> int:
        """
        Add capacity for a new skill via expansion.
        
        In the "Router0 Always IDK" architecture:
        - Base model starts with frozen router0 (outputs ~0 = IDK)
        - Each skill adds its own router + capacity
        - After training, freeze the skill's router to lock it
        - Returns: skill_id
        """
        # Initialize registry
        if not hasattr(self, '_skill_registry'):
            self._skill_registry = {}
            self._skill_counter = 0

        # Skill ID
        skill_id = self._skill_counter
        self._skill_counter += 1
        
        # Capture pre-expansion params
        old_d_model = self.d_model
        # Use simple ID set tracking
        pre_params = set(id(p) for p in self.parameters())
        
        # Expand
        new_d_model = self.d_model + expansion_delta
        print(f"\n🎯 Expanding for Skill '{skill_name}' (ID: {skill_id})")
        self.expand_model(
            new_input_dim=new_d_model,
            new_output_dim=new_d_model, # Assuming symmetric vocab for now
            router_type=router_type,
            router_init_bias=0.0, # Open by default
            cross_term_policy='full'
        )
        
        # Capture new params
        post_params = set(id(p) for p in self.parameters())
        new_params = post_params - pre_params
        
        self._skill_registry[skill_id] = {
            'name': skill_name,
            'old_d_model': old_d_model,
            'new_d_model': new_d_model,
            'router_type': router_type,
            'frozen': False,
            'params': new_params
        }
        
        print(f"   Skill '{skill_name}' registered with ID {skill_id} ({len(new_params)} new params)")
        return skill_id
        
    def freeze_skill(self, skill_id: int):
        """
        Freeze a skill's parameters (routers + weights) to prevent forgetting.
        
        V 0.12.0: Now freezes BOTH router AND weight regions.
        Weight regions are frozen via gradient masking (dge_mask).
        """
        if not hasattr(self, '_skill_registry') or skill_id not in self._skill_registry:
            raise ValueError(f"Skill ID {skill_id} not found.")
            
        skill = self._skill_registry[skill_id]
        if skill['frozen']:
            print(f"Warning: Skill {skill_id} already frozen.")
            return

        # 1. Freeze router/gate parameters (by param ID)
        router_count = 0
        for p in self.parameters():
            if id(p) in skill['params']:
                p.requires_grad = False
                router_count += 1
        
        # 2. Freeze weight regions via gradient masking
        # The skill owns columns [old_d_model : new_d_model]
        old_d = skill['old_d_model']
        new_d = skill['new_d_model']
        
        weight_count = 0
        for module in self.modules():
            if isinstance(module, MoEGatedLinear):
                # Get or create dge_mask for this weight
                if not hasattr(module.weight, 'dge_mask'):
                    # Initialize mask to all 1s (all trainable)
                    module.weight.dge_mask = torch.ones_like(module.weight)
                
                # Freeze skill's input columns (affects input dimension)
                # Weight shape: [out_features, in_features]
                # Skill's columns: [:, old_d : new_d]
                in_features = module.weight.shape[1]
                if in_features >= new_d:
                    # Freeze columns corresponding to this skill
                    module.weight.dge_mask[:, old_d:new_d] = 0.0
                    weight_count += module.weight[:, old_d:new_d].numel()
                    
        skill['frozen'] = True
        print(f"Skill '{skill['name']}' (ID: {skill_id}) frozen: {router_count} router params, {weight_count} weight elements masked.")

    # =========================================================================
    # SKILL MANAGEMENT API (V 0.9.6: Router0 Always IDK Architecture)
    # =========================================================================
    
    def expand_for_skill(self, skill_name: str, expansion_delta: int = 64, router_type: str = 'rbf') -> int:
        """
        Add capacity for a new skill via expansion.
        
        In the "Router0 Always IDK" architecture:
        - Base model starts with frozen router0 (outputs ~0 = IDK)
        - Each skill adds its own router + capacity
        - After training, freeze the skill's router to lock it
        
        Args:
            skill_name: Human-readable name for the skill
            expansion_delta: Amount to expand d_model
            router_type: Type of router ('rbf' recommended for OOD detection)
            
        Returns:
            skill_id: Integer ID of the new skill (for freeze_skill)
        """
        # Initialize skill registry if needed
        if not hasattr(self, '_skill_registry'):
            self._skill_registry = {}
            self._skill_counter = 0
        
        # Assign skill ID
        skill_id = self._skill_counter
        self._skill_counter += 1
        
        # Record pre-expansion state - capture param names BEFORE expansion
        old_d_model = self.d_model
        pre_expansion_params = set(id(p) for p in self.parameters())
        
        # Expand model
        new_d_model = self.d_model + expansion_delta
        print(f"\n🎯 Expanding for Skill '{skill_name}' (ID: {skill_id})")
        self.expand_model(
            new_input_dim=new_d_model,
            new_output_dim=new_d_model,
            router_type=router_type,
            router_init_bias=0.0,  # V 0.9.7: Gates start OPEN for plasticity
            cross_term_policy='full'
        )
        
        # Capture NEW params created during expansion
        post_expansion_params = set(id(p) for p in self.parameters())
        new_params = post_expansion_params - pre_expansion_params
        
        # Register skill with its specific params
        self._skill_registry[skill_id] = {
            'name': skill_name,
            'old_d_model': old_d_model,
            'new_d_model': new_d_model,
            'router_type': router_type,
            'frozen': False,
            'params': new_params  # Track IDs of params belonging to this skill
        }
        
        print(f"   Skill '{skill_name}' registered with ID {skill_id} ({len(new_params)} new params)")
        return skill_id
    
    def freeze_skill(self, skill_id: int) -> None:
        """
        Freeze a skill's router after training.
        
        Once frozen, ONLY this skill's router weights become non-trainable,
        preventing forgetting while allowing future skills to learn.
        
        Args:
            skill_id: ID returned by expand_for_skill
        """
        if not hasattr(self, '_skill_registry'):
            raise ValueError("No skills registered. Call expand_for_skill first.")
        
        if skill_id not in self._skill_registry:
            raise ValueError(f"Skill ID {skill_id} not found.")
        
        skill = self._skill_registry[skill_id]
        if skill['frozen']:
            print(f"⚠️ Skill '{skill['name']}' (ID: {skill_id}) is already frozen.")
            return
        
        # Freeze ONLY this skill's parameters (weights + routers)
        skill_params = skill['params']
        frozen_count = 0
        for name, param in self.named_parameters():
            if id(param) in skill_params:
                param.requires_grad = False
                frozen_count += 1
        
        skill['frozen'] = True
        print(f"❄️ Skill '{skill['name']}' (ID: {skill_id}) frozen. {frozen_count} router params locked.")
    
    def get_skill_info(self) -> dict:
        """Get information about registered skills."""
        if not hasattr(self, '_skill_registry'):
            return {}
        return self._skill_registry.copy()

    def imprint_skill(self, skill_id: int, data_loader, num_batches: int = 1, beta_init: float = None):
        """
        Initialize skill's RBF routers by sampling from data.
        
        This enables history-agnostic expansion:
        1. Run forward pass to get hidden states at each layer
        2. Sample these states to set RBF centroids for the new skill
        3. This "imprints" the skill into a specific region of the latent space
        
        Args:
            skill_id: ID of skill to imprint
            data_loader: DataLoader or generator yielding (inputs, targets)
            num_batches: How many batches to use for sampling
            beta_init: Optional override for RBF beta parameter
        """
        if skill_id not in self._skill_registry:
            raise ValueError(f"Skill ID {skill_id} not found.")
        
        skill = self._skill_registry[skill_id]
        if skill['router_type'] != 'rbf':
            print(f"⚠️ Skill '{skill['name']}' uses {skill['router_type']}, not RBF. Skipping imprint.")
            return

        print(f"\n🧬 Imprinting Skill '{skill['name']}' (ID: {skill_id})...")
        skill_params = skill['params']
        
        # 1. Identify RBF routers belonging to this skill and register hooks
        hook_handles = []
        imprint_targets = []
        
        def get_hook(module):
            def hook(mod, inputs, output):
                # inputs[0] is [Batch, Seq, Dim]
                # Only imprint if mask is active
                if getattr(mod, 'imprint_mask', False):
                    # print(f"   Debug: Imprinting {mod} with shape {inputs[0].shape}")
                    mod.imprint_from_batch(inputs[0])
                    mod.imprint_mask = False # Only imprint once per hook activation
            return hook

        count = 0
        for name, module in self.named_modules():
            # Check if this module's parameters are in the skill's parameter set
            # We look for 'centroids' param to identify the RBF router
            is_skill_router = False
            for p_name, p in module.named_parameters():
                if id(p) in skill_params:
                    is_skill_router = True
                    break
            
            if is_skill_router and hasattr(module, 'imprint_from_batch'):
                module.imprint_mask = True
                handle = module.register_forward_hook(get_hook(module))
                hook_handles.append(handle)
                imprint_targets.append(name)
                count += 1
                
        print(f"   Found {count} routers to imprint.")
        
        # 2. Run Forward Pass to trigger hooks
        self.eval() # Ensure deterministic mode
        with torch.no_grad():
            batch_count = 0
            # Handle both list of batches and generator
            iterator = iter(data_loader) if not isinstance(data_loader, list) else data_loader
            
            try:
                for _ in range(num_batches):
                    batch = next(iterator)
                    # Handle (x, y) tuple or just x
                    if isinstance(batch, (tuple, list)):
                        x = batch[0]
                    else:
                        x = batch
                    
                    # Move to device if needed
                    device = next(self.parameters()).device
                    if isinstance(x, torch.Tensor):
                        x = x.to(device)
                    
                    # Forward pass triggers hooks
                    self(x)
                    batch_count += 1
            except StopIteration:
                pass
                
        # 3. Cleanup
        for h in hook_handles:
            h.remove()
        
        # 4. Optional: Reset beta if requested
        if beta_init is not None:
            # Update log_beta for all imprinted routers
            for name, module in self.named_modules():
                if name in imprint_targets and hasattr(module, 'log_beta'):
                    new_val = torch.ones_like(module.log_beta) * torch.log(torch.tensor(beta_init))
                    module.log_beta.data.copy_(new_val)
            print(f"   ✅ Reset beta to {beta_init} for {len(imprint_targets)} routers.")
            
        print(f"   ✅ Imprinted {count} routers from {batch_count} batches.")
        self.train()


    def get_model_confidence(self) -> float:
        """
        Get aggregate confidence from all gated layers.
        
        Returns mean confidence across all MoEGatedLinear layers.
        Used by IDK router to detect overall model uncertainty.
        
        Returns:
            float: Confidence level (0-1), higher = more confident
        """
        confidences = []
        
        for module in self.modules():
            if isinstance(module, MoEGatedLinear):
                conf = module.get_gate_confidence()
                confidences.append(conf)
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 1.0  # Default: fully confident

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
