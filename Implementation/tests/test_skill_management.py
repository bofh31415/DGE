
import unittest
import torch
import sys
import os

# Add implementation (parent directory) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from dge_model import DGESimpleTransformer

class TestSkillManagement(unittest.TestCase):
    def setUp(self):
        # Base Model
        self.vocab = 100
        self.d_model = 32
        self.model = DGESimpleTransformer(
            vocab_size=self.vocab, d_model=self.d_model, n_layer=1, n_head=4,
            router_type='rbf', initial_gating=False
        )
        
    def test_expand_and_freeze(self):
        print("\n--- Testing Expand & Freeze ---")
        
        # 1. Expand for Skill A
        skill_id_a = self.model.expand_for_skill("Skill_A", expansion_delta=16)
        
        # Verify Registry
        info = self.model.get_skill_info()
        self.assertIn(skill_id_a, info)
        self.assertEqual(info[skill_id_a]['name'], "Skill_A")
        self.assertFalse(info[skill_id_a]['frozen'])
        
        # 2. Train a bit (simulate)
        # Check gradients exist
        dummy_input = torch.randint(0, self.vocab, (2, 10))
        out, _ = self.model(dummy_input)
        loss = out.mean()
        loss.backward()
        
        # Check if new parameters have grad
        has_grad = False
        new_param_ids = info[skill_id_a]['params']
        for p in self.model.parameters():
            if id(p) in new_param_ids and p.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "New parameters should have gradients")
        
        # 3. Freeze Skill A
        self.model.zero_grad()
        self.model.freeze_skill(skill_id_a)
        self.assertTrue(info[skill_id_a]['frozen'])
        
        # 4. Train again - Should have NO grad on frozen params
        out, _ = self.model(dummy_input)
        loss = out.mean()
        loss.backward()
        
        frozen_grad_found = False
        for p in self.model.parameters():
            if id(p) in new_param_ids:
                # Should NOT have grad (or requires_grad=False)
                if p.requires_grad:
                    frozen_grad_found = True
        self.assertFalse(frozen_grad_found, "Frozen parameters should not require grad")
        print("✅ Freeze Skill logic validated.")

    def test_base_only_forward(self):
        print("\n--- Testing Base Only Forward ---")
        
        # Expand model so we have something to mask
        self.model.expand_for_skill("Skill_B", expansion_delta=32)
        
        # Input
        x = torch.randint(0, self.vocab, (1, 10))
        
        # Forward with all active
        out_full, _ = self.model(x, base_only=False)
        
        # Forward with base only
        out_base, _ = self.model(x, base_only=True)
        
        # Result should differ (unless expansion initialized to perfect zero identity)
        # With RBF/Bigram initialized near zero, difference might be small but non-zero.
        # But wait, 'base_only' zeros out the expansion contribution.
        # If expansion contributes anything, diff > 0.
        
        # IMPORTANT: If initialized with strict identity, out_full SHOULD equal out_base initially!
        # DGE initializes with zero-ish logic for new branches if configured.
        # But RBF router might have some activation.
        
        diff = (out_full - out_base).abs().max().item()
        print(f"Difference (Full vs Base): {diff}")
        
        # To prove base_only works, we can manually perturb weights in expansion zone
        # and see if base_only output remains stable.
        
        # Perturb expansion weights (New Inputs -> Output)
        # weight shape is [Out, In]. In=64. Old=32.
        # We want to perturb columns 32:64.
        with torch.no_grad():
            self.model.lm_head.weight[:, 32:] += 10.0 
            
        out_full_perturbed, _ = self.model(x, base_only=False)
        out_base_perturbed, _ = self.model(x, base_only=True)
        
        diff_perturbed = (out_full_perturbed - out_base_perturbed).abs().max().item()
        print(f"Difference after perturbation: {diff_perturbed}")
        self.assertGreater(diff_perturbed, 1.0, "Perturbation should affect full forward")
        
        # Check base stability
        # Base output should NOT depend on expansion perturbations (if fully isolated)
        # Actually... Base Only zeroes out the expansion PART.
        # But if expansion weights changed, does it affect Base output?
        # Only via feedback? DGE is feedforward.
        # Base output comes from Top-Left quadrant mainly.
        # Wait, Base Only flag sets mask on OUTPUT of layers.
        # So outputs of layers are [Base, 0].
        # So next layer sees [Base, 0].
        # So it computes Base part using Top-Left.
        # So yes, it should correspond to the original base model functionality.
        
        # Let's compare out_base and out_base_perturbed
        base_diff = (out_base - out_base_perturbed).abs().max().item()
        print(f"Base Stability Diff: {base_diff}")
        
        self.assertLess(base_diff, 1e-5, "Base output should be effectively invariant to expansion weight changes when base_only=True")
        print("✅ Base Only logic validated.")

if __name__ == '__main__':
    unittest.main()
