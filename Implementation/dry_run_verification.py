#!/usr/bin/env python3
"""
DRY RUN VERIFICATION SCRIPT
============================
Tests ALL components of the TinyStories → GSM8K chain WITHOUT downloading
full datasets. Uses minimal synthetic data to verify:

1. Model creation (correct dimensions)
2. Model expansion (correct new dimensions)
3. Training loop (forward/backward pass)
4. Replay buffer (add/sample)
5. Checkpointing (save/load)
6. Data loaders (synthetic or tiny real)
7. GPU detection

Run this BEFORE deploying to RunPod to catch any issues locally.

Usage:
    python dry_run_verification.py
"""

import os
import sys
import json
import torch
import tempfile
import shutil

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🔬 DRY RUN VERIFICATION - TinyStories → GSM8K Chain")
print("=" * 70)

# Track all tests
tests_passed = 0
tests_failed = 0
test_results = []

def test(name, condition, details=""):
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        print(f"  ✅ {name}")
        test_results.append({"test": name, "status": "PASS", "details": details})
    else:
        tests_failed += 1
        print(f"  ❌ {name}: {details}")
        test_results.append({"test": name, "status": "FAIL", "details": details})

# ===========================================================================
# TEST 1: Imports
# ===========================================================================
print("\n📦 TEST 1: Module Imports")
print("-" * 40)

try:
    from core.model import DGESimpleTransformer
    test("Import DGESimpleTransformer", True)
except Exception as e:
    test("Import DGESimpleTransformer", False, str(e))

try:
    from core.training import train_dataset, DEVICE
    test("Import train_dataset", True)
    test(f"DEVICE detected: {DEVICE}", True, str(DEVICE))
except Exception as e:
    test("import core.training as dge_training", False, str(e))

try:
    from data.replay_buffer import ReplayBuffer
    test("Import ReplayBuffer", True)
except Exception as e:
    test("Import ReplayBuffer", False, str(e))

try:
    from data.loader import load_tinystories, load_gsm8k, TextDataset
    test("Import data loaders", True)
except Exception as e:
    test("Import data loaders", False, str(e))

try:
    from utils.logger import DGELogger
    test("Import DGELogger", True)
except Exception as e:
    test("Import DGELogger", False, str(e))

try:
    from version import __version__
    test(f"Version: {__version__}", True)
except Exception as e:
    test("Import version", False, str(e))

# ===========================================================================
# TEST 2: Model Creation (TinyStories Config)
# ===========================================================================
print("\n🏗️ TEST 2: Model Creation (TinyStories Config)")
print("-" * 40)

try:
    model = DGESimpleTransformer(
        vocab_size=50257,  # GPT-2
        d_model=384,
        n_layer=12,
        n_head=6,
        max_seq_len=1024
    )
    
    test("Model created", True)
    test(f"d_model = 384", model.d_model == 384)
    test(f"n_layer = 12", len(model.layers) == 12)
    test(f"pos_emb shape = (1, 1024, 384)", model.pos_emb.shape == (1, 1024, 384))
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    test(f"Parameters: {params:,}", params > 50_000_000, f"Expected >50M, got {params:,}")
    
except Exception as e:
    test("Model creation", False, str(e))
    model = None

# ===========================================================================
# TEST 3: Forward Pass
# ===========================================================================
print("\n⚡ TEST 3: Forward Pass")
print("-" * 40)

if model is not None:
    try:
        model = model.to(DEVICE)
        x = torch.randint(0, 1000, (2, 64)).to(DEVICE)  # [batch=2, seq=64]
        y = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        
        logits, loss = model(x, y)
        
        test("Forward pass runs", True)
        test(f"Logits shape: {logits.shape}", logits.shape == (2, 64, 50257))
        test(f"Loss is scalar: {loss.item():.4f}", loss.dim() == 0)
        
    except Exception as e:
        test("Forward pass", False, str(e))

# ===========================================================================
# TEST 4: Backward Pass
# ===========================================================================
print("\n🔄 TEST 4: Backward Pass")
print("-" * 40)

if model is not None:
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        
        x = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        y = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        
        logits, loss = model(x, y)
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        test("Gradients computed", has_grads)
        
        optimizer.step()
        test("Optimizer step", True)
        
    except Exception as e:
        test("Backward pass", False, str(e))

# ===========================================================================
# TEST 5: Model Expansion
# ===========================================================================
print("\n📈 TEST 5: Model Expansion")
print("-" * 40)

if model is not None:
    try:
        old_d = model.d_model
        old_params = sum(p.numel() for p in model.parameters())
        
        model.expand_model(
            new_input_dim=1024,  # 384 + 640
            new_output_dim=50257,
            router_type='bigram',
            use_gradient_rescue=True,
            router_init_bias=0.0,
            gating_threshold=0.0
        )
        
        new_d = model.d_model
        new_params = sum(p.numel() for p in model.parameters())
        new_n_head = model.layers[0].n_head
        
        test(f"d_model expanded: {old_d} → {new_d}", new_d == 1024)
        test(f"n_head scaled: 6 → {new_n_head}", new_n_head == 16)
        test(f"Parameters increased: {old_params:,} → {new_params:,}", new_params > old_params)
        
        # Test forward pass after expansion
        model = model.to(DEVICE)
        x = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        y = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        logits, loss = model(x, y)
        
        test("Forward pass after expansion", logits.shape == (2, 64, 50257))
        
    except Exception as e:
        test("Model expansion", False, str(e))

# ===========================================================================
# TEST 6: Replay Buffer
# ===========================================================================
print("\n🔁 TEST 6: Replay Buffer")
print("-" * 40)

try:
    buffer = ReplayBuffer(max_size=100, task_name="test")
    
    # Add samples
    for i in range(50):
        x = torch.randint(0, 100, (1, 32))
        y = torch.randint(0, 100, (1, 32))
        buffer.add(x, y)
    
    test(f"Buffer size: {len(buffer)}", len(buffer) == 50)
    
    # Sample
    x_s, y_s = buffer.sample(16)
    test(f"Sample shape: {x_s.shape}", x_s.shape == (16, 32))
    
    # Save/Load
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer.save(tmpdir)
        buffer2 = ReplayBuffer.load(tmpdir)
        test(f"Buffer reload: {len(buffer2)}", len(buffer2) == 50)
        
except Exception as e:
    test("Replay buffer", False, str(e))

# ===========================================================================
# TEST 7: Checkpoint Save/Load
# ===========================================================================
print("\n💾 TEST 7: Checkpoint Save/Load")
print("-" * 40)

if model is not None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            torch.save(model.state_dict(), os.path.join(tmpdir, "weights.pt"))
            config = {"d_model": model.d_model, "n_layer": len(model.layers)}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            
            test("Checkpoint saved", True)
            
            # Reload
            state = torch.load(os.path.join(tmpdir, "weights.pt"), weights_only=True)
            test(f"Checkpoint keys: {len(state)}", len(state) > 100)
            
    except Exception as e:
        test("Checkpoint save/load", False, str(e))

# ===========================================================================
# TEST 8: TextDataset (Synthetic)
# ===========================================================================
print("\n📄 TEST 8: TextDataset (Synthetic)")
print("-" * 40)

try:
    texts = ["This is a test sentence.", "Another test here."] * 10
    
    # Simple tokenizer mock
    class MockTokenizer:
        pad_token = "<pad>"
        def __call__(self, text, truncation=True, max_length=64, padding='max_length', return_tensors='pt'):
            ids = [ord(c) % 1000 for c in text[:max_length]]
            ids = ids + [0] * (max_length - len(ids))
            return {'input_ids': torch.tensor([ids])}
    
    dataset = TextDataset(texts, MockTokenizer(), seq_len=32, vocab_size=1000)
    test(f"Dataset length: {len(dataset)}", len(dataset) == 20)
    
    x, y = dataset[0]
    test(f"Sample x shape: {x.shape}", x.shape == (32,))
    test(f"Sample y shape: {y.shape}", y.shape == (32,))
    
except Exception as e:
    test("TextDataset", False, str(e))

# ===========================================================================
# TEST 9: train_dataset (1 step)
# ===========================================================================
print("\n🏋️ TEST 9: train_dataset (1 step dry run)")
print("-" * 40)

if model is not None:
    try:
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create tiny synthetic dataloader
        x_data = torch.randint(0, 1000, (8, 64))
        y_data = torch.randint(0, 1000, (8, 64))
        tiny_dataset = TensorDataset(x_data, y_data)
        tiny_loader = DataLoader(tiny_dataset, batch_size=2)
        
        buffer = ReplayBuffer(max_size=100, task_name="test")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DGELogger(tmpdir)
            
            # Train for 1 "epoch" (4 batches)
            final_step = train_dataset(
                model=model,
                dataloader=tiny_loader,
                epochs=1,
                optimizer=optimizer,
                logger=logger,
                start_step=0,
                checkpoint_fn=None,
                checkpoint_interval=1000,
                replay_buffer=buffer,
                replay_ratio=0.0,
                task_name="test",
                auto_populate_buffer=True
            )
            
            test(f"train_dataset completed: {final_step} steps", final_step > 0)
            test(f"Buffer populated: {len(buffer)} samples", len(buffer) > 0)
            
    except Exception as e:
        test("train_dataset", False, str(e))

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("📊 VERIFICATION SUMMARY")
print("=" * 70)
print(f"  ✅ Passed: {tests_passed}")
print(f"  ❌ Failed: {tests_failed}")
print(f"  Total:    {tests_passed + tests_failed}")
print("=" * 70)

if tests_failed == 0:
    print("\n🚀 ALL TESTS PASSED - Ready for RunPod deployment!")
else:
    print(f"\n⚠️ {tests_failed} TESTS FAILED - Fix issues before deploying!")
    print("\nFailed tests:")
    for r in test_results:
        if r["status"] == "FAIL":
            print(f"  - {r['test']}: {r['details']}")

# Save results
results_file = "dry_run_results.json"
with open(results_file, "w") as f:
    json.dump({
        "passed": tests_passed,
        "failed": tests_failed,
        "tests": test_results
    }, f, indent=2)
print(f"\n📄 Results saved to: {results_file}")
