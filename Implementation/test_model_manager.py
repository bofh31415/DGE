import torch
import os
import shutil
from model_manager import ModelManager, Diary

def test_manager():
    print("Testing ModelManager...")
    mgr = ModelManager("test_models")
    
    # 1. Create Family
    config = {"d_model": 64, "vocab": 100}
    mgr.create_family("TestFamily", config)
    
    family_path = os.path.join("test_models", "TestFamily")
    assert os.path.exists(os.path.join(family_path, "family_config.json"))
    assert os.path.exists(os.path.join(family_path, "diary.md"))
    
    print("Family creation: OK")
    
    # 2. Save Stage 1
    model = torch.nn.Linear(10, 10)
    metrics = {"acc": 0.5}
    stage_path = mgr.save_stage(model, "TestFamily", "Stage1", config, metrics)
    
    assert os.path.exists(os.path.join(stage_path, "weights.pt"))
    assert os.path.exists(os.path.join(stage_path, "diary.md"))
    
    # Check if Family Diary was inherited
    with open(os.path.join(stage_path, "diary.md"), 'r') as f:
        content = f.read()
        assert "TestFamily" in content
        
    print("Stage 1 Save: OK")
    
    # 3. Save Stage 2 (Child of Stage 1)
    metrics2 = {"acc": 0.8}
    stage2_path = mgr.save_stage(model, "TestFamily", "Stage2", config, metrics2, parent_stage="Stage1")
    
    # Check inheritance
    with open(os.path.join(stage2_path, "diary.md"), 'r') as f:
        content = f.read()
        assert "Inherited from Stage1" in content
        assert "acc" in content # Should contain metrics from stage 1 if we copied history? 
        # Wait, inheritance copies the FILE. So yes.
        
    print("Stage 2 Inheritance: OK")
    
    # 4. List Models
    models = mgr.list_models()
    assert len(models) == 2
    print(f"Found {len(models)} models.")
    
    # Cleanup
    shutil.rmtree("test_models")
    print("Cleanup: OK")
    print("Passed all tests.")

if __name__ == "__main__":
    test_manager()
