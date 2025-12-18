"""
Example: Using HFRepoManager for Unified HF Structure
=====================================================
Shows how to integrate HFRepoManager into experiment scripts.
"""

from hf_repo_manager import HFRepoManager, wait_for_uploads

# Initialize for your experiment
manager = HFRepoManager("tinystories_gsm8k")

# ============================================================================
# CRASH RECOVERY - At experiment start
# ============================================================================

# Check for resume state (downloads from HF if not local)
resume_state = manager.get_resume_state()

if resume_state:
    phase = resume_state.get("phase", 0)
    step = resume_state.get("step", 0)
    print(f"Resuming from phase {phase}, step {step}")
    
    # Ensure checkpoint is available locally
    ckpt_path = f"models/tinystories_gsm8k/resume_checkpoint"
    if manager.ensure_local_checkpoint(ckpt_path, "resume"):
        # Load weights
        model.load_state_dict(torch.load(f"{ckpt_path}/weights.pt"))
        optimizer.load_state_dict(torch.load(f"{ckpt_path}/optimizer.pt"))

# ============================================================================
# SHARED BASE MODEL - Before Phase 2 training
# ============================================================================

# Check if TinyStories base already exists
base_path = manager.download_shared_base(
    dataset="tinystories",
    d_model=384,
    n_head=6,
    n_layer=12
)

if base_path:
    print("✅ Using shared TinyStories base")
    model.load_state_dict(torch.load(base_path))
    skip_phase2 = True
else:
    # Train from scratch
    train_tinystories(...)
    
    # Upload for future experiments to reuse
    manager.upload_shared_base(
        local_weights=f"models/exp/weights.pt",
        dataset="tinystories",
        d_model=384,
        n_head=6,
        n_layer=12,
        step=final_step
    )

# ============================================================================
# CHECKPOINT UPLOAD - During training
# ============================================================================

# Resume checkpoint (overwrites)
manager.upload_checkpoint(
    local_path="models/exp/resume_checkpoint",
    step=current_step,
    checkpoint_type="resume",
    is_milestone=False,
    background=True  # Non-blocking
)

# Milestone (permanent)
manager.upload_checkpoint(
    local_path="models/exp/milestone_tinystories",
    step=current_step,
    checkpoint_type="tinystories",
    is_milestone=True,
    background=True
)

# ============================================================================
# LOGS UPLOAD - After checkpoints
# ============================================================================

manager.upload_logs("models/exp/logs", background=True)

# ============================================================================
# RESULTS UPLOAD - At experiment end
# ============================================================================

manager.upload_results("models/exp/experiment_results.json")

# Wait for all background uploads
wait_for_uploads()
