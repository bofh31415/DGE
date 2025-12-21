
import os
import subprocess
import json
import time
from datetime import datetime

# V 0.16.0: DGE Grand Tour Orchestrator with Progress Tracking
# Executes the 4-stage experimental suite for deep DGE insights.

# Import progress tracker (will work on RunPod)
try:
    from experiments.progress_tracker import update_progress, clear_progress
except ImportError:
    # Fallback for local runs
    def update_progress(*args, **kwargs): pass
    def clear_progress(): pass

STAGES = [
    {
        "id": 1,
        "name": "Integrity Core (Symbol Synergy)",
        "script": "run_synergy_experiment.py",
        "description": "Verifies true additive synergy with Hierarchical Heads."
    },
    {
        "id": 2,
        "name": "Longevity Stress (10-Skill Chain)",
        "script": "run_longevity_chain.py",
        "description": "Tests stability of Skill #1 after 10 sequential expansions."
    },
    {
        "id": 3,
        "name": "Intelligence Transfer (Rosetta Stone)",
        "script": "run_rosetta_stone_experiment.py",
        "description": "English Logic -> German Expression synergy."
    },
    {
        "id": 4,
        "name": "Efficiency Anatomy (Neuro-Bodybuilding)",
        "script": "run_neuro_bodybuilding.py",
        "description": "Sparsity scaling limits of DGE quadrants."
    }
]

def run_stage(stage, total_stages):
    print(f"\n{'='*70}")
    print(f"🚩 STAGE {stage['id']}/{total_stages}: {stage['name']}")
    print(f"   {stage['description']}")
    print(f"{'='*70}")
    
    # Update progress at stage start
    update_progress(
        stage=stage['id'],
        total_stages=total_stages,
        stage_name=stage['name'],
        status="running"
    )
    
    start_time = time.time()
    
    # Run the script via subprocess
    command = ["python", stage['script']]
    
    try:
        # We use Popen to stream output if possible, or just run and wait
        process = subprocess.run(command, check=True)
        status = "SUCCESS"
    except subprocess.CalledProcessError as e:
        print(f"❌ Stage {stage['id']} FAILED with error: {e}")
        status = "FAILED"
    
    duration = time.time() - start_time
    print(f"\n✅ Stage {stage['id']} Finished in {duration/60:.2f} minutes. Status: {status}")
    
    # Update progress at stage complete
    update_progress(
        stage=stage['id'],
        total_stages=total_stages,
        stage_name=stage['name'],
        status="completed" if status == "SUCCESS" else "failed"
    )
    
    return {
        "id": stage['id'],
        "name": stage['name'],
        "status": status,
        "duration_min": duration / 60
    }

def main():
    print("🌍 DGE GRAND TOUR STARTING...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Stages: {len(STAGES)}")
    
    # Clear any old progress
    clear_progress()
    
    results = []
    total_stages = len(STAGES)
    
    for stage in STAGES:
        res = run_stage(stage, total_stages)
        results.append(res)
        
        # If a stage fails, we still continue to the next one to get as much data as possible
        # but we could stop if needed.
        
    print("\n" + "#"*70)
    print("🏆 DGE GRAND TOUR COMPLETE")
    print("#"*70)
    
    # Final progress update
    update_progress(
        stage=total_stages,
        total_stages=total_stages,
        stage_name="Complete",
        status="completed"
    )
    
    report_path = "grand_tour_report.json"
    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": results
    }
    
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nFinal report saved to {report_path}")
    print("Insights are now available in your models/ directory and checkpoints are on HuggingFace.")

if __name__ == "__main__":
    main()
