
import os
import time
import requests
import json
from dotenv import load_dotenv

# V 0.16.0: Pod Self-Termination Utility
# This script is injected into RunPod instances to terminate them after work is done.

def sync_forensics():
    """Ensure all diaries, logs, and checkpoints are pushed to HF."""
    print("🏛️ [FORENSICS] Scanning for vital data (checkpoints, diaries, logs)...")
    
    # We leverage the hf_repo_manager or simply use the CLI for a final broad push
    repo = os.getenv("HF_REPO", "darealSven/dge")
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("⚠️ [FORENSICS] No models directory found. Skipping sync.")
        return True

    try:
        import subprocess
        print(f"🚀 [FORENSICS] Final push to {repo}...")
        
        # We push the entire models directory to ensure nothing is missed
        # This includes diaries and logs nested within family folders
        result = subprocess.run(
            ["huggingface-cli", "upload", repo, models_dir, "--path-in-repo", "models"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ [FORENSICS] Final sync successful.")
            return True
        else:
            print(f"❌ [FORENSICS] Final sync failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ [FORENSICS] Error during sync: {e}")
        return False

def terminate_self():
    load_dotenv()
    
    api_key = os.getenv("RUNPOD_API_KEY")
    pod_id = os.getenv("RUNPOD_POD_ID") # Set during deployment by runpod_manager
    
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY not found in .env. Cannot self-terminate.")
        return
        
    if not pod_id:
        print("❌ Error: RUNPOD_POD_ID not found in .env. Cannot self-terminate.")
        return

    # V0.17.0: Forensic Sync is MANDATORY before termination
    if not sync_forensics():
        print("⚠️ [AUTO-TERMINATE] Forensic sync failed! Aborting termination to prevent data loss.")
        return

    print(f"🚀 [AUTO-TERMINATE] Attempting to terminate pod {pod_id}...")
    
    # Wait a moment to ensure file buffers are flushed
    time.sleep(10) 
    
    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    mutation = """
    mutation($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    
    try:
        response = requests.post(url, json={'query': mutation, 'variables': variables})
        if response.status_code == 200:
            print(f"✅ [AUTO-TERMINATE] Termination request sent successfully for {pod_id}.")
        else:
            print(f"❌ [AUTO-TERMINATE] Request failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ [AUTO-TERMINATE] Error: {e}")

if __name__ == "__main__":
    terminate_self()
