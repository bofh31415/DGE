
import os
import time
import requests
import json
from dotenv import load_dotenv

# V 0.16.0: Pod Self-Termination Utility
# This script is injected into RunPod instances to terminate them after work is done.

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

    print(f"🚀 [AUTO-TERMINATE] Attempting to terminate pod {pod_id}...")
    
    # Wait a moment to ensure file buffers are flushed and background uploads are done
    # The hf_repo_manager should have finished its work by the time this is called,
    # but we add a safety buffer.
    time.sleep(30) 
    
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
