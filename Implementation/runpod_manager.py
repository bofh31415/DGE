
import os
import time
import requests
import json
from dotenv import load_dotenv

# V 0.14.0: RunPod API Automation
# Automates the "fire and forget" experiment workflow.

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
GIT_TOKEN = os.getenv("GIT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"

def run_query(query, variables=None):
    """Execute a GraphQL query/mutation."""
    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY not found in .env. Please add it to automate pods.")
        
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
        
    response = requests.post(GRAPHQL_URL, json=payload)
    if response.status_code != 200:
        raise Exception(f"Query failed with code {response.status_code}. {response.text}")
        
    result = response.json()
    if 'errors' in result:
        raise Exception(f"GraphQL errors: {result['errors']}")
        
    return result['data']

def deploy_experiment(command, gpu_type="NVIDIA GeForce RTX 4090", gpu_count=1):
    """
    Deploy a pod, clone the repo, setup env, and run the experiment.
    'Fire and forget' mode.
    """
    print(f"🚀 Deploying remote experiment on {gpu_type} (40GB Encrypted Storage)...")
    
    # Construct the robust startup command
    # Use tmux to ensure process survives terminal disconnect
    setup_cmd = (
        f"apt-get update && apt-get install -y git tmux && "
        f"git clone https://{GIT_TOKEN}@github.com/bofh31415/DGE.git && "
        f"cd DGE/Implementation && "
        f"pip install -r requirements.txt && "
        f"echo 'HF_TOKEN={HF_TOKEN}' > .env && "
        f"echo 'GIT_TOKEN={GIT_TOKEN}' >> .env && "
        f"echo 'RUNPOD_API_KEY={RUNPOD_API_KEY}' >> .env && "
        f"tmux new -d -s experiment 'source .env && {command}'"
    )

    mutation = """
    mutation($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        status
      }
    }
    """
    
    variables = {
        "input": {
            "gpuCount": gpu_count,
            "gpuTypeId": gpu_type,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "containerDiskInGb": 40,
            "volumeInGb": 40,
            "volumeEncrypted": True,
            "dockerArgs": f"bash -c '{setup_cmd}'"
        }
    }
    
    data = run_query(mutation, variables)
    pod = data['podFindAndDeployOnDemand']
    
    pod_id = pod['id']
    print(f"✅ Pod created! ID: {pod_id}")
    print(f"🔗 Monitor here: https://www.runpod.io/console/pods")
    
    return pod_id

def terminate_pod(pod_id):
    """Terminate a pod to save costs."""
    print(f"🛑 Terminating pod {pod_id}...")
    mutation = """
    mutation($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    run_query(mutation, variables)
    print("✅ Termination request sent.")

def list_pods():
    """List all active pods."""
    query = """
    query {
      myself {
        pods {
          id
          name
          runtime {
            uptimeInSeconds
          }
          status
          gpuTypeId
        }
      }
    }
    """
    data = run_query(query)
    return data['myself']['pods']

if __name__ == "__main__":
    # Internal CLI for testing
    import argparse
    parser = argparse.ArgumentParser(description="RunPod Experiment Manager")
    parser.add_argument("--deploy", type=str, help="Experiment command to run")
    parser.add_argument("--terminate", type=str, help="Pod ID to terminate")
    parser.add_argument("--list", action="store_true", help="List active pods")
    parser.add_argument("--gpu", type=str, default="NVIDIA GeForce RTX 4090", help="GPU Type")
    
    args = parser.parse_args()
    
    try:
        if args.deploy:
            deploy_experiment(args.deploy, gpu_type=args.gpu)
        elif args.terminate:
            terminate_pod(args.terminate)
        elif args.list:
            pods = list_pods()
            if not pods:
                print("No active pods found.")
            else:
                for p in pods:
                    print(f"[{p['id']}] {p['gpuTypeId']} - {p['status']} (Uptime: {p['runtime']['uptimeInSeconds']}s)")
    except Exception as e:
        print(f"❌ Error: {e}")
