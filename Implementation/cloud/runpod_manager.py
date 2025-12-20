
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


# GPU Performance Database (TFLOPS for FP16 training, approximate)
# Source: Official specs and benchmarks
GPU_PERFORMANCE = {
    "NVIDIA GeForce RTX 4090": {"tflops": 82.6, "vram": 24},
    "NVIDIA GeForce RTX 4080": {"tflops": 48.7, "vram": 16},
    "NVIDIA GeForce RTX 4070 Ti": {"tflops": 40.1, "vram": 12},
    "NVIDIA GeForce RTX 3090": {"tflops": 35.6, "vram": 24},
    "NVIDIA GeForce RTX 3080": {"tflops": 29.8, "vram": 10},
    "NVIDIA A100 PCIe": {"tflops": 77.9, "vram": 40},
    "NVIDIA A100 SXM": {"tflops": 77.9, "vram": 80},
    "NVIDIA A40": {"tflops": 37.4, "vram": 48},
    "NVIDIA L40": {"tflops": 181.0, "vram": 48},
    "NVIDIA H100 PCIe": {"tflops": 204.9, "vram": 80},
    "NVIDIA H100 SXM": {"tflops": 267.6, "vram": 80},
}

def get_available_gpus():
    """
    Queries RunPod API for available GPUs with spot prices.
    Returns a list of dicts with id, name, price, tflops, vram, value_score.
    """
    query = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(input: {onDemand: false}) {
          price
        }
      }
    }
    """
    try:
        data = run_query(query)
        gpus = data.get('gpuTypes', [])
        
        result = []
        for g in gpus:
            if not g.get('lowestPrice'):
                continue
                
            gpu_id = g['id']
            name = g['displayName']
            price = g['lowestPrice']['price']
            vram = g.get('memoryInGb', 0)
            
            # Lookup performance data
            perf = GPU_PERFORMANCE.get(name, {"tflops": 20.0, "vram": vram})  # Default estimate
            tflops = perf['tflops']
            
            # Value score: TFLOPS per dollar per hour
            value_score = tflops / price if price > 0 else 0
            
            result.append({
                "id": gpu_id,
                "name": name,
                "price": price,
                "tflops": tflops,
                "vram": vram,
                "value": value_score
            })
        
        # Sort by value score (best value first)
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
        
    except Exception as e:
        print(f"⚠️ Error fetching GPUs: {e}")
        return []

def display_gpu_selection_menu(gpus):
    """Display GPU options with prices and performance metrics."""
    print("\n" + "="*85)
    print("                         GPU SELECTION MENU (Spot Instances)")
    print("="*85)
    print(f"{'Idx':<4} | {'GPU':<30} | {'$/hr':<7} | {'TFLOPS':<8} | {'VRAM':<6} | {'Value':<8} | {'⭐'}")
    print("-"*85)
    
    best_value_idx = 0  # First one is best (sorted by value)
    
    for idx, g in enumerate(gpus):
        star = "⭐ BEST" if idx == best_value_idx else ""
        print(f"{idx+1:<4} | {g['name']:<30} | ${g['price']:<6.2f} | {g['tflops']:<8.1f} | {g['vram']:<4}GB | {g['value']:<8.1f} | {star}")
    
    print("-"*85)
    print(f"💡 Recommendation: #{1} ({gpus[0]['name']}) - Best TFLOPS per dollar!")
    print("="*85)
    
    return best_value_idx

def select_gpu_interactive():
    """Interactive GPU selection. Returns (gpu_id, price) or None if cancelled."""
    gpus = get_available_gpus()
    
    if not gpus:
        print("❌ No GPUs available. Check your RUNPOD_API_KEY.")
        return None, None
    
    best_idx = display_gpu_selection_menu(gpus)
    
    while True:
        choice = input(f"\nSelect GPU [1-{len(gpus)}] (Enter for recommended, 'q' to cancel): ").strip().lower()
        
        if choice == 'q':
            return None, None
        elif choice == '':
            # Use recommended
            selected = gpus[best_idx]
            print(f"✅ Selected: {selected['name']} @ ${selected['price']}/hr")
            return selected['id'], selected['price']
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(gpus):
                    selected = gpus[idx]
                    print(f"✅ Selected: {selected['name']} @ ${selected['price']}/hr")
                    return selected['id'], selected['price']
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid input. Enter a number or press Enter for recommended.")

def find_cheapest_gpu(gpu_display_name="NVIDIA GeForce RTX 4090"):
    """
    Queries the RunPod API to find the current cheapest spot price for a specific GPU.
    (Legacy function - still used for non-interactive deployments)
    """
    gpus = get_available_gpus()
    
    # Filter by name if specified
    if gpu_display_name:
        candidates = [g for g in gpus if gpu_display_name.lower() in g['name'].lower()]
    else:
        candidates = gpus
    
    if candidates:
        best = candidates[0]  # Already sorted by value
        print(f"💰 Found cheapest {best['name']}: ${best['price']}/hr")
        return best['id'], best['price']
    
    # Fallback
    return "NVIDIA GeForce RTX 4090", 0.69

def deploy_experiment(command, gpu_type=None, gpu_count=1, auto_terminate=True, is_spot=True):
    """
    Deploy a pod, clone the repo, setup env, and run the experiment.
    'Fire and forget' mode.
    """
    if gpu_type is None:
        gpu_type, price = find_cheapest_gpu()
    else:
        price = "Unknown"
        
    label = " (SPOT Instance)" if is_spot else " (ON-DEMAND)"
    print(f"🚀 Deploying remote experiment on {gpu_type}{label}...")
    print(f"📈 Estimated Cost: ${price}/hr")
    
    # Construct the robust startup command
    # V0.16.0: Added pod_cleanup.py execution at the end
    # V0.17.0: Start remote_inference_server in parallel
    cleanup_step = " && python pod_cleanup.py" if auto_terminate else ""
    repo_name = os.getenv("HF_REPO", "darealSven/dge")
    
    setup_cmd = (
        f"apt-get update && apt-get install -y git tmux && "
        f"git clone https://{GIT_TOKEN}@github.com/bofh31415/DGE.git && "
        f"cd DGE/Implementation && "
        f"pip install -r requirements.txt && "
        f"echo 'HF_TOKEN={HF_TOKEN}' > .env && "
        f"echo 'GIT_TOKEN={GIT_TOKEN}' >> .env && "
        f"echo 'RUNPOD_API_KEY={RUNPOD_API_KEY}' >> .env && "
        f"echo 'HF_REPO={repo_name}' >> .env && "
        f"echo 'RUNPOD_POD_ID='$(printenv RUNPOD_POD_ID) >> .env && "
        f"tmux new -d -s inference 'source .env && python remote_inference_server.py' && "
        f"tmux new -d -s experiment 'source .env && {command}{cleanup_step}'"
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
            "interruptible": is_spot,
            "ports": "5000/http", # V0.17.0: Expose inference port
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
            address
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
