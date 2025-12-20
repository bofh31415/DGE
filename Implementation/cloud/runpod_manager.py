
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


# GPU Performance Database (TFLOPS for FP16/BF16 training, approximate)
# Keys match RunPod's displayName format
GPU_PERFORMANCE = {
    # Consumer RTX 40 Series
    "RTX 4090": {"tflops": 82.6, "vram": 24},
    "RTX 4080 SUPER": {"tflops": 52.2, "vram": 16},
    "RTX 4080": {"tflops": 48.7, "vram": 16},
    "RTX 4070 Ti SUPER": {"tflops": 44.1, "vram": 16},
    "RTX 4070 Ti": {"tflops": 40.1, "vram": 12},
    "RTX 4070 SUPER": {"tflops": 35.5, "vram": 12},
    "RTX 4070": {"tflops": 29.2, "vram": 12},
    # Consumer RTX 30 Series
    "RTX 3090 Ti": {"tflops": 40.0, "vram": 24},
    "RTX 3090": {"tflops": 35.6, "vram": 24},
    "RTX 3080 Ti": {"tflops": 34.1, "vram": 12},
    "RTX 3080": {"tflops": 29.8, "vram": 10},
    "RTX 3070 Ti": {"tflops": 21.7, "vram": 8},
    "RTX 3070": {"tflops": 20.3, "vram": 8},
    # RTX 50 Series (Blackwell)
    "RTX 5090": {"tflops": 170.0, "vram": 32},  # Estimated
    "RTX 5080": {"tflops": 95.0, "vram": 16},   # Estimated
    # Professional RTX A Series
    "RTX A6000": {"tflops": 38.7, "vram": 48},
    "RTX A5000": {"tflops": 27.8, "vram": 24},
    "RTX A4500": {"tflops": 23.7, "vram": 20},
    "RTX A4000": {"tflops": 19.2, "vram": 16},
    "RTX A2000": {"tflops": 7.99, "vram": 6},
    # Professional RTX Ada Series  
    "RTX 4000 Ada": {"tflops": 26.7, "vram": 20},
    "RTX 4000 Ada SFF": {"tflops": 19.2, "vram": 20},
    "RTX PRO 6000": {"tflops": 91.1, "vram": 96},
    # Data Center - NVIDIA A Series
    "A100 PCIe": {"tflops": 77.9, "vram": 40},
    "A100 SXM": {"tflops": 77.9, "vram": 80},
    "A40": {"tflops": 37.4, "vram": 48},
    "A30": {"tflops": 20.0, "vram": 24},
    "A10": {"tflops": 31.2, "vram": 24},
    # Data Center - NVIDIA L Series (Ada Lovelace)
    "L40": {"tflops": 181.0, "vram": 48},
    "L40S": {"tflops": 366.0, "vram": 48},
    "L4": {"tflops": 30.3, "vram": 24},
    # Data Center - NVIDIA H Series (Hopper)
    "H100 PCIe": {"tflops": 204.9, "vram": 80},
    "H100 SXM": {"tflops": 267.6, "vram": 80},
    "H100 NVL": {"tflops": 267.6, "vram": 94},
    "H200 SXM": {"tflops": 267.6, "vram": 141},  # Same compute as H100, more VRAM
    # Data Center - NVIDIA B Series (Blackwell)
    "B200": {"tflops": 500.0, "vram": 180},  # Estimated
    # Legacy V100
    "Tesla V100": {"tflops": 14.0, "vram": 16},
    "V100 FHHL": {"tflops": 14.0, "vram": 16},
    "V100 SXM2": {"tflops": 15.7, "vram": 16},
    "V100 SXM2 32GB": {"tflops": 15.7, "vram": 32},
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
        securePrice
        communityPrice
      }
    }
    """

    try:
        data = run_query(query)
        gpus = data.get('gpuTypes', [])
        
        result = []
        for g in gpus:
            # Use communityPrice (spot) if available, otherwise securePrice
            price = g.get('communityPrice') or g.get('securePrice')
            if not price or price <= 0:
                continue
                
            gpu_id = g['id']
            name = g['displayName']
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
    # V0.18.0: Simplified to avoid nested quote issues
    cleanup_step = " && python -m experiments.pod_cleanup" if auto_terminate else ""
    repo_name = os.getenv("HF_REPO", "darealSven/dge")
    
    # Use double quotes for outer, escape inner quotes
    setup_cmd = (
        f"apt-get update && apt-get install -y git && "
        f"git clone https://{GIT_TOKEN}@github.com/bofh31415/DGE.git && "
        f"cd DGE/Implementation && "
        f"pip install -r requirements.txt && "
        f"export HF_TOKEN={HF_TOKEN} && "
        f"export GIT_TOKEN={GIT_TOKEN} && "
        f"export RUNPOD_API_KEY={RUNPOD_API_KEY} && "
        f"export HF_REPO={repo_name} && "
        f"{command}{cleanup_step}"
    )

    mutation = """
    mutation($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        desiredStatus
      }
    }
    """

    
    variables = {
        "input": {
            "name": f"dge-experiment-{int(time.time())}",
            "gpuCount": gpu_count,
            "gpuTypeId": gpu_type,
            "cloudType": "COMMUNITY" if is_spot else "SECURE",
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "containerDiskInGb": 40,
            "volumeInGb": 40,
            "ports": "5000/http,22/tcp",
            "startSsh": True,
            "dockerArgs": setup_cmd
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
          desiredStatus
          machine {
            gpuDisplayName
          }
          runtime {
            uptimeInSeconds
            ports {
              ip
              publicPort
            }
          }
        }
      }
    }
    """
    data = run_query(query)
    pods = data['myself']['pods']
    
    # Transform to expected format for compatibility
    result = []
    for p in pods:
        # Extract first port address if available
        ports = p.get('runtime', {}).get('ports', []) if p.get('runtime') else []
        addr = f"{ports[0]['ip']}:{ports[0]['publicPort']}" if ports else "N/A"
        
        result.append({
            'id': p['id'],
            'name': p.get('name', ''),
            'status': p.get('desiredStatus', 'UNKNOWN'),
            'gpuTypeId': p.get('machine', {}).get('gpuDisplayName', 'Unknown'),
            'runtime': {
                'uptimeInSeconds': p.get('runtime', {}).get('uptimeInSeconds', 0) if p.get('runtime') else 0,
                'address': addr
            }
        })
    return result

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
