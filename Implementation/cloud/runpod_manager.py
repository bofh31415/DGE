
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
            gpu_id = g['id']
            name = g['displayName']
            
            # Prices
            spot_price = g.get('communityPrice')
            secure_price = g.get('securePrice')
            
            # Skip if both are unavailable
            if (not spot_price or spot_price <= 0) and (not secure_price or secure_price <= 0):
                continue
                
            vram = g.get('memoryInGb', 0)
            
            # Lookup performance data
            perf = GPU_PERFORMANCE.get(name, {"tflops": 20.0, "vram": vram})
            tflops = perf['tflops']
            
            # Calculate Value Scores (TFLOPS / Price)
            val_spot = tflops / spot_price if spot_price and spot_price > 0 else 0
            val_secure = tflops / secure_price if secure_price and secure_price > 0 else 0
            
            # Sorting metric: heavily favor Spot value, fallback to Secure value
            sort_metric = max(val_spot, val_secure * 0.8) 
            
            result.append({
                "id": gpu_id,
                "name": name,
                "communityPrice": spot_price,
                "securePrice": secure_price,
                "tflops": tflops,
                "vram": vram,
                "value_spot": val_spot,
                "value_secure": val_secure,
                "sort_metric": sort_metric
            })
            
        # Sort by value (best first)
        result.sort(key=lambda x: x['sort_metric'], reverse=True)
        return result
        
    except Exception as e:
        print(f"⚠️ Error fetching GPUs: {e}")
        return []

def display_gpu_selection_menu(gpus):
    """
    Display a formatted table of available GPUs with Spot and Secure prices.
    Highlights the best value options.
    """
    print("\n" + "="*105)
    print(f"{'GPU SELECTION MENU':^105}")
    print("="*105)
    
    # Header
    # Idx | GPU Name | Spot $/hr | Secure $/hr | TFLOPS | VRAM | Val(Spot) | Val(Sec) | Rec
    header = f"{'Idx':<4} | {'GPU':<30} | {'Spot $':<9} | {'Secure $':<9} | {'TFLOPS':<8} | {'VRAM':<6} | {'Val(S)':<7} | {'Val(D)':<7} | {'⭐'}"
    print(header)
    print("-" * 105)
    
    for i, gpu in enumerate(gpus, 1):
        idx = str(i)
        name = gpu['name'][:30]
        
        # Prices
        spot_price = f"${gpu['communityPrice']:.2f}" if gpu['communityPrice'] else "N/A"
        sec_price = f"${gpu['securePrice']:.2f}" if gpu.get('securePrice') and gpu['securePrice'] > 0 else "N/A"
        
        tflops = f"{gpu['tflops']:.1f}"
        vram = f"{int(gpu['vram']):<3} GB"
        
        # Value Scores
        val_spot = f"{gpu['value_spot']:.1f}" if gpu['communityPrice'] else "-"
        val_sec = f"{gpu['value_secure']:.1f}" if gpu.get('securePrice') and gpu['securePrice'] > 0 else "-"
        
        # Star rating (based on Spot value usually)
        star = "⭐ BEST" if i == 1 else ""
        if gpu.get('securePrice') and gpu['securePrice'] > 0 and i <= 3:
             star += " (Sec)"
             
        print(f"{idx:<4} | {name:<30} | {spot_price:<9} | {sec_price:<9} | {tflops:<8} | {vram:<6} | {val_spot:<7} | {val_sec:<7} | {star}")
        
    print("-" * 105)
    print("💡 Val = TFLOPS / Price. Higher is better value.")
    print("💡 'Secure' = Dedicated (High Reliability). 'Spot' = Interruptible (Cheaper).")
    print("=" * 105)

def select_gpu_interactive():
    """
    Interactive sequence to select a GPU and deployment mode.
    Returns: (gpu_id, is_spot_mode, cost_per_hr)
    """
    import time as t
    start = t.time()
    print("\n🔍 Fetching latest GPU pricing...", end='', flush=True)
    gpus = get_available_gpus()
    elapsed = t.time() - start
    print(f" ({elapsed:.1f}s)")
    
    if not gpus:
        print("❌ No GPUs found available via RunPod API.")
        return None, True, 0


    display_gpu_selection_menu(gpus)
    
    best_gpu = gpus[0]
    
    while True:
        choice = input(f"\nSelect GPU [1-{len(gpus)}] (Enter for recommended '{best_gpu['name']}'), or 'q' to cancel: ").strip()
        
        if choice.lower() == 'q':
            return None, True
            
        selected_gpu = None
        if not choice:
            selected_gpu = best_gpu
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(gpus):
                    selected_gpu = gpus[idx-1]
                else:
                    print("❌ Invalid index.")
                    continue
            except ValueError:
                print("❌ Invalid input.")
                continue
        
        # Now ask for Mode (Spot vs Secure)
        print(f"\n✅ Selected: {selected_gpu['name']}")
        
        has_spot = selected_gpu['communityPrice'] and selected_gpu['communityPrice'] > 0
        has_secure = selected_gpu.get('securePrice') and selected_gpu['securePrice'] > 0
        
        if not has_spot and not has_secure:
            print("❌ Error: This GPU has no valid pricing logic.")
            return None, True
            
        print("\nSelect Deployment Mode:")
        if has_spot:
            print(f"1. Spot (Community)   - ${selected_gpu['communityPrice']:.2f}/hr  (Interruptible, cheaper)")
        if has_secure:
            print(f"2. Secure (Dedicated) - ${selected_gpu['securePrice']:.2f}/hr  (Reliable, no interruptions)")
            
        mode_choice = input("\nSelect Mode [1/2] (Enter for Spot if available): ").strip()
        
        is_spot = True
        if mode_choice == '2':
            if has_secure:
                is_spot = False # Secure
            else:
                print("❌ Secure mode not available for this GPU.")
                continue
        elif mode_choice == '1' or mode_choice == '':
             if not has_spot:
                 print("❌ Spot mode not available per metrics, trying Secure...")
                 is_spot = False
        
        mode_str = "SPOT" if is_spot else "SECURE (Dedicated)"
        cost = selected_gpu['communityPrice'] if is_spot else selected_gpu['securePrice']
        
        confirm = input(f"\n🚀 Deploy on {selected_gpu['name']} [{mode_str}] @ ${cost:.2f}/hr? (y/n): ")
        if confirm.lower() == 'y':
            return selected_gpu['id'], is_spot, cost
        else:
            print("❌ Deployment cancelled.")
            return None, True, 0

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

def deploy_experiment(command, gpu_type=None, gpu_count=1, auto_terminate=True, is_spot=True, price=None):
    """
    Deploy a pod, clone the repo, setup env, and run the experiment.
    'Fire and forget' mode.
    """
    if gpu_type is None:
        gpu_type, found_price = find_cheapest_gpu()
        price = found_price if price is None else price
    else:
        if price is None:
             price = "Unknown"
        
    label = " (SPOT Instance)" if is_spot else " (ON-DEMAND)"
    print(f"🚀 Deploying remote experiment on {gpu_type}{label}...")
    
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    print(f"📈 Estimated Cost: {price_str}/hr")
    
    # Construct the robust startup command
    # V0.21.2: Added PYTHONPATH for module discovery
    cleanup_step = " && python -m experiments.pod_cleanup" if auto_terminate else ""
    repo_name = os.getenv("HF_REPO", "darealSven/dge")
    log_file = "/workspace/startup.log"
    work_dir = "/workspace/DGE/Implementation"
    
    # Build the inner command string (will be passed to bash -c)
    inner_cmd = (
        f"echo [1/6] Starting setup... >> {log_file} && date >> {log_file} && "
        f"echo [2/6] Updating system... >> {log_file} && apt-get update -qq && apt-get install -y git -qq && "
        f"echo [3/6] Cloning repo... >> {log_file} && rm -rf /workspace/DGE && "
        f"git clone --depth 1 -b exp/hierarchical-output https://{GIT_TOKEN}@github.com/bofh31415/DGE.git /workspace/DGE && "
        f"echo [4/6] Installing dependencies... >> {log_file} && "
        f"cd {work_dir} && pip install -r requirements.txt >> {log_file} 2>&1 && "
        f"export PYTHONPATH={work_dir}:$PYTHONPATH && "
        f"export HF_TOKEN={HF_TOKEN} && "
        f"export GIT_TOKEN={GIT_TOKEN} && "
        f"export RUNPOD_API_KEY={RUNPOD_API_KEY} && "
        f"export HF_REPO={repo_name} && "
        f"echo [5/6] Starting experiment... >> {log_file} && date >> {log_file} && "
        f"cd {work_dir} && {command} 2>&1 | tee -a {log_file} && "
        f"echo [6/6] Experiment complete. >> {log_file}{cleanup_step}"
    )

    
    # Wrap in bash -c for proper execution
    inner_cmd_escaped = inner_cmd.replace("'", "'\\''")
    setup_cmd = f"bash -c '{inner_cmd_escaped}'"






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
            "volumeMountPath": "/workspace",
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
    
    
    # Wait for pod to actually start (with timeout)
    timeout_seconds = 300  # 5 minutes to allow for pip install
    poll_interval = 15  # Check less frequently
    elapsed = 0
    
    print(f"\n⏳ Waiting for container startup (Timeout: {timeout_seconds}s)...")
    print("   Note: 'pip install' takes 2-3 minutes. Check RunPod Console for live logs.")
    
    container_started = False
    while elapsed < timeout_seconds:
        remaining = timeout_seconds - elapsed
        print(f"   ⏱️ Time remaining: {remaining}s | Status: Checking...", end='\r')
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        try:
            pods = list_pods()
            our_pod = next((p for p in pods if p['id'] == pod_id), None)
            if our_pod:
                runtime = our_pod.get('runtime', {})
                uptime = runtime.get('uptimeInSeconds', 0) 
                status = our_pod.get('status', 'UNKNOWN')
                
                # Check if container is running (even with 0 uptime)
                if status == "RUNNING":
                    container_started = True
                    if uptime > 0:
                        print(f"\n✅ Pod runtime started! Uptime: {uptime}s")
                        return pod_id
                    else:
                        # Container is RUNNING but uptime is 0 - setup in progress
                        print(f"   ⏱️ Time remaining: {remaining}s | Status: {status} (setup in progress)", end='\r')
                else:
                    print(f"   ⏱️ Time remaining: {remaining}s | Status: {status}   ", end='\r')
        except Exception as e:
            print(f"\n   Poll error: {e}")
            
    print() # Newline after loop
    
    # If container started but setup is still running, DON'T terminate - just return
    if container_started:
        print(f"✅ Container is RUNNING! Setup still in progress.")
        print(f"   The experiment will continue. Monitor at: https://www.runpod.io/console/pods")
        return pod_id
    else:
        # Container never started - something is wrong, terminate
        print(f"⚠️ Container did not start within {timeout_seconds}s.")
        print(f"🛑 Auto-terminating stuck pod {pod_id} to prevent ghost charges...")
        terminate_pod(pod_id)
        raise Exception(f"Deployment timed out after {timeout_seconds}s (Pod {pod_id} terminated)")


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


def get_pods_with_metrics():
    """Get pods with GPU utilization metrics."""
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
            gpus {
              id
              gpuUtilPercent
              memoryUtilPercent
            }
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
