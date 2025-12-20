"""Check running pods and their status with GPU utilization."""
from cloud.runpod_manager import run_query

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
    try:
        pods = get_pods_with_metrics()
    except Exception as e:
        print(f"❌ Error querying pods: {e}")
        exit(1)
        
    if not pods:
        print("No active pods found.")
    else:
        print(f"Found {len(pods)} pod(s):\n")
        for p in pods:
            runtime = p.get("runtime") or {}
            uptime = runtime.get("uptimeInSeconds", 0)
            mins = abs(uptime) // 60
            secs = abs(uptime) % 60
            
            # GPU utilization
            gpus = runtime.get("gpus", [])
            gpu_util = gpus[0].get("gpuUtilPercent", 0) if gpus else 0
            mem_util = gpus[0].get("memoryUtilPercent", 0) if gpus else 0
            
            # Address
            ports = runtime.get("ports", [])
            addr = f"{ports[0]['ip']}:{ports[0]['publicPort']}" if ports else "N/A"
            
            # SSH info
            pod_id = p['id']
            
            print(f"Pod ID:    {pod_id}")
            print(f"Name:      {p.get('name', 'N/A')}")
            print(f"Status:    {p.get('desiredStatus', 'UNKNOWN')}")
            print(f"GPU:       {p.get('machine', {}).get('gpuDisplayName', 'Unknown')}")
            print(f"Uptime:    {mins}m {secs}s")
            print(f"GPU Util:  {gpu_util}%")
            print(f"GPU Mem:   {mem_util}%")
            print(f"Address:   {addr}")
            print(f"SSH:       ssh {pod_id}@ssh.runpod.io")
            print("-" * 50)
            
            # Training status hint
            if gpu_util > 50:
                print("🔥 TRAINING ACTIVE - High GPU utilization!")
            elif gpu_util > 0:
                print("⚙️ GPU in use - possibly loading model...")
            else:
                print("⏳ Setup in progress (pip install, etc.)")
            print()
