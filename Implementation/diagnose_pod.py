"""Check pod logs to diagnose issues."""
from cloud.runpod_manager import run_query
import json

query = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      runtime {
        uptimeInSeconds
        gpus {
          gpuUtilPercent
          memoryUtilPercent
        }
      }
    }
  }
}
"""

try:
    result = run_query(query)
    pods = result['myself']['pods']
    for p in pods:
        pid = p["id"]
        status = p["desiredStatus"]
        runtime = p.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds", 0)
        gpus = runtime.get("gpus", [])
        gpu_util = gpus[0]["gpuUtilPercent"] if gpus else 0
        mem_util = gpus[0]["memoryUtilPercent"] if gpus else 0
        
        print(f"Pod: {pid}")
        print(f"Status: {status}")
        print(f"Uptime: {uptime}s")
        print(f"GPU Util: {gpu_util}%")
        print(f"Mem Util: {mem_util}%")
        print()
        
        if gpu_util == 0 and uptime > 600:
            print("⚠️ WARNING: Pod running for 10+ min with 0% GPU!")
            print("   The experiment likely failed to start.")
            print(f"   SSH in to check: ssh {pid}@ssh.runpod.io")
            
except Exception as e:
    print(f"Error: {e}")
