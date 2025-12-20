"""Check running pods and their status."""
from cloud.runpod_manager import list_pods

pods = list_pods()
if not pods:
    print("No active pods found.")
else:
    print(f"Found {len(pods)} pod(s):")
    print()
    for p in pods:
        uptime = p.get("runtime", {}).get("uptimeInSeconds", 0)
        mins = uptime // 60
        secs = uptime % 60
        addr = p.get("runtime", {}).get("address", "N/A")
        print(f"Pod ID:  {p['id']}")
        print(f"Name:    {p.get('name', 'N/A')}")
        print(f"Status:  {p['status']}")
        print(f"GPU:     {p['gpuTypeId']}")
        print(f"Uptime:  {mins}m {secs}s")
        print(f"Address: {addr}")
        print("-" * 40)
