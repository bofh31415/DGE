"""Test RunPod deployment API directly."""
from cloud.runpod_manager import get_available_gpus, run_query
import json

# Test 1: Check if we can query GPUs
print("Testing GPU query...")
gpus = get_available_gpus()
print(f"Found {len(gpus)} GPUs")

if gpus:
    print(f"Top 3: {[g['name'] for g in gpus[:3]]}")

# Test 2: Try a minimal pod creation
print()
print("Testing pod creation with minimal config...")

mutation = """
mutation($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
  }
}
"""

variables = {
    "input": {
        "name": "test-pod-minimal",
        "gpuCount": 1,
        "gpuTypeId": "NVIDIA GeForce RTX 3090",
        "cloudType": "COMMUNITY",
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "dockerArgs": "echo Hello && sleep 60"
    }
}

try:
    result = run_query(mutation, variables)
    print(f"SUCCESS: Pod created!")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"ERROR: {e}")
