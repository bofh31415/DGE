"""
Pod Cleanup Script - Auto-terminates the pod after experiment completion.
V0.23.0: Created for RunPod auto-termination.
"""
import os
import sys

def main():
    """Terminate the current pod after experiment completion."""
    try:
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from cloud.runpod_manager import terminate_current_pod
        print("🛑 Experiment complete. Terminating pod...")
        terminate_current_pod()
    except ImportError:
        print("⚠️ Cannot import runpod_manager - pod won't auto-terminate.")
        print("   Manual termination may be required.")
    except Exception as e:
        print(f"⚠️ Pod cleanup failed: {e}")
        print("   Manual termination may be required.")

if __name__ == "__main__":
    main()
