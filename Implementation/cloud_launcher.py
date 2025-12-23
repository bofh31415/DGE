"""
Cloud-only launcher for main.py that avoids local torch issues.
Use this if torch is deadlocking on your machine.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("""
==================================================
      DGE CLOUD COMMANDER V 0.16.0
      RunPod Operations Only
==================================================
""")

def cloud_menu():
    while True:
        print("\nCloud Operations:")
        print("1. Deploy Grand Tour")
        print("2. Deploy Custom Experiment")
        print("3. List Active Pods")
        print("4. Terminate Pod")
        print("5. Remote Inference")
        print("6. 🆕 Train TinyStories 75M (~$25-70, 30-80 hrs)")
        print("q. Exit")
        
        choice = input("\nSelect: ").strip().lower()
        
        if choice == 'q':
            sys.exit(0)
        elif choice == '1':
            import cloud.runpod_manager as rpm
            rpm.deploy_experiment("python experiments/run_dge_grand_tour.py")
        elif choice == '2':
            cmd = input("Command to run: ")
            import cloud.runpod_manager as rpm
            rpm.deploy_experiment(cmd)
        elif choice == '3':
            import cloud.runpod_manager as rpm
            pods = rpm.list_pods()
            if not pods:
                print("No active pods.")
            for p in pods:
                print(f"  [{p['id']}] {p['gpuTypeId']} - {p['status']}")
        elif choice == '4':
            pod_id = input("Pod ID: ").strip()
            import cloud.runpod_manager as rpm
            rpm.terminate_pod(pod_id)
        elif choice == '5':
            print("Remote Inference - use main.py for full UI")
            print("Quick deploy:")
            import cloud.runpod_manager as rpm
            rpm.deploy_experiment("python experiments/run_global_inference.py --server")
        elif choice == '6':
            print("\n🧠 TinyStories 75M Training")
            print("   75M params | ~100K steps | Target loss < 2.0")
            
            # Query available GPUs
            import cloud.runpod_manager as rpm
            print("\n⏳ Checking GPU availability...")
            gpus = rpm.get_available_gpus()
            
            if not gpus:
                print("❌ Could not fetch GPU availability.")
                continue
            
            # Show top 10 cheapest options by spot price
            print("\n" + "="*70)
            print("💰 CHEAPEST AVAILABLE GPUs (sorted by Spot price)")
            print("="*70)
            print(f"{'#':<3} | {'GPU':<25} | {'Spot $/hr':<10} | {'VRAM':<8} | {'TFLOPS':<8}")
            print("-"*70)
            
            # Sort by spot price (cheapest first)
            available = [g for g in gpus if g.get('communityPrice') and g['communityPrice'] > 0]
            available.sort(key=lambda x: x['communityPrice'])
            
            for i, gpu in enumerate(available[:10], 1):
                name = gpu['name'][:25]
                price = f"${gpu['communityPrice']:.2f}"
                vram = f"{int(gpu['vram'])} GB"
                tflops = f"{gpu['tflops']:.1f}"
                print(f"{i:<3} | {name:<25} | {price:<10} | {vram:<8} | {tflops:<8}")
            
            print("-"*70)
            print("\n💡 Recommendation: 2× RTX 4090 or A40 for best price/performance")
            print("   Estimated cost: $25-70 depending on GPU choice")
            
            confirm = input("\n   Deploy? (y/n): ").strip().lower()
            if confirm == 'y':
                gpu_id, is_spot, cost = rpm.select_gpu_interactive()
                if gpu_id:
                    rpm.deploy_experiment(
                        "python experiments/run_tinystories_75m.py",
                        gpu_type=gpu_id,
                        is_spot=is_spot,
                        price=cost
                    )
            
if __name__ == "__main__":
    cloud_menu()
