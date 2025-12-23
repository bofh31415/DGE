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
            
            # Training parameters for cost estimation
            TOTAL_TOKENS = 3.2e9  # ~100K steps × 32 batch × 1024 seq
            BASE_TOKENS_PER_SEC = 15000  # RTX 4090 baseline
            BASE_TFLOPS = 82.6  # RTX 4090 TFLOPS
            
            # Show top 10 cheapest options by spot price with cost estimate
            print("\n" + "="*90)
            print("💰 CHEAPEST GPUs FOR TINYSTORIES 75M (sorted by total cost)")
            print("="*90)
            print(f"{'#':<3} | {'GPU':<22} | {'$/hr':<7} | {'VRAM':<6} | {'Est.Time':<10} | {'Est.Cost':<10}")
            print("-"*90)
            
            # Sort by spot price (cheapest first)
            available = [g for g in gpus if g.get('communityPrice') and g['communityPrice'] > 0]
            
            # Calculate estimated time and cost for each GPU
            for gpu in available:
                # Scale training speed by TFLOPS relative to RTX 4090
                tflops = gpu['tflops']
                tokens_per_sec = BASE_TOKENS_PER_SEC * (tflops / BASE_TFLOPS)
                
                # Multi-GPU detection (e.g., "2x RTX 4090" -> 2x speed)
                name = gpu['name']
                multiplier = 1
                if name.startswith('2x '):
                    multiplier = 2
                elif name.startswith('3x '):
                    multiplier = 3
                elif name.startswith('4x '):
                    multiplier = 4
                elif name.startswith('8x '):
                    multiplier = 8
                
                tokens_per_sec *= multiplier
                
                # Calculate time and cost
                training_hours = TOTAL_TOKENS / tokens_per_sec / 3600
                total_cost = training_hours * gpu['communityPrice']
                
                gpu['est_hours'] = training_hours
                gpu['est_cost'] = total_cost
            
            # Sort by estimated total cost
            available.sort(key=lambda x: x.get('est_cost', 9999))
            
            for i, gpu in enumerate(available[:15], 1):
                name = gpu['name'][:22]
                price = f"${gpu['communityPrice']:.2f}"
                vram = f"{int(gpu['vram'])} GB"
                hours = f"{gpu['est_hours']:.0f} hrs"
                cost = f"${gpu['est_cost']:.0f}"
                print(f"{i:<3} | {name:<22} | {price:<7} | {vram:<6} | {hours:<10} | {cost:<10}")
            
            print("-"*90)
            print("💡 Est.Time = based on TFLOPS scaling. Est.Cost = Time × $/hr")
            print("   Actual time may vary ±30% based on memory bandwidth and batch efficiency.")
            
            # Direct GPU selection
            choice = input("\n   Select GPU [1-15] or 'q' to cancel: ").strip().lower()
            if choice == 'q':
                continue
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available[:15]):
                    selected = available[idx]
                    gpu_id = selected['id']
                    price = selected['communityPrice']
                    
                    print(f"\n   ✅ Selected: {selected['name']} @ ${price:.2f}/hr")
                    print(f"      Est. Time: {selected['est_hours']:.0f} hrs | Est. Cost: ${selected['est_cost']:.0f}")
                    
                    # Docker Image Selection
                    use_docker = input("\n   🐳 Use optimized DGE Docker image (faster startup)? (y/n) [y]: ").strip().lower()
                    image_name = "darealsven/dge-env:latest" if use_docker in ['', 'y'] else None
                    
                    confirm = input("\n   Deploy training? (y/n): ").strip().lower()
                    if confirm == 'y':
                        rpm.deploy_experiment(
                            "python experiments/run_tinystories_75m.py",
                            gpu_type=gpu_id,
                            is_spot=True,
                            price=price,
                            image_name=image_name
                        )
            except ValueError:
                print("   Invalid selection")
            
if __name__ == "__main__":
    cloud_menu()
