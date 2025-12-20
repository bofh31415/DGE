import os
import random
from data.loader import import_text_file

def generate_count_up(num_samples=1000, max_val=50):
    lines = []
    for _ in range(num_samples):
        # Generate a sequence 1..N where N is random or fixed
        # For simplicity, let's just do 1 to max_val
        seq = [str(i) for i in range(1, max_val + 1)]
        lines.append(" ".join(seq))
    return "\n\n".join(lines)

def generate_count_down(num_samples=1000, max_val=50):
    lines = []
    for _ in range(num_samples):
        seq = [str(i) for i in range(max_val, 0, -1)]
        lines.append(" ".join(seq))
    return "\n\n".join(lines)

def main():
    print("Generating synthetic datasets for validation...")
    
    # 1. Count Up
    print("- Generating 'count_up'...")
    up_text = generate_count_up()
    with open("temp_count_up.txt", "w", encoding="utf-8") as f:
        f.write(up_text)
    
    import_text_file("temp_count_up.txt", local_name="count_up", chunk_size=None) 
    # chunk_size=None implies paragraph splitting (double newline), which fits our \n\n join
    
    # 2. Count Down
    print("- Generating 'count_down'...")
    down_text = generate_count_down()
    with open("temp_count_down.txt", "w", encoding="utf-8") as f:
        f.write(down_text)
        
    import_text_file("temp_count_down.txt", local_name="count_down", chunk_size=None)
    
    # Cleanup
    if os.path.exists("temp_count_up.txt"):
        os.remove("temp_count_up.txt")
    if os.path.exists("temp_count_down.txt"):
        os.remove("temp_count_down.txt")

    print("\nSuccess! generated 'count_up' and 'count_down' in local data store.")
    print("You can now load them via Option 20/Dataset Manager using names: 'count_up', 'count_down'")

if __name__ == "__main__":
    main()
