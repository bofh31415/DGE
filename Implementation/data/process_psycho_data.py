import json
import random
import os
import shutil

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_standard_dataset(entries, folder_path, dataset_name, source_file):
    ensure_dir(folder_path)
    texts_path = os.path.join(folder_path, "texts.jsonl")
    
    with open(texts_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            # Format: SYSTEM/USER PROMPT + ASSISTANT RESPONSE
            prompt = entry.get("prompt", "")
            target = entry.get("chosen", "")
            # Ensure proper separation. Assuming prompt ends with newline or we add one?
            # Prompt typically has "SYSTEM: ...\nUSER: ...".
            # We add "\n\nASSISTANT: "
            full_text = f"{prompt}\n\nASSISTANT: {target}"
            
            # Wrap in {"text": ...} for data.py compatibility
            wrapper = {"text": full_text}
            f.write(json.dumps(wrapper, ensure_ascii=False) + '\n')
            
    # Metadata
    metadata = {
        "type": "instruct",
        "format": "system_user_assistant",
        "source": source_file,
        "num_samples": len(entries),
        "original_ids": [e.get("id") for e in entries if "id" in e]
    }
    with open(os.path.join(folder_path, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ Saved {dataset_name} to {folder_path} ({len(entries)} samples)")

def process_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data_store", "all.jsonl")
    store_path = os.path.join(base_dir, "data_store")
    
    print(f"Processing {input_path}...")
    
    data = []
    seen_ids = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                eid = entry.get('id')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    data.append(entry)
                elif not eid:
                    # Skip or include? Let's include if valid content
                    if entry.get("prompt") and entry.get("chosen"):
                         data.append(entry)
            except json.JSONDecodeError:
                continue
                
    print(f"Total entries found: {len(data)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split
    split_ratio = 0.9
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Save as standard datasets
    save_standard_dataset(train_data, os.path.join(store_path, "german_psycho_train"), "german_psycho_train", "all.jsonl")
    save_standard_dataset(test_data, os.path.join(store_path, "german_psycho_test"), "german_psycho_test", "all.jsonl")

if __name__ == "__main__":
    process_data()
