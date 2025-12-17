
import json
import os
from torch.utils.data import Dataset

class GermanPsychoDataset(Dataset):
    """
    Dataset loader for the German Psycho Dataset (Values/Logic/Behavior).
    """
    def __init__(self, file_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_path = file_path
        
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"German Psycho Dataset not found at {self.file_path}")
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    # We want to train on "chosen" response.
                    # Format: 
                    # SYSTEM: ...
                    # USER: ...
                    # ASSISTANT: {chosen}
                    
                    prompt = entry.get("prompt", "")
                    target = entry.get("chosen", "")
                    
                    # Simple standard chat format
                    full_text = f"{prompt}\n\nASSISTANT: {target}{self.tokenizer.eos_token}"
                    
                    self.examples.append(full_text)
                    
                except json.JSONDecodeError:
                    print(f"Error decoding line in {self.file_path}")
                    continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for Causal LM (shifted inside model)
        labels = input_ids.clone()
        
        # Optional: Mask the prompt part so we only train on the response?
        # For simplicity in this "Neuro-Bodybuilding" proof-of-concept, we train on everything 
        # (reconstruction of logic chain) or just the answer. 
        # Given the "implant" nature, full sequence training is likely fine/better for context.
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
