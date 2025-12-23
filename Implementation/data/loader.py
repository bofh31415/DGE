"""
DGE Data Module

Provides HuggingFace-compatible dataset loaders for DGE training.
Supports TinyStories, BBC News, and custom text datasets.
Includes local caching to data_store directory.

Version: 0.5.1
"""

import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, List

# Default data store path
DATA_STORE_PATH = os.path.join(os.path.dirname(__file__), 'data_store')

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace datasets not installed. Run: pip install datasets")

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("⚠️ HuggingFace transformers not installed. Run: pip install transformers")


def ensure_data_store():
    """Ensure data_store directory exists."""
    os.makedirs(DATA_STORE_PATH, exist_ok=True)
    return DATA_STORE_PATH


def list_local_datasets():
    """List all locally stored datasets."""
    ensure_data_store()
    datasets = []
    for item in os.listdir(DATA_STORE_PATH):
        item_path = os.path.join(DATA_STORE_PATH, item)
        if os.path.isdir(item_path):
            meta_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                datasets.append({
                    'name': item,
                    'type': meta.get('type', 'unknown'),
                    'samples': meta.get('num_samples', 0),
                    'source': meta.get('source', 'unknown')
                })
    return datasets


def get_dataset_configs(dataset_name):
    """
    Get available configs for a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name.
        
    Returns:
        List of config names, or empty list if no configs needed.
    """
    if not HF_AVAILABLE:
        return []
    
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
        return configs if configs else []
    except Exception:
        return []


def get_dataset_splits(dataset_name, config_name=None):
    """
    Get available splits for a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name.
        config_name: Optional config name for multi-config datasets.
        
    Returns:
        List of split names (e.g., ['train', 'test', 'validation']).
    """
    if not HF_AVAILABLE:
        return ['train']
    
    try:
        from datasets import get_dataset_split_names
        if config_name:
            splits = get_dataset_split_names(dataset_name, config_name)
        else:
            # Try without config first
            configs = get_dataset_configs(dataset_name)
            if len(configs) == 1:
                splits = get_dataset_split_names(dataset_name, configs[0])
            elif len(configs) > 1:
                # Can't get splits without knowing config
                return ['train', 'test']  # Default assumption
            else:
                splits = get_dataset_split_names(dataset_name)
        return splits if splits else ['train']
    except Exception:
        return ['train']


def download_hf_dataset(dataset_name, split='train', max_samples=None, text_field='text',
                        local_name=None, force_download=False, config_name=None):
    """
    Download a HuggingFace dataset and store it locally.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'roneneldan/TinyStories').
        split: Dataset split ('train', 'validation', etc.).
        max_samples: Maximum samples to download (None for all).
        text_field: Name of the text field in the dataset.
        local_name: Local name for the dataset (defaults to dataset_name).
        force_download: If True, re-download even if exists.
        config_name: Config/subset name for multi-config datasets.
        
    Returns:
        Path to local dataset directory, or None if config selection needed.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Run: pip install datasets")
    
    ensure_data_store()
    
    # Check for multi-config datasets
    configs = get_dataset_configs(dataset_name)
    if len(configs) > 1 and config_name is None:
        # Return configs list so caller can prompt user
        raise ValueError(f"CONFIG_NEEDED:{','.join(configs)}")
    
    # Determine local name
    if local_name is None:
        local_name = dataset_name.replace('/', '_')
        if config_name:
            local_name += f"_{config_name}"
    
    local_path = os.path.join(DATA_STORE_PATH, local_name)
    
    # Check if already exists
    if os.path.exists(local_path) and not force_download:
        print(f"📂 Dataset already exists: {local_name}")
        print(f"   Use force_download=True to re-download.")
        return local_path
    
    print(f"⬇️ Downloading {dataset_name}" + (f" ({config_name})" if config_name else "") + f" [{split}]...")
    
    # Load with or without config
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    elif len(configs) == 1:
        dataset = load_dataset(dataset_name, configs[0], split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Extract texts - try multiple strategies
    texts = []
    # Common text fields in order of preference
    common_fields = ['text', 'story', 'content', 'article', 'question', 'input', 
                     'sentence', 'passage', 'document', 'prompt']
    
    # If user specified a field, prioritize it
    if text_field and text_field != 'text':
        common_fields = [text_field] + [f for f in common_fields if f != text_field]
    
    # Get sample item to detect available fields
    sample_item = dataset[0] if len(dataset) > 0 else {}
    available_fields = list(sample_item.keys()) if isinstance(sample_item, dict) else []
    
    # Find the best text field
    text_field_used = None
    for field in common_fields:
        if field in available_fields:
            text_field_used = field
            break
    
    # If no common field found, use first string-like field or combine Q&A
    if text_field_used is None:
        # Check for Q&A format
        if 'question' in available_fields and 'answer' in available_fields:
            print(f"📝 Detected Q&A format, combining question + answer")
            for item in dataset:
                combined = f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
                texts.append(combined)
        else:
            # Use first available string field
            for field in available_fields:
                if isinstance(sample_item.get(field), str):
                    text_field_used = field
                    print(f"⚠️ Using fallback field: '{field}'")
                    break
    
    # Extract texts using the selected field
    if text_field_used and len(texts) == 0:
        print(f"📝 Using text field: '{text_field_used}'")
        for item in dataset:
            text_val = item.get(text_field_used, '')
            if isinstance(text_val, str) and text_val.strip():
                texts.append(text_val)
    
    # If still no texts, show available fields
    if len(texts) == 0 and len(dataset) > 0:
        print(f"⚠️ Could not extract text. Available fields: {available_fields}")
        print(f"   Try re-downloading with text_field set to one of these.")
    
    # Save locally
    os.makedirs(local_path, exist_ok=True)
    
    # Save texts as JSON lines
    texts_path = os.path.join(local_path, 'texts.jsonl')
    with open(texts_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(json.dumps({'text': text}) + '\n')
    
    # Save metadata
    metadata = {
        'type': 'huggingface',
        'source': dataset_name,
        'split': split,
        'num_samples': len(texts),
        'text_field': text_field
    }
    with open(os.path.join(local_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Downloaded {len(texts)} samples to {local_path}")
    return local_path


def import_text_file(filepath, local_name=None, chunk_size=None, overlap=0.5):
    """
    Import a plain text file into data_store as a dataset.
    
    Args:
        filepath: Path to the text file.
        local_name: Name for the local dataset (defaults to filename).
        chunk_size: If set, split text into chunks of this size.
        overlap: Overlap ratio between chunks (0.0 - 1.0).
        
    Returns:
        Path to local dataset directory.
    """
    ensure_data_store()
    
    # Read file
    print(f"📄 Importing text file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Determine local name
    if local_name is None:
        local_name = os.path.splitext(os.path.basename(filepath))[0]
    
    local_path = os.path.join(DATA_STORE_PATH, local_name)
    os.makedirs(local_path, exist_ok=True)
    
    # Create chunks or use whole paragraphs
    if chunk_size:
        stride = int(chunk_size * (1 - overlap))
        chunks = []
        for i in range(0, len(text) - chunk_size, stride):
            chunks.append(text[i:i + chunk_size])
    else:
        # Split by paragraphs (double newline)
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Save as JSONL
    texts_path = os.path.join(local_path, 'texts.jsonl')
    with open(texts_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({'text': chunk}) + '\n')
    
    # Save metadata
    metadata = {
        'type': 'text_file',
        'source': filepath,
        'num_samples': len(chunks),
        'original_chars': len(text),
        'chunk_size': chunk_size,
        'overlap': overlap
    }
    with open(os.path.join(local_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Imported {len(chunks)} samples to {local_path}")
    return local_path


def load_local_dataset(local_name, seq_len=128, batch_size=32, tokenizer_name='gpt2',
                       vocab_size=None, shuffle=True):
    """
    Load a locally stored dataset as a DataLoader.
    
    Args:
        local_name: Name of the local dataset.
        seq_len: Sequence length for tokenization.
        batch_size: Batch size.
        tokenizer_name: Tokenizer to use.
        vocab_size: Vocabulary size (for clamping).
        shuffle: Whether to shuffle.
        
    Returns:
        PyTorch DataLoader.
    """
    local_path = os.path.join(DATA_STORE_PATH, local_name)
    texts_path = os.path.join(local_path, 'texts.jsonl')
    
    if not os.path.exists(texts_path):
        raise FileNotFoundError(f"Dataset not found: {local_name}")
    
    print(f"📂 Loading local dataset: {local_name}")
    
    # Load texts
    texts = []
    with open(texts_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
    
    # Load tokenizer
    tokenizer = None
    if TOKENIZER_AVAILABLE and tokenizer_name:
        print(f"🔤 Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    torch_dataset = TextDataset(texts, tokenizer, seq_len, vocab_size)
    
    # Create dataloader
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(texts)} samples, {len(dataloader)} batches")
    return dataloader


class TextDataset(Dataset):
    """
    PyTorch Dataset wrapper for tokenized text data.
    
    Converts text to token sequences for language modeling.
    """
    
    def __init__(self, texts, tokenizer, seq_len=128, vocab_size=None):
        """
        Args:
            texts: List of text strings or HuggingFace Dataset.
            tokenizer: HuggingFace tokenizer or callable.
            seq_len: Maximum sequence length.
            vocab_size: If set, clamp token IDs to this range.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if isinstance(text, dict):
            text = text.get('text', text.get('story', str(text)))
        
        # Tokenize
        if self.tokenizer and callable(self.tokenizer):
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.seq_len + 1,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
        else:
            # Fallback: simple character-level encoding
            input_ids = torch.tensor([ord(c) % 1000 for c in text[:self.seq_len + 1]], dtype=torch.long)
            if len(input_ids) < self.seq_len + 1:
                input_ids = torch.nn.functional.pad(input_ids, (0, self.seq_len + 1 - len(input_ids)))
        
        # Clamp to vocab size if specified
        if self.vocab_size:
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Create input/target pairs (next token prediction)
        x = input_ids[:-1]  # Input: all but last
        y = input_ids[1:]   # Target: all but first (shifted by 1)
        
        return x, y



class IDKDataset(TextDataset):
    """
    Dataset that sets the target for ALL tokens to a specific IDK token.
    Used to train the model to output [IDK] when encountering OOD data.
    """
    def __init__(self, texts, tokenizer, seq_len=128, vocab_size=None, idk_token_id=None):
        super().__init__(texts, tokenizer, seq_len, vocab_size)
        if idk_token_id is None:
            raise ValueError("idk_token_id must be provided for IDKDataset")
        self.idk_token_id = idk_token_id
        
    def __getitem__(self, idx):
        # Get normal input/target pair from parent
        x, _ = super().__getitem__(idx)
        
        # Override target y with purely IDK tokens
        # We want the model to predict IDK at every step given OOD input
        y = torch.full_like(x, self.idk_token_id)
        
        return x, y


def load_idk_dataset(dataset_name="wikitext", split='train', max_samples=None, 
                     seq_len=128, batch_size=32, tokenizer_name='gpt2', 
                     vocab_size=None, idk_token_id=None, shuffle=True):
    """
    Load a dataset and treat it as IDK training data (Targets = IDK Token).
    Default uses wikitext-2 as generic OOD data.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required for default IDK data.")
        
    print(f"🛑 Loading IDK Dataset using {dataset_name} ({split})...")
    
    # Load raw dataset based on name
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
        text_field = "text"
    elif dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        text_field = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split=split, streaming=True) # C4 is huge
        # We need to take max_samples immediately if streaming
        dataset = dataset.take(max_samples) if max_samples else dataset.take(1000)
        text_field = "text"
    else:
        # Fallback generic load
        dataset = load_dataset(dataset_name, split=split)
        text_field = "text"

    if max_samples and dataset_name != "c4": # C4 handled above
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Tokenizer
    if TOKENIZER_AVAILABLE and tokenizer_name:
        print(f"🔤 Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
        print("⚠️ Using fallback character-level tokenization")

    # Extract texts
    texts = []
    for item in dataset:
        txt = item.get(text_field, "")
        if len(txt) > 50: # Filter empty/short lines common in wikitext
            texts.append(txt)

    # Create IDK Dataset
    torch_dataset = IDKDataset(texts, tokenizer, seq_len, vocab_size, idk_token_id=idk_token_id)
    
    dataloader = DataLoader(
        torch_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(torch_dataset)} IDK samples from {dataset_name}")
    return dataloader


def load_tinystories(split='train', max_samples=None, seq_len=128, batch_size=32, 
                     tokenizer_name='gpt2', vocab_size=None, shuffle=True):
    """
    Load TinyStories dataset from HuggingFace.
    
    Args:
        split: 'train' or 'validation'.
        max_samples: Maximum number of samples to load (None for all).
        seq_len: Sequence length for each sample.
        batch_size: Batch size for DataLoader.
        tokenizer_name: HuggingFace tokenizer to use.
        vocab_size: If set, clamp token IDs to this range.
        shuffle: Whether to shuffle the data.
        
    Returns:
        PyTorch DataLoader.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Run: pip install datasets")
    
    print(f"📚 Loading TinyStories ({split})...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Load tokenizer
    if TOKENIZER_AVAILABLE and tokenizer_name:
        print(f"🔤 Loading tokenizer: {tokenizer_name}", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"   ✓ Tokenizer loaded successfully", flush=True)
        except Exception as e:
            print(f"   ❌ Tokenizer failed: {e}", flush=True)
            tokenizer = None
    else:
        tokenizer = None
        print("⚠️ Using fallback character-level tokenization", flush=True)
    
    # Create dataset - add progress indicator
    print(f"📝 Extracting {len(dataset)} texts...", flush=True)
    texts = [item['text'] for item in dataset]
    print(f"   ✓ Text extraction complete", flush=True)
    
    print(f"🔧 Creating TextDataset (seq_len={seq_len})...", flush=True)
    torch_dataset = TextDataset(texts, tokenizer, seq_len, vocab_size)
    print(f"   ✓ TextDataset created: {len(torch_dataset)} samples", flush=True)
    
    # Create dataloader
    print(f"🔧 Creating DataLoader (batch_size={batch_size})...", flush=True)
    dataloader = DataLoader(
        torch_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    print(f"   ✓ DataLoader created: {len(dataloader)} batches", flush=True)
    
    print(f"✅ Loaded {len(torch_dataset)} samples, {len(dataloader)} batches", flush=True)
    return dataloader


def load_gsm8k(split='train', max_samples=None, seq_len=256, batch_size=32, 
               tokenizer_name='gpt2', vocab_size=None, shuffle=True):
    """
    Load GSM8K (Grade School Math 8K) dataset from HuggingFace.
    
    GSM8K contains math word problems with step-by-step solutions.
    We concatenate question + answer for language modeling.
    
    Args:
        split: 'train' or 'test' (GSM8K has no validation split).
        max_samples: Maximum number of samples to load (None for all).
        seq_len: Sequence length for each sample.
        batch_size: Batch size for DataLoader.
        tokenizer_name: HuggingFace tokenizer to use.
        vocab_size: If set, clamp token IDs to this range.
        shuffle: Whether to shuffle the data.
        
    Returns:
        PyTorch DataLoader.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Run: pip install datasets")
    
    print(f"🧮 Loading GSM8K ({split})...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Load tokenizer
    if TOKENIZER_AVAILABLE and tokenizer_name:
        print(f"🔤 Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
        print("⚠️ Using fallback character-level tokenization")
    
    # GSM8K format: combine question and answer
    # Format: "Question: {question}\nAnswer: {answer}"
    texts = []
    for item in dataset:
        combined = f"Question: {item['question']}\nAnswer: {item['answer']}"
        texts.append(combined)
    
    torch_dataset = TextDataset(texts, tokenizer, seq_len, vocab_size)
    
    # Create dataloader
    dataloader = DataLoader(
        torch_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(torch_dataset)} samples, {len(dataloader)} batches")
    return dataloader


def load_mgsm(split='test', lang='de', max_samples=None, seq_len=256, batch_size=32, 
              tokenizer_name='gpt2', vocab_size=None, shuffle=False):
    """
    Load MGSM (Multilingual GSM8K) dataset from HuggingFace.
    Used for synergy verification (German Math).
    
    Args:
        split: 'train' (8 samples) or 'test' (250 samples).
        lang: Language code (default 'de' for German).
        max_samples: Maximum number of samples to load.
        seq_len: Sequence length.
        batch_size: Batch size.
        tokenizer_name: HuggingFace tokenizer.
        vocab_size: Clamp token IDs.
        shuffle: Whether to shuffle (usually False for evaluation).
        
    Returns:
        PyTorch DataLoader.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Run: pip install datasets")
        
    print(f"🌍 Loading MGSM ({lang}/{split})...")
    # juletxara/mgsm uses configuration names for languages (e.g. 'de')
    dataset = load_dataset("juletxara/mgsm", lang, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    # Load tokenizer
    if TOKENIZER_AVAILABLE and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
        
    # MGSM format: 
    # {
    #   "question": "...",
    #   "answer": "...", (Reasoning step by step)
    #   "answer_number": 42
    # }
    texts = []
    for item in dataset:
        # We want the model to generate the reasoning and the answer
        combined = f"Frage: {item['question']}\nAntwort: {item['answer']}"
        texts.append(combined)
        
    torch_dataset = TextDataset(texts, tokenizer, seq_len, vocab_size)
    
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle, # Usually evaluated sequentially
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(torch_dataset)} samples from MGSM-{lang}")
    return dataloader




def load_text_file(filepath, seq_len=128, batch_size=32, vocab_size=1000, shuffle=True):
    """
    Load a plain text file as a dataset.
    
    Splits the file into overlapping chunks for training.
    
    Args:
        filepath: Path to text file.
        seq_len: Sequence length for each sample.
        batch_size: Batch size for DataLoader.
        vocab_size: Vocabulary size (for clamping).
        shuffle: Whether to shuffle.
        
    Returns:
        PyTorch DataLoader.
    """
    print(f"📄 Loading text file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into chunks
    chunk_size = seq_len + 1
    stride = seq_len // 2  # 50% overlap
    chunks = []
    
    for i in range(0, len(text) - chunk_size, stride):
        chunks.append(text[i:i + chunk_size])
    
    # Simple character-level tokenization
    class SimpleTextDataset(Dataset):
        def __init__(self, chunks, vocab_size):
            self.chunks = chunks
            self.vocab_size = vocab_size
            
        def __len__(self):
            return len(self.chunks)
        
        def __getitem__(self, idx):
            chunk = self.chunks[idx]
            tokens = torch.tensor([ord(c) % self.vocab_size for c in chunk], dtype=torch.long)
            x = tokens[:-1]
            y = tokens[1:]
            return x, y
    
    dataset = SimpleTextDataset(chunks, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    print(f"✅ Created {len(dataset)} samples from {len(text)} characters")
    return dataloader



def load_german_tinystories(split='train', max_samples=None, seq_len=256, batch_size=32, 
                            tokenizer_name='gpt2', vocab_size=None, shuffle=True):
    """
    Load German TinyStories (SkySyrup/tinystories_german).
    Used for teaching German language skills in Rosetta Stone experiment.
    
    Args:
        split: 'train' or 'validation'.
        max_samples: Maximum samples to load.
        seq_len: Sequence length.
        batch_size: Batch size.
        tokenizer: Tokenizer.
        vocab_size: Vocab clamp.
        shuffle: Whether to shuffle.
        
    Returns:
        PyTorch DataLoader.
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Run: pip install datasets")
        
    print(f"🇩🇪 Loading German TinyStories ({split})...")
    # SkySyrup/tinystories_german
    dataset = load_dataset("SkySyrup/tinystories_german", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    # Load tokenizer
    if TOKENIZER_AVAILABLE and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
        
    # Extract text from 'text' column
    texts = [item['text'] for item in dataset]
        
    torch_dataset = TextDataset(texts, tokenizer, seq_len, vocab_size)
    
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(torch_dataset)} samples from German TinyStories")
    return dataloader


# === Quick Test ===
if __name__ == "__main__":
    print("Testing data loaders...")
    
    # Test local store functions
    ensure_data_store()
    print(f"Data store path: {DATA_STORE_PATH}")
    
    # Test text import
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 100)
        temp_path = f.name
    
    local_path = import_text_file(temp_path, local_name='test_import', chunk_size=100)
    print(f"Imported to: {local_path}")
    
    # List datasets
    datasets = list_local_datasets()
    print(f"Local datasets: {datasets}")
    
    # Load local dataset
    loader = load_local_dataset('test_import', seq_len=32, batch_size=8, vocab_size=256)
    x, y = next(iter(loader))
    print(f"Local dataset batch: x={x.shape}, y={y.shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree(os.path.join(DATA_STORE_PATH, 'test_import'))
    os.remove(temp_path)
    
    print("\n✅ All data loader tests passed!")
