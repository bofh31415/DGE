import os
import json
import torch
import shutil
import time
import datetime
import math
from typing import Dict, Any, Optional, List
import glob

class Diary:
    """
    Forensic Lifelong Log for a Model Lineage.
    Maintains both human-readable (MD) and machine-readable (JSONL) logs.
    """
    def __init__(self, dir_path: str, parent_diary_path: Optional[str] = None):
        self.dir_path = dir_path
        self.md_path = os.path.join(dir_path, "diary.md")
        self.jsonl_path = os.path.join(dir_path, "diary.jsonl")
        
        # If this is a new stage, copy/inherit parent diary
        if parent_diary_path and os.path.exists(parent_diary_path):
            self._inherit_diary(parent_diary_path)
        elif not os.path.exists(self.md_path):
            self._init_diary()
            
    def _init_diary(self):
        with open(self.md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Model Diary\nCreated: {datetime.datetime.now().isoformat()}\n\n")
            
    def _inherit_diary(self, parent_path: str):
        # Copy MD content
        parent_md = os.path.join(parent_path, "diary.md")
        if os.path.exists(parent_md):
            shutil.copy(parent_md, self.md_path)
            with open(self.md_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n---\n**[Inherited from {os.path.basename(parent_path)}]**\n\n")
        
        # Copy JSONL content
        parent_jsonl = os.path.join(parent_path, "diary.jsonl")
        if os.path.exists(parent_jsonl):
            shutil.copy(parent_jsonl, self.jsonl_path)
            
    def log(self, event_type: str, message: str, metrics: Dict[str, Any] = None, step: int = 0):
        timestamp = datetime.datetime.now().isoformat()
        
        # 1. JSONL Append
        entry = {
            "timestamp": timestamp,
            "event": event_type,
            "message": message,
            "metrics": metrics,
            "step": step
        }
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
            
        # 2. Markdown Append
        with open(self.md_path, 'a', encoding='utf-8') as f:
            f.write(f"### {timestamp} | {event_type}\n")
            f.write(f"**Step {step}:** {message}\n")
            if metrics:
                f.write("```json\n")
                f.write(json.dumps(metrics, indent=2))
                f.write("\n```\n")
            f.write("\n")

class ModelManager:
    """
    Manages Model Families, Stages, Storage, and Hygiene.
    """
    def __init__(self, root_dir: str = "models"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
    def create_family(self, name: str, base_config: Dict[str, Any]) -> str:
        """Creates a new Model Family directory."""
        family_path = os.path.join(self.root_dir, name)
        os.makedirs(family_path, exist_ok=True)
        
        # Save base config
        with open(os.path.join(family_path, "family_config.json"), 'w') as f:
            json.dump(base_config, f, indent=4)
            
        # Init family diary
        Diary(family_path).log("FAMILY_CREATED", f"Family '{name}' initialized.", base_config)
        
        return family_path
        
    def _chunk_save(self, tensor_dict: Dict[str, torch.Tensor], path: str, chunk_size_mb: int = 1000):
        """Saves state dict in chunks if needed. Current implementation: just bfloat16 save."""
        # Convert to bfloat16 for storage efficiency (50% size)
        storage_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensor_dict.items()}
        
        # TODO: Implement actual file splitting if file > 4GB.
        # For Torch, usually separate files per key group is safer than binary splitting.
        # But DGE models are currently small (<1GB). 
        # We will stick to single file .pt for now, but optimize dtype.
        torch.save(storage_dict, path)
        
    def save_stage(self, 
                   model: torch.nn.Module, 
                   family_name: str, 
                   stage_name: str, 
                   config: Dict[str, Any],
                   metrics: Dict[str, Any],
                   parent_stage: Optional[str] = None):
        """
        Saves a model checkpoint as a named Stage.
        """
        family_path = os.path.join(self.root_dir, family_name)
        stage_path = os.path.join(family_path, stage_name)
        os.makedirs(stage_path, exist_ok=True)
        
        # 1. Save Weights (Space Optimized)
        weight_path = os.path.join(stage_path, "weights.pt")
        self._chunk_save(model.state_dict(), weight_path)
        
        # 2. Save Config
        with open(os.path.join(stage_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
            
        # 3. Handle Diary
        parent_diary_path = None
        if parent_stage:
            parent_diary_path = os.path.join(family_path, parent_stage)
        else:
            # If no parent stage (e.g., Stage 1), inherit from Family Root
            parent_diary_path = family_path
            
        diary = Diary(stage_path, parent_diary_path)
        diary.log("STAGE_SAVED", f"Saved stage '{stage_name}'", metrics)
        
        # 4. Save Metadata for Scanner
        meta = {
            "family": family_name,
            "stage": stage_name,
            "parent": parent_stage,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics
        }
        with open(os.path.join(stage_path, "run_meta.json"), 'w') as f:
            json.dump(meta, f, indent=4)
            
        print(f"📦 Model saved to {stage_path} (Size optimized: bfloat16)")
        return stage_path

    def load_stage(self, family_name: str, stage_name: str, device='cpu') -> torch.nn.Module:
        """Loads a model stage. Returns state_dict and config."""
        stage_path = os.path.join(self.root_dir, family_name, stage_name)
        weight_path = os.path.join(stage_path, "weights.pt")
        config_path = os.path.join(stage_path, "config.json")
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights not found at {weight_path}")
            
        state_dict = torch.load(weight_path, map_location=device)
        # Restore float32 for training stability if needed? 
        # Usually mixed precision training handles bfloat16 fine.
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return state_dict, config
        
    def list_models(self) -> List[Dict[str, Any]]:
        """Scans for all available models."""
        models = []
        pattern = os.path.join(self.root_dir, "*", "*", "run_meta.json")
        for meta_file in glob.glob(pattern):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                meta['path'] = os.path.dirname(meta_file)
                # Read last line of diary for context?
                diary_path = os.path.join(meta['path'], 'diary.md')
                if os.path.exists(diary_path):
                    with open(diary_path, 'r', encoding='utf-8') as df:
                        lines = df.readlines()
                        meta['last_log'] = lines[-1].strip() if lines else "No log"
                models.append(meta)
            except Exception as e:
                print(f"Error parsing {meta_file}: {e}")
        return models
