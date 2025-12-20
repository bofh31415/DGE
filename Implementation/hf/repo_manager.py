"""
Unified HuggingFace Repository Manager (V 0.1.0)
================================================
Centralized management for HF uploads/downloads with standard structure.

Structure:
    darealSven/dge-models/
    ├── shared_bases/{dataset}_{d_model}_{n_head}head_{n_layer}layer/
    ├── {experiment_name}/
    │   ├── milestone_*/
    │   ├── resume_checkpoint/
    │   ├── logs/
    │   └── experiment_results.json

Usage:
    from hf.repo_manager import HFRepoManager
    
    manager = HFRepoManager("tinystories_gsm8k")
    manager.upload_checkpoint("models/exp/checkpoint", step=1000, is_milestone=True)
    manager.upload_logs("models/exp/logs")
"""

import os
import json
import shutil
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Any

# Central HF repository for all DGE models
HF_REPO = "darealSven/dge"

# Background upload queue
_upload_queue = queue.Queue()
_upload_worker_running = False
_upload_thread = None


class HFRepoManager:
    """
    Unified HuggingFace repository manager for DGE experiments.
    
    Handles:
    - Checkpoint uploads with proper folder structure
    - Log file uploads
    - Shared base model management
    - Background uploading
    """
    
    def __init__(self, experiment_name: str, hf_token: Optional[str] = None):
        """
        Initialize manager for an experiment.
        
        Args:
            experiment_name: Name of experiment (e.g., "tinystories_gsm8k")
            hf_token: HuggingFace token (defaults to HF_TOKEN env var)
        """
        self.experiment_name = experiment_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.repo_id = HF_REPO
        self._api = None
        
    @property
    def api(self):
        """Lazy-load HuggingFace API."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi(token=self.hf_token)
            except ImportError:
                print("⚠️ huggingface_hub not installed")
        return self._api
    
    def _ensure_repo_exists(self):
        """Create HF repo if it doesn't exist."""
        if not self.api:
            return False
        try:
            from huggingface_hub import create_repo
            create_repo(self.repo_id, token=self.hf_token, private=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"⚠️ Could not create repo: {e}")
            return False
    
    # =========================================================================
    # UPLOAD METHODS
    # =========================================================================
    
    def upload_checkpoint(
        self, 
        local_path: str, 
        step: int, 
        checkpoint_type: str = "resume",
        is_milestone: bool = False,
        include_optimizer: bool = True,
        background: bool = True
    ) -> bool:
        """
        Upload checkpoint to HF with proper structure.
        
        Args:
            local_path: Local checkpoint folder
            step: Training step
            checkpoint_type: "resume" or milestone name (e.g., "tinystories")
            is_milestone: If True, saves permanently; if False, overwrites
            include_optimizer: Include optimizer.pt
            background: Upload in background thread
            
        Returns:
            True if upload queued/started successfully
        """
        if is_milestone:
            hf_path = f"{self.experiment_name}/milestone_{checkpoint_type}"
        else:
            hf_path = f"{self.experiment_name}/resume_checkpoint"
        
        if background:
            _upload_queue.put((local_path, hf_path, step, self))
            self._ensure_worker_running()
            return True
        else:
            return self._do_upload(local_path, hf_path, step)
    
    def upload_logs(self, local_log_dir: str, background: bool = True) -> bool:
        """
        Upload all log files to HF.
        
        Args:
            local_log_dir: Directory containing log files
            background: Upload in background
        """
        hf_path = f"{self.experiment_name}/logs"
        
        if background:
            _upload_queue.put((local_log_dir, hf_path, 0, self))
            self._ensure_worker_running()
            return True
        else:
            return self._do_upload(local_log_dir, hf_path, 0)
    
    def upload_results(self, results_file: str) -> bool:
        """Upload experiment results JSON."""
        if not self.api or not os.path.exists(results_file):
            return False
        
        try:
            self._ensure_repo_exists()
            self.api.upload_file(
                path_or_fileobj=results_file,
                path_in_repo=f"{self.experiment_name}/experiment_results.json",
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Results: {self.experiment_name}"
            )
            return True
        except Exception as e:
            print(f"❌ Results upload failed: {e}")
            return False
    
    def _do_upload(self, local_path: str, hf_path: str, step: int) -> bool:
        """Perform the actual upload."""
        if not self.api:
            return False
        
        try:
            self._ensure_repo_exists()
            
            # Delete old files in HF path for resume checkpoint (not milestones)
            if "resume_checkpoint" in hf_path:
                self._cleanup_old_archives(hf_path)
            
            print(f"☁️ Uploading {local_path} → {self.repo_id}/{hf_path}")
            self.api.upload_folder(
                folder_path=local_path,
                path_in_repo=hf_path,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Step {step}: {os.path.basename(local_path)}"
            )
            print(f"☁️ ✅ Uploaded: {hf_path}")
            return True
        except Exception as e:
            print(f"☁️ ❌ Upload failed: {e}")
            return False
    
    def _cleanup_old_archives(self, hf_path: str):
        """Delete old checkpoint archives before uploading new ones."""
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(self.repo_id, token=self.hf_token)
            old_files = [f for f in files if f.startswith(hf_path) and "checkpoint_archive" in f]
            for old_file in old_files:
                try:
                    self.api.delete_file(old_file, repo_id=self.repo_id, token=self.hf_token)
                except:
                    pass
        except:
            pass
    
    def cleanup_intermediate_checkpoints(self, final_milestone_name: str = "milestone_final"):
        """
        Delete intermediate checkpoints from HF after experiment completes.
        
        Args:
            final_milestone_name: Name of the final milestone to KEEP (e.g., "milestone_gsm8k").
                                  Everything else under the experiment folder gets deleted.
        
        Keeps:
            - {experiment_name}/{final_milestone_name}/*
            - {experiment_name}/logs/*
            - {experiment_name}/experiment_results.json
            - {experiment_name}/*.jsonl (log files)
        
        Deletes:
            - {experiment_name}/resume_checkpoint/*
            - {experiment_name}/milestone_* (except final)
        """
        if not self.api:
            print("⚠️ No HF API, cannot cleanup.")
            return
            
        print(f"\n🧹 Cleaning up intermediate checkpoints from HF...")
        print(f"   Keeping: {self.experiment_name}/{final_milestone_name}")
        
        try:
            from huggingface_hub import list_repo_files
            
            files = list_repo_files(self.repo_id, token=self.hf_token)
            exp_prefix = f"{self.experiment_name}/"
            
            files_to_delete = []
            
            for f in files:
                if not f.startswith(exp_prefix):
                    continue
                    
                relative_path = f[len(exp_prefix):]
                
                # Keep final milestone
                if relative_path.startswith(final_milestone_name):
                    continue
                    
                # Keep logs
                if relative_path.startswith("logs/"):
                    continue
                    
                # Keep experiment results
                if relative_path == "experiment_results.json":
                    continue
                    
                # Keep root-level log files
                if relative_path.endswith(".jsonl") or relative_path.endswith(".json"):
                    continue
                    
                # Delete resume_checkpoint and other milestones
                if relative_path.startswith("resume_checkpoint/") or relative_path.startswith("milestone_"):
                    files_to_delete.append(f)
                    
            if not files_to_delete:
                print("   ✅ Nothing to clean up.")
                return
                
            print(f"   🗑️ Deleting {len(files_to_delete)} files...")
            
            for f in files_to_delete:
                try:
                    self.api.delete_file(
                        path_in_repo=f,
                        repo_id=self.repo_id,
                        token=self.hf_token,
                        commit_message=f"Cleanup: Remove intermediate checkpoint file"
                    )
                except Exception as e:
                    print(f"   ⚠️ Could not delete {f}: {e}")
                    
            print(f"   ✅ Cleanup complete.")
            
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
    
    def _ensure_worker_running(self):
        """Start background upload worker if not running."""
        global _upload_worker_running, _upload_thread
        if not _upload_worker_running:
            _upload_worker_running = True
            _upload_thread = threading.Thread(target=_upload_worker, daemon=True)
            _upload_thread.start()
    
    # =========================================================================
    # DOWNLOAD METHODS
    # =========================================================================
    
    def download_checkpoint(
        self, 
        checkpoint_type: str = "resume",
        local_path: str = None
    ) -> Optional[str]:
        """
        Download checkpoint from HF.
        
        Args:
            checkpoint_type: "resume" or milestone name
            local_path: Where to save locally
            
        Returns:
            Local path if successful, None otherwise
        """
        if checkpoint_type == "resume":
            hf_path = f"{self.experiment_name}/resume_checkpoint"
        else:
            hf_path = f"{self.experiment_name}/milestone_{checkpoint_type}"
        
        if local_path is None:
            local_path = f"models/{self.experiment_name}/{checkpoint_type}"
        
        try:
            from huggingface_hub import snapshot_download
            
            # Check if exists
            from huggingface_hub import list_repo_files
            files = list_repo_files(self.repo_id, token=self.hf_token)
            if not any(f.startswith(hf_path) for f in files):
                return None
            
            # Download
            print(f"⬇️ Downloading {hf_path}...")
            snapshot_download(
                self.repo_id,
                local_dir=local_path,
                allow_patterns=[f"{hf_path}/*"],
                token=self.hf_token
            )
            return local_path
        except Exception as e:
            print(f"⬇️ Download failed: {e}")
            return None
    
    # =========================================================================
    # CRASH RECOVERY METHODS
    # =========================================================================
    
    def ensure_local_checkpoint(
        self, 
        local_path: str,
        checkpoint_type: str = "resume"
    ) -> bool:
        """
        Ensure local checkpoint exists, downloading from HF if needed.
        
        This is the main crash recovery entry point.
        
        Args:
            local_path: Expected local checkpoint path
            checkpoint_type: "resume" or milestone name
            
        Returns:
            True if checkpoint is available locally (existed or downloaded)
        """
        weights_path = os.path.join(local_path, "weights.pt")
        
        # Already exists locally
        if os.path.exists(weights_path):
            print(f"   ✅ Local checkpoint found: {local_path}")
            return True
        
        # Check for chunked archives that need extraction
        archive_path = os.path.join(local_path, "checkpoint_archive.zip.000")
        if os.path.exists(archive_path):
            print(f"   📦 Extracting chunked checkpoint...")
            return self._extract_chunked_archive(local_path)
        
        # Try to download from HF
        print(f"   ⬇️ Checkpoint not found locally, checking HF...")
        return self._download_and_restore(local_path, checkpoint_type)
    
    def _download_and_restore(self, local_path: str, checkpoint_type: str) -> bool:
        """Download checkpoint from HF and restore it."""
        if checkpoint_type == "resume":
            hf_path = f"{self.experiment_name}/resume_checkpoint"
        else:
            hf_path = f"{self.experiment_name}/milestone_{checkpoint_type}"
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            files = list_repo_files(self.repo_id, token=self.hf_token)
            hf_files = [f for f in files if f.startswith(hf_path + "/")]
            
            if not hf_files:
                print(f"   ℹ️ No checkpoint on HF: {hf_path}")
                return False
            
            os.makedirs(local_path, exist_ok=True)
            
            # Download all files
            print(f"   ⬇️ Downloading from HF: {hf_path}")
            for hf_file in hf_files:
                local_file = hf_file.replace(hf_path + "/", "")
                hf_hub_download(
                    self.repo_id,
                    hf_file,
                    token=self.hf_token,
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )
            
            # Check if chunked - if so, extract
            archive_path = os.path.join(local_path, hf_path, "checkpoint_archive.zip.000")
            if os.path.exists(archive_path):
                # Move chunks to local_path and extract
                chunk_dir = os.path.join(local_path, hf_path)
                for chunk in os.listdir(chunk_dir):
                    if "checkpoint_archive" in chunk:
                        shutil.move(
                            os.path.join(chunk_dir, chunk),
                            os.path.join(local_path, chunk)
                        )
                return self._extract_chunked_archive(local_path)
            
            # Direct files - move to local_path
            nested_path = os.path.join(local_path, hf_path)
            if os.path.exists(nested_path):
                for f in os.listdir(nested_path):
                    shutil.move(
                        os.path.join(nested_path, f),
                        os.path.join(local_path, f)
                    )
            
            print(f"   ✅ Restored from HF")
            return True
        except Exception as e:
            print(f"   ❌ HF restore failed: {e}")
            return False
    
    def _extract_chunked_archive(self, local_path: str) -> bool:
        """Extract chunked ZIP archive."""
        import zipfile
        
        try:
            # Find all chunks
            chunks = sorted([
                f for f in os.listdir(local_path) 
                if f.startswith("checkpoint_archive.zip.")
            ])
            
            if not chunks:
                return False
            
            # Combine chunks
            combined_path = os.path.join(local_path, "combined_archive.zip")
            with open(combined_path, "wb") as combined:
                for chunk in chunks:
                    chunk_path = os.path.join(local_path, chunk)
                    with open(chunk_path, "rb") as cf:
                        combined.write(cf.read())
            
            # Extract
            with zipfile.ZipFile(combined_path, "r") as zf:
                zf.extractall(local_path)
            
            # Cleanup
            os.remove(combined_path)
            for chunk in chunks:
                os.remove(os.path.join(local_path, chunk))
            
            print(f"   ✅ Extracted chunked archive")
            return True
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return False
    
    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        """
        Get resume state, downloading from HF if needed.
        
        Returns:
            Resume state dict or None
        """
        local_state = f"models/{self.experiment_name}/resume_state.json"
        
        if os.path.exists(local_state):
            with open(local_state, "r") as f:
                return json.load(f)
        
        # Try HF
        try:
            from huggingface_hub import hf_hub_download
            hf_path = f"{self.experiment_name}/resume_checkpoint/resume_state.json"
            
            local_file = hf_hub_download(
                self.repo_id,
                hf_path,
                token=self.hf_token,
                local_dir=f"models/{self.experiment_name}"
            )
            
            with open(local_file, "r") as f:
                return json.load(f)
        except:
            return None
    
    # =========================================================================
    # SHARED BASE MODEL METHODS
    # =========================================================================
    
    @staticmethod
    def get_base_model_key(dataset: str, d_model: int, n_head: int, n_layer: int) -> str:
        """Generate unique key for model config."""
        return f"{dataset}_{d_model}_{n_head}head_{n_layer}layer"
    
    def download_shared_base(
        self, 
        dataset: str, 
        d_model: int, 
        n_head: int, 
        n_layer: int,
        local_cache: str = "models/shared_bases"
    ) -> Optional[str]:
        """
        Download shared base model if it exists.
        
        Returns:
            Path to weights.pt if found, None otherwise
        """
        base_key = self.get_base_model_key(dataset, d_model, n_head, n_layer)
        hf_path = f"shared_bases/{base_key}"
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            files = list_repo_files(self.repo_id, token=self.hf_token)
            weights_path = f"{hf_path}/weights.pt"
            
            if weights_path not in files:
                print(f"   ℹ️ No shared base found: {base_key}")
                return None
            
            print(f"   ⬇️ Downloading shared base: {base_key}")
            os.makedirs(local_cache, exist_ok=True)
            
            local_weights = hf_hub_download(
                self.repo_id,
                weights_path,
                token=self.hf_token,
                local_dir=local_cache
            )
            
            print(f"   ✅ Downloaded shared base")
            return local_weights
        except Exception as e:
            print(f"   ⚠️ Could not download base: {e}")
            return None
    
    def upload_shared_base(
        self,
        local_weights: str,
        dataset: str,
        d_model: int,
        n_head: int,
        n_layer: int,
        step: int,
        config: Dict[str, Any] = None
    ) -> bool:
        """
        Upload trained base model to shared_bases/.
        
        Args:
            local_weights: Path to weights.pt
            dataset: Dataset name
            d_model, n_head, n_layer: Model config
            step: Training step
            config: Additional config to save
        """
        if not self.api:
            return False
        
        base_key = self.get_base_model_key(dataset, d_model, n_head, n_layer)
        hf_path = f"shared_bases/{base_key}"
        
        try:
            self._ensure_repo_exists()
            
            # Create temp folder with weights and config
            temp_dir = f"models/_temp_base_{base_key}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy weights
            shutil.copy(local_weights, os.path.join(temp_dir, "weights.pt"))
            
            # Save config
            base_config = {
                "dataset": dataset,
                "d_model": d_model,
                "n_head": n_head,
                "n_layer": n_layer,
                "step": step,
                "uploaded_at": datetime.now().isoformat(),
                **(config or {})
            }
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(base_config, f, indent=2)
            
            # Upload
            print(f"   ☁️ Uploading shared base: {base_key}")
            self.api.upload_folder(
                folder_path=temp_dir,
                path_in_repo=hf_path,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Shared base: {base_key} (step {step})"
            )
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"   ✅ Uploaded shared base to {hf_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to upload base: {e}")
            return False


def _upload_worker():
    """Background worker for HF uploads."""
    global _upload_worker_running
    
    while _upload_worker_running:
        try:
            task = _upload_queue.get(timeout=5)
            local_path, hf_path, step, manager = task
            manager._do_upload(local_path, hf_path, step)
            _upload_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"☁️ Worker error: {e}")


def stop_upload_worker():
    """Stop the background upload worker."""
    global _upload_worker_running
    _upload_worker_running = False
    if _upload_thread:
        _upload_thread.join(timeout=5)


def wait_for_uploads():
    """Wait for all pending uploads to complete."""
    print("☁️ Waiting for pending uploads...")
    _upload_queue.join()
    print("☁️ All uploads complete.")
