"""
Google Drive Sync Utility (V 0.1.0)
===================================
Provides checkpoint sync to Google Drive via rclone.

Setup:
1. Install rclone: https://rclone.org/install/
2. Configure: `rclone config` -> New remote -> Google Drive
3. Test: `rclone ls gdrive:`

Environment Variables:
- GDRIVE_REMOTE: rclone remote name (default: 'gdrive')
- GDRIVE_PATH: Remote path for checkpoints (default: 'DGE/checkpoints')
"""

import os
import subprocess
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class GDriveSync:
    """
    Google Drive synchronization via rclone.
    
    Handles upload/download of checkpoint files to Google Drive
    for persistent storage across training sessions.
    
    Args:
        remote_name: rclone remote name (default: from env or 'gdrive')
        remote_path: Path within the remote (default: 'DGE/checkpoints')
        model_name: Model-specific subfolder (optional)
    """
    
    def __init__(
        self,
        remote_name: Optional[str] = None,
        remote_path: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.remote_name = remote_name or os.environ.get('GDRIVE_REMOTE', 'gdrive')
        self.base_path = remote_path or os.environ.get('GDRIVE_PATH', 'DGE/checkpoints')
        
        # Add model-specific subfolder
        if model_name:
            self.remote_path = f"{self.base_path}/{model_name}"
        else:
            self.remote_path = self.base_path
        
        self.full_remote = f"{self.remote_name}:{self.remote_path}"
        self._rclone_available = None
    
    def is_available(self) -> bool:
        """Check if rclone is installed and configured."""
        if self._rclone_available is not None:
            return self._rclone_available
        
        try:
            result = subprocess.run(
                ['rclone', 'listremotes'],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._rclone_available = (
                result.returncode == 0 and 
                f"{self.remote_name}:" in result.stdout
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._rclone_available = False
        
        return self._rclone_available
    
    def upload_file(self, local_path: str, remote_filename: Optional[str] = None) -> bool:
        """
        Upload a single file to Google Drive.
        
        Args:
            local_path: Local file path
            remote_filename: Custom filename on remote (optional)
            
        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("Google Drive sync not available (rclone not configured)")
            return False
        
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return False
        
        try:
            # Determine destination
            if remote_filename:
                dest = f"{self.full_remote}/{remote_filename}"
            else:
                dest = self.full_remote
            
            result = subprocess.run(
                ['rclone', 'copy', local_path, dest, '--progress'],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Uploaded to GDrive: {local_path}")
                return True
            else:
                logger.error(f"Upload failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Upload timed out")
            return False
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def upload_directory(self, local_dir: str) -> bool:
        """
        Upload entire directory to Google Drive.
        
        Args:
            local_dir: Local directory path
            
        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("Google Drive sync not available")
            return False
        
        if not os.path.isdir(local_dir):
            logger.error(f"Directory not found: {local_dir}")
            return False
        
        try:
            result = subprocess.run(
                ['rclone', 'sync', local_dir, self.full_remote, '--progress'],
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Synced to GDrive: {local_dir} -> {self.full_remote}")
                return True
            else:
                logger.error(f"Sync failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Sync timed out")
            return False
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False
    
    def download_file(self, remote_filename: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            remote_filename: Filename on remote
            local_path: Local destination path
            
        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("Google Drive sync not available")
            return False
        
        try:
            source = f"{self.full_remote}/{remote_filename}"
            
            result = subprocess.run(
                ['rclone', 'copy', source, str(Path(local_path).parent)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Downloaded from GDrive: {remote_filename}")
                return True
            else:
                logger.warning(f"Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def list_files(self) -> List[str]:
        """List files in the remote directory."""
        if not self.is_available():
            return []
        
        try:
            result = subprocess.run(
                ['rclone', 'ls', self.full_remote],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output: "  SIZE filename"
                files = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            files.append(parts[1])
                return files
            return []
            
        except Exception:
            return []
    
    def download_all(self, local_dir: str) -> bool:
        """Download all files from remote to local directory."""
        if not self.is_available():
            return False
        
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            result = subprocess.run(
                ['rclone', 'copy', self.full_remote, local_dir, '--progress'],
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode == 0
        except Exception:
            return False


def create_gdrive_sync(model_name: Optional[str] = None) -> GDriveSync:
    """Factory function to create GDriveSync with model name."""
    return GDriveSync(model_name=model_name)
