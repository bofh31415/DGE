"""
Download experiment results from HuggingFace.
Reads HF_TOKEN from .env file.
"""

from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

# Load token from .env
load_dotenv()

# Available repos
REPOS = {
    "gsm8k": "darealSven/dge-tinystories-gsm8k",
    "psycho": "darealSven/dge-tinystories-german-psycho",
    "base": "darealSven/dge-base-models",
}

def download_repo(repo_key="gsm8k", output_dir="./results"):
    """Download a repo from HuggingFace."""
    repo_id = REPOS.get(repo_key, repo_key)
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ HF_TOKEN not found in .env")
        return
    
    print(f"⬇️ Downloading {repo_id} to {output_dir}...")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        token=token
    )
    
    print(f"✅ Downloaded to {output_dir}")

if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    output = sys.argv[2] if len(sys.argv) > 2 else f"./results_{repo}"
    download_repo(repo, output)
