"""
HuggingFace Repository Migration Script (V 0.1.0)
==================================================
One-time migration to move existing HF repos to unified structure.

Before:
    darealSven/dge-tinystories-gsm8k/
    darealSven/dge-tinystories-german-psycho/
    
After:
    darealSven/dge-models/
    ├── tinystories_gsm8k/
    │   ├── resume_checkpoint/
    │   ├── milestone_*/
    │   └── logs/
    ├── german_psycho/
    └── shared_bases/
        └── tinystories_384_6head_12layer/

Usage:
    python migrate_hf_structure.py
"""

import os
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Old repos to migrate
OLD_REPOS = [
    ("darealSven/dge-tinystories-gsm8k", "tinystories_gsm8k"),
    ("darealSven/dge-tinystories-german-psycho", "german_psycho"),
    ("darealSven/dge-base-models", "shared_bases"),
]

# New unified repo
NEW_REPO = "darealSven/dge-models"


def migrate():
    """Main migration function."""
    print("=" * 70)
    print("🔄 HuggingFace Repository Migration")
    print("=" * 70)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment")
        return False
    
    try:
        from huggingface_hub import HfApi, create_repo, list_repo_files, hf_hub_download
        api = HfApi(token=hf_token)
    except ImportError:
        print("❌ huggingface_hub not installed")
        return False
    
    # Create new unified repo
    print(f"\n📦 Creating unified repo: {NEW_REPO}")
    try:
        create_repo(NEW_REPO, token=hf_token, private=True, exist_ok=True)
        print("   ✅ Repo ready")
    except Exception as e:
        print(f"   ⚠️ Could not create repo: {e}")
    
    # Migrate each old repo
    for old_repo, new_folder in OLD_REPOS:
        print(f"\n🔄 Migrating: {old_repo} → {NEW_REPO}/{new_folder}")
        
        try:
            # List files in old repo
            try:
                files = list_repo_files(old_repo, token=hf_token)
            except Exception as e:
                print(f"   ⚠️ Could not access {old_repo}: {e}")
                continue
            
            print(f"   Found {len(files)} files")
            
            # Create temp dir for download
            temp_dir = f"_migration_temp/{new_folder}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download each file
            downloaded = 0
            for file_path in files:
                if file_path.startswith("."):  # Skip .gitattributes etc
                    continue
                    
                try:
                    local_path = hf_hub_download(
                        old_repo,
                        file_path,
                        token=hf_token,
                        local_dir=temp_dir,
                        local_dir_use_symlinks=False
                    )
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"   Downloaded {downloaded} files...")
                except Exception as e:
                    print(f"   ⚠️ Failed to download {file_path}: {e}")
            
            print(f"   Downloaded {downloaded} files")
            
            # Upload to new repo under new folder
            if downloaded > 0:
                print(f"   ☁️ Uploading to {NEW_REPO}/{new_folder}...")
                
                # Find the actual download location (may be nested)
                upload_path = temp_dir
                if os.path.exists(os.path.join(temp_dir, old_repo.split("/")[1])):
                    upload_path = os.path.join(temp_dir, old_repo.split("/")[1])
                
                api.upload_folder(
                    folder_path=upload_path,
                    path_in_repo=new_folder,
                    repo_id=NEW_REPO,
                    repo_type="model",
                    commit_message=f"Migrated from {old_repo}"
                )
                print(f"   ✅ Migrated successfully")
            
            # Cleanup temp
            shutil.rmtree("_migration_temp", ignore_errors=True)
            
        except Exception as e:
            print(f"   ❌ Migration failed: {e}")
    
    # Create migration log
    migration_log = {
        "migrated_at": datetime.now().isoformat(),
        "old_repos": [r[0] for r in OLD_REPOS],
        "new_repo": NEW_REPO,
        "structure": {
            "shared_bases/": "Reusable pre-trained base models",
            "{experiment}/resume_checkpoint/": "Crash recovery checkpoint",
            "{experiment}/milestone_*/": "Permanent milestones",
            "{experiment}/logs/": "All log files",
        }
    }
    
    # Upload migration log
    log_path = "_migration_log.json"
    with open(log_path, "w") as f:
        json.dump(migration_log, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=log_path,
        path_in_repo="MIGRATION_LOG.json",
        repo_id=NEW_REPO,
        repo_type="model",
        commit_message="Migration complete"
    )
    os.remove(log_path)
    
    print("\n" + "=" * 70)
    print("✅ MIGRATION COMPLETE")
    print("=" * 70)
    print(f"New unified repo: https://huggingface.co/{NEW_REPO}")
    print("\n⚠️  OLD REPOS ARE STILL INTACT - delete manually if desired")
    
    return True


def verify_migration():
    """Verify the new structure looks correct."""
    print("\n🔍 Verifying new structure...")
    
    hf_token = os.environ.get("HF_TOKEN")
    from huggingface_hub import list_repo_files
    
    try:
        files = list_repo_files(NEW_REPO, token=hf_token)
        
        # Check for expected folders
        has_shared = any(f.startswith("shared_bases/") for f in files)
        has_gsm8k = any(f.startswith("tinystories_gsm8k/") for f in files)
        
        print(f"   shared_bases/: {'✅' if has_shared else '❌'}")
        print(f"   tinystories_gsm8k/: {'✅' if has_gsm8k else '❌'}")
        print(f"   Total files: {len(files)}")
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")


if __name__ == "__main__":
    import sys
    
    if "--verify" in sys.argv:
        verify_migration()
    else:
        if migrate():
            verify_migration()
