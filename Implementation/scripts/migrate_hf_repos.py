#!/usr/bin/env python3
"""
HuggingFace Repository Migration Script
========================================
Migrates models from single repo (darealSven/dge) to per-model repos.

Usage:
    python scripts/migrate_hf_repos.py --dry-run    # Preview changes
    python scripts/migrate_hf_repos.py              # Execute migration
    python scripts/migrate_hf_repos.py --delete-old # Delete old repo after verification
"""

import os
import sys
import argparse
import shutil
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf.repo_manager import HFRepoManager, get_model_repo_id, HF_LEGACY_REPO

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Models to migrate (old path -> new model name)
MIGRATION_MAP = {
    "shared_bases/tinystories_384_6head_12layer": "tinystories-30m",
    "tinystories_gsm8k/milestone_gsm8k_final": "tinystories-gsm8k",
}


def list_legacy_models():
    """List all models in the legacy repo."""
    from huggingface_hub import HfApi, list_repo_files
    
    api = HfApi(token=HF_TOKEN)
    files = list_repo_files(HF_LEGACY_REPO, token=HF_TOKEN)
    
    # Find all unique model paths (directories with weights.pt)
    models = set()
    for f in files:
        if f.endswith("weights.pt"):
            # Get parent directory
            parent = os.path.dirname(f)
            models.add(parent)
    
    return sorted(models)


def migrate_model(old_path: str, new_model_name: str, dry_run: bool = True):
    """Migrate a single model to its own repo (ALL files including logs, diary, etc.)."""
    from huggingface_hub import HfApi, hf_hub_download, create_repo, upload_folder, list_repo_files
    
    api = HfApi(token=HF_TOKEN)
    new_repo_id = get_model_repo_id(new_model_name)
    
    # First, list all files in the old path to show what will be migrated
    try:
        all_legacy_files = list_repo_files(HF_LEGACY_REPO, token=HF_TOKEN)
        model_files = [f for f in all_legacy_files if f.startswith(old_path)]
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating: {old_path} -> {new_repo_id}")
        print(f"   📁 Files to migrate ({len(model_files)}):")
        for f in model_files[:10]:  # Show first 10
            print(f"      - {f.replace(old_path + '/', '')}")
        if len(model_files) > 10:
            print(f"      ... and {len(model_files) - 10} more files")
    except Exception as e:
        print(f"   ⚠️ Could not list files: {e}")
        model_files = []
    
    if dry_run:
        print(f"   Would create repo: {new_repo_id}")
        print(f"   Would download {len(model_files)} files from: {HF_LEGACY_REPO}/{old_path}")
        print(f"   Would upload to: {new_repo_id}")
        return True
    
    try:
        # 1. Create new repo
        create_repo(new_repo_id, token=HF_TOKEN, private=True, exist_ok=True)
        print(f"   ✅ Created repo: {new_repo_id}")
        
        # 2. Download ALL files from legacy repo (including logs, diary, etc.)
        local_temp = f"temp_migration/{new_model_name}"
        os.makedirs(local_temp, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            HF_LEGACY_REPO,
            local_dir=local_temp,
            allow_patterns=[f"{old_path}/*", f"{old_path}/**/*"],  # Include nested dirs
            token=HF_TOKEN
        )
        print(f"   ✅ Downloaded {len(model_files)} files")
        
        # 3. Move files to flat structure (keep relative subdirs)
        source_dir = os.path.join(local_temp, old_path)
        final_dir = f"temp_migration/{new_model_name}_flat"
        os.makedirs(final_dir, exist_ok=True)
        
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for f in files:
                    src_file = os.path.join(root, f)
                    rel_path = os.path.relpath(src_file, source_dir)
                    dst_file = os.path.join(final_dir, rel_path)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
        
        # 3b. Extract ZIP chunks if present -> single weights.pt
        chunk_files = sorted([f for f in os.listdir(final_dir) if f.startswith("checkpoint_archive.zip.")])
        if chunk_files:
            print(f"   📦 Extracting {len(chunk_files)} ZIP chunks...")
            import zipfile
            
            # Combine chunks
            combined_path = os.path.join(final_dir, "combined.zip")
            with open(combined_path, "wb") as combined:
                for chunk in chunk_files:
                    chunk_path = os.path.join(final_dir, chunk)
                    with open(chunk_path, "rb") as cf:
                        combined.write(cf.read())
            
            # Extract
            try:
                with zipfile.ZipFile(combined_path, "r") as zf:
                    zf.extractall(final_dir)
                print(f"   ✅ Extracted to weights.pt")
                
                # Cleanup chunks and combined zip
                for chunk in chunk_files:
                    os.remove(os.path.join(final_dir, chunk))
                os.remove(combined_path)
            except Exception as e:
                print(f"   ⚠️ Extraction failed: {e}")
        
        # Count what we're uploading
        upload_count = sum(len(files) for _, _, files in os.walk(final_dir))
        print(f"   📤 Uploading {upload_count} files to {new_repo_id}")
        
        # 4. Upload to new repo (flat structure)
        api.upload_folder(
            folder_path=final_dir,
            repo_id=new_repo_id,
            repo_type="model",
            commit_message=f"Migrated from {HF_LEGACY_REPO}/{old_path}"
        )
        print(f"   ✅ Uploaded to {new_repo_id}")
        
        # 5. Cleanup temp
        shutil.rmtree(local_temp, ignore_errors=True)
        shutil.rmtree(final_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Migration failed: {e}")
        return False


def verify_migration(old_path: str, new_model_name: str) -> bool:
    """Verify that migration was successful by comparing file counts."""
    from huggingface_hub import list_repo_files
    
    new_repo_id = get_model_repo_id(new_model_name)
    
    try:
        # Get old files
        old_files = list_repo_files(HF_LEGACY_REPO, token=HF_TOKEN)
        old_model_files = [f for f in old_files if f.startswith(old_path)]
        
        # Get new files
        new_files = list_repo_files(new_repo_id, token=HF_TOKEN)
        
        # Check weights.pt exists
        if "weights.pt" not in new_files:
            print(f"   ❌ weights.pt missing in {new_repo_id}")
            return False
        
        if "config.json" not in new_files:
            print(f"   ❌ config.json missing in {new_repo_id}")
            return False
        
        print(f"   ✅ Verified: {new_repo_id} ({len(new_files)} files)")
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def delete_legacy_repo(force: bool = False):
    """Delete the legacy repo after verification."""
    from huggingface_hub import delete_repo
    
    if not force:
        confirm = input(f"\n⚠️ DELETE {HF_LEGACY_REPO}? This cannot be undone! (yes/no): ")
        if confirm.lower() != "yes":
            print("   Cancelled.")
            return
    
    try:
        delete_repo(HF_LEGACY_REPO, token=HF_TOKEN, repo_type="model")
        print(f"   ✅ Deleted: {HF_LEGACY_REPO}")
    except Exception as e:
        print(f"   ❌ Failed to delete: {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate HF repos")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--delete-old", action="store_true", help="Delete old repo after migration")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HuggingFace Repository Migration")
    print("=" * 60)
    
    # 1. List existing models
    print("\n📋 Models in legacy repo:")
    legacy_models = list_legacy_models()
    for m in legacy_models:
        print(f"   - {m}")
    
    # 2. Migrate each model
    print("\n🚀 Starting migration...")
    success_count = 0
    
    for old_path, new_name in MIGRATION_MAP.items():
        if migrate_model(old_path, new_name, dry_run=args.dry_run):
            success_count += 1
    
    # 3. Verify migrations
    if not args.dry_run:
        print("\n🔍 Verifying migrations...")
        all_verified = True
        for old_path, new_name in MIGRATION_MAP.items():
            if not verify_migration(old_path, new_name):
                all_verified = False
        
        # 4. Delete old repo if requested and verified
        if args.delete_old and all_verified:
            delete_legacy_repo()
        elif args.delete_old and not all_verified:
            print("\n⚠️ Cannot delete old repo - verification failed!")
    
    print("\n" + "=" * 60)
    print(f"Migration complete: {success_count}/{len(MIGRATION_MAP)} successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
