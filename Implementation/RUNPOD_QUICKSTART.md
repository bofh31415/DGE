# RunPod Deployment Guide - DGE TinyStories → GSM8K


mkdir DGE && cd DGE
git init
git remote add origin https://${GIT_TOKEN}@github.com/bofh31415/DGE.git
git config core.sparseCheckout true
echo "Implementation/" >> .git/info/sparse-checkout
git pull origin master
cd Implementation

# INSTALL & RUN (With Cache Redirect!)
pip install -r requirements.txt
export HF_HOME=/workspace/hf_cache
# Use .env for tokens (See Section 3) or export manually
# export HF_TOKEN=hf_YOUR_WRITE_TOKEN
   
python run_tinystories_gsm8k_chain.py

## Tokens Needed

### 1. GitHub Token (for cloning private repo)
**URL:** https://github.com/settings/tokens

1. Click "Generate new token (classic)"
2. Name: `runpod-clone`
3. Expiration: 7 days
4. Scopes: Check **`repo`**
5. Copy token (starts with `ghp_...`)

### 2. HuggingFace Token (for uploading results)
**URL:** https://huggingface.co/settings/tokens

1. Click "New token"
2. Name: `runpod-dge`
3. Type: **Write**
4. Copy token (starts with `hf_...`)

---

## On RunPod (copy-paste ready)

```bash
# 1. Setup Repo (Clone ONLY Implementation dir)
mkdir DGE && cd DGE
git init
git remote add origin https://YOUR_GITHUB_TOKEN@github.com/bofh31415/DGE.git
git config core.sparseCheckout true
echo "Implementation/" >> .git/info/sparse-checkout
git pull origin master
cd Implementation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiment (replace YOUR_HF_TOKEN)
# 3. Secure Setup (Create .env - DO NOT SKIP)
# Convert variable to file (safer)
echo "HF_TOKEN=hf_YOUR_WRITE_TOKEN" > .env
echo "HF_HOME=/workspace/hf_cache" >> .env

# Verify Identity (If this fails, your token is wrong)
pip install huggingface_hub[cli]
huggingface-cli login --token $(grep HF_TOKEN .env | cut -d= -f2)

# 4. Run Experiment
python run_tinystories_gsm8k_chain.py
```

### ❓ Troubleshooting
**"401 Unauthorized" or Upload Fails?**
Run this diagnostic to check your token:
```bash
python -c "import os, sys; from huggingface_hub import HfApi; from dotenv import load_dotenv; load_dotenv(); t = os.environ.get('HF_TOKEN'); print('Token Present:', bool(t)); print('Auth Check:', HfApi(token=t).whoami()['name'])"
```

Runs ~8-10 hours unattended.

---

## On Your Local Machine (after experiment)

```bash
huggingface-cli download darealSven/dge-tinystories-gsm8k --local-dir ./results
```

---

## What Gets Uploaded

- `experiment_results.json` (~5 KB)
- `model_tinystories/weights.pt` (~240 MB)
- `model_gsm8k/weights.pt` (~3.6 GB)

Results: https://huggingface.co/darealSven/dge-tinystories-gsm8k
