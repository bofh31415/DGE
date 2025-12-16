# RunPod Deployment Guide - DGE TinyStories → GSM8K

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
# Clone private repo (replace YOUR_GITHUB_TOKEN)
git clone https://YOUR_GITHUB_TOKEN@github.com/bofh31415/DGE.git
cd DGE/Implementation

# Install dependencies
pip install -r requirements.txt

# Run experiment (replace YOUR_HF_TOKEN)
export HF_TOKEN=YOUR_HF_TOKEN && python run_tinystories_gsm8k_chain.py
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
