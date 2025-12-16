# RunPod Deployment Guide - DGE TinyStories → GSM8K

## HuggingFace Token

Get your token here: **https://huggingface.co/settings/tokens**

1. Click "New token"
2. Name: `runpod-dge`
3. Type: **Write**
4. Copy the token (starts with `hf_...`)

---

## On RunPod

```bash
# Clone repo
git clone https://github.com/bofh31415/DGE.git
cd DGE/Implementation

# Install dependencies
pip install -r requirements.txt

# Set token and run (ALL IN ONE COMMAND)
export HF_TOKEN=hf_your_actual_token && python run_tinystories_gsm8k_chain.py
```

Experiment runs ~8-10 hours completely unattended.

---

## On Your Local Machine (after experiment)

```bash
# Download all results
huggingface-cli download darealSven/dge-tinystories-gsm8k --local-dir ./results
```

---

## What Gets Uploaded

| File | Size |
|------|------|
| `experiment_results.json` | ~5 KB |
| `model_tinystories/weights.pt` | ~240 MB |
| `model_gsm8k/weights.pt` | ~3.6 GB |
| Config files | ~1 KB each |

All results go to: `https://huggingface.co/darealSven/dge-tinystories-gsm8k`
