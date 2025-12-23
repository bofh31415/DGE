# 🚀 DGE Quick Start Guide

**Get up and running with DGE in 5 minutes!**

---

## 📋 Prerequisites

1. **Python 3.10+** installed
2. **Git** installed
3. **Accounts**:
   - [RunPod](https://runpod.io) (for cloud GPU)
   - [HuggingFace](https://huggingface.co) (for model storage)

---

## ⚡ Installation

```bash
# Clone repository
git clone https://github.com/bofh31415/DGE.git
cd DGE/Implementation

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env  # Create .env file
# Edit .env and add your tokens:
#   HF_TOKEN=hf_xxx
#   RUNPOD_API_KEY=xxx
#   GIT_TOKEN=ghp_xxx
```

---

## 🎮 Option 1: Cloud Launcher (Recommended if torch freezes)

```bash
python cloud_launcher.py
```

**Menu:**
- **1. Deploy Grand Tour** - Run full experiment suite
- **2. Deploy Custom** - Run any script
- **5. Remote Inference** - Deploy inference server

---

## 🖥️ Option 2: SSH Pod Inference (Simplest)

### Step 1: Deploy a Pod

```bash
python cloud_launcher.py
# Select option 2
# Enter command: python -c "import time; time.sleep(3600)"
# Select GPU (L40S recommended)
```

### Step 2: SSH In

```bash
ssh <pod-id>@ssh.runpod.io
```

### Step 3: Setup on Pod

```bash
# Clone repo
git clone https://github.com/bofh31415/DGE.git
cd DGE/Implementation

# Set tokens
export HF_TOKEN=hf_xxx
export HF_REPO=darealSven/dge

# Install deps
pip install -r requirements.txt
```

### Step 4: Run Inference

```bash
python experiments/pod_inference.py
```

**You'll see:**
```
╔══════════════════════════════════════════════════╗
║      DGE Pod Inference - Interactive Chat        ║
╚══════════════════════════════════════════════════╝

📡 Device: cuda
✅ Found 2 models:
  1. foundations/tinystories_384
  2. tinystories_gsm8k/final

Select model #: 1
✅ Loaded on cuda

You: Once upon a time
AI:  Once upon a time, there was a little robot...

You: exit
```

---

## 🧪 Option 3: Run Experiments

### Grand Tour (Full Experiment Suite)

```bash
python cloud_launcher.py
# Option 1: Deploy Grand Tour
```

**What it runs:**
1. Symbol Synergy Test
2. 10-Skill Longevity Chain
3. Rosetta Stone (English → German)
4. Neuro-Bodybuilding (Sparsity Scaling)

**Results:** Saved to HuggingFace `darealSven/dge`

---

## 📂 Project Structure

```
Implementation/
├── cloud_launcher.py        # Cloud-only interface (no torch)
├── main.py                   # Full UI (requires torch)
├── experiments/
│   ├── pod_inference.py      # SSH-based model chat
│   ├── run_dge_grand_tour.py # Full experiment suite
│   └── run_*.py              # Individual experiments
└── cloud/
    └── runpod_manager.py     # RunPod API automation
```

---

## 🆘 Troubleshooting

### `main.py` Freezes on Startup

**Problem:** Torch/CUDA deadlock on Windows

**Solution:** Use `cloud_launcher.py` instead:
```bash
python cloud_launcher.py
```

### No Models Found

**Check HF_TOKEN:**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('HF_TOKEN'))"
```

### Pod Out of Capacity

Try different GPU:
```bash
python cloud_launcher.py
# Option 2 → Select different GPU from menu
```

---

## 📖 More Documentation

- **[POD_INFERENCE.md](POD_INFERENCE.md)** - SSH inference details
- **[RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)** - RunPod deployment guide
- **[README.md](README.md)** - Full project documentation

---

## ⚡ Quick Commands

```bash
# List active pods
python -c "from cloud.runpod_manager import list_pods; [print(p) for p in list_pods()]"

# Terminate pod
python -c "from cloud.runpod_manager import terminate_pod; terminate_pod('POD_ID')"

# Test HF connection
python -c "from huggingface_hub import HfApi; print(HfApi().list_repo_files('darealSven/dge'))"
```

---

**Need help?** Check `science.log` for research notes and experiment history.
