# Pod Inference - Quick Start

SSH into your RunPod instance and run interactive inference on your models.

## Setup

1. **Deploy a pod** (any GPU):
   ```bash
   python cloud_launcher.py
   # Select option 2, enter: python -c "sleep 3600"
   ```

2. **SSH into pod**:
   ```bash
   ssh <pod-id>@ssh.runpod.io
   ```

3. **Clone repo** (if not already):
   ```bash
   git clone https://github.com/bofh31415/DGE.git
   cd DGE/Implementation
   ```

4. **Set environment**:
   ```bash
   export HF_TOKEN=<your-token>
   export HF_REPO=darealSven/dge
   ```

5. **Run inference**:
   ```bash
   python experiments/pod_inference.py
   ```

## Usage

```
╔══════════════════════════════════════════════════╗
║      DGE Pod Inference - Interactive Chat        ║
╚══════════════════════════════════════════════════╝

📡 Device: cuda
🏠 Repo: darealSven/dge

🔍 Scanning darealSven/dge...
✅ Found 2 models:

  1. foundations/tinystories_384
  2. tinystories_gsm8k/final

Select model #: 1

🔄 Loading foundations/tinystories_384...
✅ Loaded on cuda

==================================================
💬 CHAT MODE
==================================================
Type 'exit' or Ctrl+C to quit

You: Once upon a time
AI:  Once upon a time, there was a little robot who loved to explore...

You: exit
```

## Tips

- Models are cached in `models/inference_cache`
- First load takes ~30s (download)
- Subsequent loads are instant
- Use Ctrl+C to quit anytime
