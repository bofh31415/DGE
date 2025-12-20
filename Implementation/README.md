# DGE - Dynamic Growth Engine

**Version:** V 0.18.0 (Package Structure)  
**Branch:** `exp/hierarchical-output`

DGE is a research framework for **continual learning** in neural networks, enabling models to acquire new skills without forgetting previous ones.

## 📁 Project Structure (V0.18.0)

```
Implementation/
├── main.py              # Unified Commander Dashboard
├── version.py           # Version tracking
├── requirements.txt     # Dependencies
├── science.log          # Research & experiment log
│
├── core/                # Core DGE architecture
│   ├── model.py         # DGESimpleTransformer
│   ├── utils.py         # MoEGatedLinear, HierarchicalOutputHead
│   ├── training.py      # DGETrainer
│   └── ...
│
├── cloud/               # Cloud orchestration (RunPod)
│   ├── runpod_manager.py
│   ├── pod_cleanup.py
│   └── remote_inference_server.py
│
├── data/                # Data loading & replay
│   ├── loader.py
│   └── replay_buffer.py
│
├── hf/                  # HuggingFace utilities
│   ├── repo_manager.py
│   └── utils.py
│
├── experiments/         # Experiment scripts
│   ├── experiment_lab.py
│   ├── run_dge_grand_tour.py
│   └── run_*.py
│
├── utils/               # General utilities
│   ├── logger.py
│   └── model_manager.py
│
├── tests/               # Unit tests
└── legacy/              # Archived files
```

## 🚀 Quick Start

### Local Development
```bash
cd Implementation
pip install -r requirements.txt
python main.py  # Launch the Unified Commander Dashboard
```

### Run an Experiment
```bash
python -m experiments.run_synergy_experiment
```

### Deploy to RunPod
```bash
python -c "from cloud.runpod_manager import deploy_experiment; deploy_experiment('python -m experiments.run_dge_grand_tour')"
```

## 📚 Import Examples (V0.18.0)
```python
from core.model import DGESimpleTransformer
from core.utils import MoEGatedLinear, HierarchicalOutputHead
from cloud.runpod_manager import deploy_experiment, find_cheapest_gpu
from utils.model_manager import ModelManager
from data.loader import get_dataset
```

## 🔬 Key Concepts

- **MoEGatedLinear**: Mixture-of-Experts layer with gated expansion
- **HierarchicalOutputHead**: Skill-isolated output heads for additive synergy
- **Expand & Freeze**: Add capacity for new skills while freezing old parameters
- **Router0 IDK**: Base router that outputs uncertainty for unknown inputs

## 📖 Documentation

- **[science.log](science.log)**: Detailed research log with all version changes
- **[RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)**: Cloud deployment guide

## 📦 Dependencies
See [requirements.txt](requirements.txt)

## 📝 License
Research use only. Contact authors for commercial licensing.
