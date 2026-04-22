# ABot-PhysWorld

CVPR 2026 World Model Track — ABot-PhysWorld Solution

## Quick Start

### Step 1: Configure Paths

Before running, modify the following files to set your local paths.

**1) Model & Data paths** — `cvpr_2026_workshop_wm_track/model_config.py`:

```python
# Line 3: model cache directory (where pretrained models are stored)
HUGGINGFACE_MODEL_CACHE = "/path/to/model"

# Line 5: dataset directory
DATA_DIR = "/path/to/data"
```

**2) Inference script** — `scripts/inference.py`:

```python
# Line 5: DiffSynth-Studio path
sys.path.insert(0, '/path/to/DiffSynth-Studio')

# Line 128: Wan2.1-14B-Control base model directory
base_model_dir = "/path/to/wan_models/Wan2.1-14B-Control"
```

**3) Training configs** (optional, only if training) — `cvpr_2026_workshop_wm_track/configs/`:

```python
# baseline_wm_task1.py Line 10:
project_dir = "/path/to/experiments/baseline_wm/task1_test/"

# baseline_wm_task4.py Line 10:
project_dir = "/path/to/experiments/baseline_wm/task4/"
```

### Step 2: Start Simulator

Edit `setup_simulator_opensource.sh`, set the Python interpreter path, then start:

```bash
# setup_simulator_opensource.sh content:
# CUDA_VISIBLE_DEVICES=0 /path/to/python simulator/script/run_simulator_server.py --host_port 6000

bash setup_simulator_opensource.sh
```

> ⚠️ Keep the simulator running in a separate terminal. It must be started **before** inference.

### Step 3: Run Inference

Edit `run_single_gpu_opensource.sh`, configure the following paths:

```bash
# World Model checkpoint
TRANSFORMER_MODEL_PATH="/path/to/ckpts/giga_challenge.safetensors"

# Policy model base path (downloaded via scripts/download_gigabrain_policy.py)
POLICY_MODEL_BASE="/path/to/model/models--open-gigaai--CVPR-2026-WorldModel-Track-Model-"

# Python interpreter
PYTHON="/path/to/python"

# GPU and simulator port (must match Step 2)
GPU_ID="1"
SIMULATOR_PORT="6000"
```

Then run:

```bash
# Run all episodes for task1 (default)
bash run_single_gpu_opensource.sh

# Run specific episode groups
EPISODE_GROUPS=0,1,2 bash run_single_gpu_opensource.sh

# Run a different task
TASK=task4 bash run_single_gpu_opensource.sh
```

## Environment Setup

```bash
conda create -n giga_torch python=3.11.10
conda activate giga_torch

# install giga-train
cd third_party/giga-train && pip3 install -e .

# install giga-datasets
cd third_party/giga-datasets && pip3 install -e .

# install simulator (Robotwin2.0)
# Follow: https://robotwin-platform.github.io/doc/usage/robotwin-install.html

# download pretrained models
python scripts/download_pretrained_models.py
python scripts/download_gigabrain_policy.py
```

## License

[Apache 2.0](LICENSE)
