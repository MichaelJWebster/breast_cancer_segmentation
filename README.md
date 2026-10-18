# breast_cancer_segmentation

Breast cancer segmentation using the `segmentation-models-pytorch` library to perform segmentation on breast ultrasound images. The goal is to classify regions as malignant, benign, or normal.

## Data

The data comes from this [Kaggle competition](https://www.kaggle.com/competitions/breast-cancer-us-classification-ml-4-im-2025-2026/data) (you need to join the competition and accept its rules before you can download). Download the dataset and place it in the `./data` directory so that the training and test sets live under `./data/complete_set` (training images in `./data/complete_set/training_set/{benign,malignant}`). To keep the data elsewhere, set `BC_DATA_DIR` to the directory that contains `complete_set`.

## Environment Setup

These instructions create a Python 3.12.8 virtual environment and install all the packages required to run `bc_classification.ipynb`.

### Prerequisites

* Python **3.12.8** installed and available on your `PATH` (for example via [pyenv](https://github.com/pyenv/pyenv) or the system package manager).

### 1. Create and activate a virtual environment

```bash
# Create the venv (use the full path to your Python 3.12.8 if it is not the default `python3`).
python3.12 -m venv .venv

# Activate it
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch

PyTorch should be installed **first** so that the matching CUDA/CPU variant is picked up by the other packages. Visit <https://pytorch.org/get-started/locally/> for the command that matches your hardware. For example:

```bash
# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.4 (pick the URL that matches your driver)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you already installed `torch` and `torchvision` in step 3, pip will skip those lines in `requirements.txt` automatically.

### 5. Install Jupyter (if not already available)

```bash
pip install notebook
# or
pip install jupyterlab
```

### 6. Run the notebook

Training uses `accelerator="auto"`: CUDA on an NVIDIA GPU, MPS on Apple Silicon, otherwise CPU. Set `BC_EPOCHS=1` for a quick smoke test.

```bash
jupyter notebook bc_classification.ipynb
```
