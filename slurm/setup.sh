#!/bin/bash
#SBATCH --job-name=syco_setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Tracing Sycophancy - Environment Setup

ml gcc/14.2.0
ml python/3.12.1
ml cuda/11.7.1

# --- Environment setup ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/tracing-sycophancy}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export TMPDIR="${TMPDIR:-/scratch/users/$USER/tmp}"
export HF_HOME="${HF_HOME:-/scratch/users/$USER/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="${TORCH_HOME:-/scratch/users/$USER/torch}"

mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

which python
python --version

# --- Install dependencies ---
pip3 install --no-cache-dir --upgrade pip

# Install the project in editable mode (pulls all deps from pyproject.toml)
pip3 install --no-cache-dir -e "$PROJECT_DIR"

# PyTorch (adjust CUDA version for your cluster)
# pip3 install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# HuggingFace (if running models directly rather than via API)
# pip3 install --no-cache-dir transformers accelerate

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
