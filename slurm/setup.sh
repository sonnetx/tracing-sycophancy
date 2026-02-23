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
ml cuda/12.4.0

# --- Environment setup ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# Clean slate — remove stale venv if it exists
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Cache dirs — keep everything on scratch, not home
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
export MODEL_DIR="/scratch/users/$USER/models"

mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MODEL_DIR"

which python
python --version

# --- Install dependencies ---
pip3 install --no-cache-dir --upgrade pip

# Install scientific packages as wheels only (no source builds)
# These exact versions have manylinux_2_17 wheels for Python 3.12
pip3 install --no-cache-dir --only-binary :all: numpy==1.26.4 pandas==2.2.3 scipy==1.14.1

# Install the project in editable mode (pulls remaining deps from pyproject.toml)
pip3 install --no-cache-dir -e "$PROJECT_DIR"

# PyTorch + HuggingFace for GPU inference (no vLLM server needed)
pip3 install --no-cache-dir torch==2.6.0
pip3 install --no-cache-dir transformers accelerate sentencepiece

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
