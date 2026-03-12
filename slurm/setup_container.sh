#!/bin/bash
#SBATCH --job-name=syco_setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP: Pulls vLLM container and creates venv with project deps.
# Run once before using run_experiment.sh or test_all_models.sh.
#
# Usage:
#   sbatch slurm/setup_container.sh
# =============================================================================

set -e

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"
PIP_CACHE="/scratch/users/$USER/pip_cache"

mkdir -p "$SIF_STORE" "$PIP_CACHE" /scratch/users/$USER/tmp /scratch/users/$USER/huggingface logs

export TMPDIR="/scratch/users/$USER/tmp"

TOOL=$(command -v apptainer || command -v singularity)
echo "INFO: Using container tool: $TOOL"

# --- Pull vLLM image (one-time, ~15GB) ---
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "INFO: Pulling vLLM v0.11.0 container image..."
    cd "$SIF_STORE"
    $TOOL pull "$SIF_IMAGE" docker://vllm/vllm-openai:v0.11.0
    cd "$PROJECT_DIR"
else
    echo "INFO: Container image already exists at $SIF_STORE/$SIF_IMAGE"
fi

# --- Create venv inside container with project deps ---
echo "INFO: Setting up virtual environment inside container..."

"$TOOL" exec \
    --containall \
    -B "$PROJECT_DIR:/workspace" \
    -B "/scratch/users/$USER:/scratch_user" \
    -B "$PIP_CACHE:/root/.cache/pip" \
    -B "/scratch/users/$USER/tmp:/tmp" \
    --home /scratch_user \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/workspace" \
    --pwd /workspace \
    "$SIF_STORE/$SIF_IMAGE" \
    bash -c "
    set -e

    echo 'INFO: Inside container'
    echo 'INFO: Python: '\$(which python3)' ('\$(python3 --version)')'
    echo 'INFO: PyTorch: '\$(python3 -c 'import torch; print(torch.__version__)')
    echo 'INFO: vLLM: '\$(python3 -c 'import vllm; print(vllm.__version__)')

    VENV=/scratch_user/container_env

    # Create venv that inherits container's system packages (torch, vllm, etc.)
    if [ ! -d \$VENV ]; then
        echo 'INFO: Creating virtual environment...'
        python3 -m venv --system-site-packages \$VENV
    else
        echo 'INFO: Virtual environment already exists'
    fi

    source \$VENV/bin/activate
    export PATH=\$VIRTUAL_ENV/bin:\$PATH
    export PYTHONPATH=/workspace

    # Permanently add /workspace to Python path via .pth file
    SITE_DIR=\$(python3 -c 'import site; print(site.getsitepackages()[0])')
    echo '/workspace' > \"\$SITE_DIR/tracing-sycophancy.pth\"

    pip3 install --upgrade pip

    # Install project deps not already in the container
    echo 'INFO: Installing project dependencies...'
    pip3 install --no-cache-dir \
        'numpy==1.26.4' \
        'pandas==2.2.3' \
        'scipy==1.14.1' \
        'statsmodels>=0.14.0' \
        'matplotlib>=3.7.0' \
        'openai>=1.0.0' \
        'anthropic>=0.20.0' \
        'ollama>=0.3.0' \
        'python-dotenv>=1.0.0' \
        'tqdm>=4.60.0' \
        'Flask>=3.0.0'

    # Pin transformers to version compatible with vLLM 0.11.0
    # (Qwen 3.5 requires transformers>=5.2 which is incompatible with vLLM 0.11.0)
    pip3 install --no-cache-dir 'transformers==4.57.1'

    # Verify
    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python3 -c \"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import vllm
print(f'vLLM: {vllm.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

# Test project imports
from src.utils import load_backend
from src.backends.vllm_backend import VLLMBackend
from src.backends.hf_transformers import TransformersBackend
from scripts.generate_responses import generate_for_item
from scripts.score_logprobs import score_item
print()
print('All imports successful!')
\"
    echo ''
    echo 'INFO: Setup complete!'
    echo 'INFO: Venv location: \$VENV'
"
