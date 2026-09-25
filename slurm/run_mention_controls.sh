#!/bin/bash
#SBATCH --job-name=syco_mention_controls
#SBATCH --partition=roxanad,gpu
#SBATCH --time=1:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Endorsement controls for the literal-string ΔLogOdds shift (see
# scripts/make_mention_controls.py). Reuses the 200-item paraphrase subset, so
# run_paraphrase_robustness.sh must have materialized its variants first. No API calls.
#
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-1025-7B,MODEL_NAME=olmo3-7b-base,CHECKPOINT=base \
#     slurm/run_mention_controls.sh

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

# --- Model ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
CHECKPOINT="${CHECKPOINT:-instruct}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

# --- Experiment ---
DATASET="${DATASET:-medical_advice}"
EXPERIMENT="${EXPERIMENT:-exp_mention_controls}"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_ETAG_TIMEOUT=10 APPTAINERENV_HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" logs

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

TOOL=$(command -v apptainer || command -v singularity)

run_in_container() {
    "$TOOL" exec --nv \
        --containall \
        -B "$PROJECT_DIR:/workspace" \
        -B "/scratch/users/$USER:/scratch_user" \
        -B "/scratch/users/$USER/tmp:/tmp" \
        --home /scratch_user \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=/workspace" \
        --env "HF_HOME=/scratch_user/huggingface" \
        --env "HF_DATASETS_CACHE=/scratch_user/huggingface/datasets" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== Mention controls: $DATASET | $MODEL_NAME ==="

VARIANT_FILE="data/processed/${DATASET}_paraphrase_variants/variant_c0_w0.jsonl"
CONTROL_FILE="data/processed/${DATASET}_mention_controls.jsonl"
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

if [ ! -f "$CONTROL_FILE" ]; then
    run_in_container python scripts/make_mention_controls.py \
        --input "$VARIANT_FILE" --output "$CONTROL_FILE"
fi

MODEL_CONFIG="$RESULT_DIR/model_config.json"
cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF

run_in_container python scripts/score_logprobs.py \
    --input "$CONTROL_FILE" \
    --output "$RESULT_DIR/logprob_scores.jsonl" \
    --backend-config "$MODEL_CONFIG" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --resume

echo "=== Done: $MODEL_NAME on $DATASET ==="
