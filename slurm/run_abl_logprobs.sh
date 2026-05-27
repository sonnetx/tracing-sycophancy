#!/bin/bash
#SBATCH --job-name=syco_abl_lp
#SBATCH --partition=roxanad
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Logprob scoring for citation-ablation challenges.
# Reads data/processed/${DATASET}_ablation.jsonl (produced by gen_ablation_challenges.sh)
# and writes logprob_scores.jsonl to data/results/exp_citation_ablation/{dataset}/{model}.
#
# Usage (one job per model × dataset):
#   sbatch --export=ALL,\
#     HF_MODEL=allenai/OLMo-3-7B-Instruct,\
#     MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct,\
#     DATASET=computational \
#     slurm/run_abl_logprobs.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

HF_MODEL="${HF_MODEL:-allenai/OLMo-3-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
CHECKPOINT="${CHECKPOINT:-instruct}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

DATASET="${DATASET:-computational}"
EXPERIMENT="${EXPERIMENT:-exp_citation_ablation}"

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
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
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

PROCESSED_ABLATION="data/processed/${DATASET}_ablation.jsonl"
if [ ! -f "$PROCESSED_ABLATION" ]; then
    echo "ERROR: $PROCESSED_ABLATION not found. Run gen_ablation_challenges.sh first."
    exit 1
fi

echo "=== Ablation logprob scoring: $DATASET | $MODEL_NAME ==="

RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

MODEL_CONFIG="$RESULT_DIR/model_config_lp.json"
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
fi

SCORES_FILE="$RESULT_DIR/logprob_scores.jsonl"
ABL_N=$(wc -l < "$PROCESSED_ABLATION")
if [ -f "$SCORES_FILE" ] && [ "$(wc -l < "$SCORES_FILE")" -ge "$ABL_N" ]; then
    echo "[Logprobs] Skipping — scores already complete ($SCORES_FILE)"
else
    echo "[Logprobs] Scoring ablation logprobs ($ABL_N items)..."
    run_in_container python scripts/score_logprobs.py \
        --input "$PROCESSED_ABLATION" \
        --output "$SCORES_FILE" \
        --backend-config "$MODEL_CONFIG" \
        --model-name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --resume
fi

echo "=== Done: $MODEL_NAME / $DATASET ==="
echo "Output: $SCORES_FILE"
