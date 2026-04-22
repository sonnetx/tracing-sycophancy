#!/bin/bash
#SBATCH --job-name=syco_sampling
#SBATCH --partition=roxanad
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sampling experiment: test whether the decoding-time override of the
# pre-commitment prior survives non-greedy decoding.
#

# Usage:
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct slurm/run_sampling_experiment.sh
#   sbatch --export=ALL,HF_MODEL=allenai/Llama-3.1-Tulu-3-8B,MODEL_NAME=tulu3-llama31-8b,MODEL_TYPE=chat,CHECKPOINT=instruct slurm/run_sampling_experiment.sh

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"

HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
CHECKPOINT="${CHECKPOINT:-instruct}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

DATASET="${DATASET:-computational}"
EXPERIMENT="${EXPERIMENT:-exp1}"
SAMPLING_EXP="${SAMPLING_EXP:-exp1_sampling}"

TEMPERATURES="${TEMPERATURES:-0.3 0.7 1.0}"
N_SAMPLES="${N_SAMPLES:-5}"
CONTEXTS="${CONTEXTS:-preemptive}"   # preemptive | in_context | "preemptive in_context"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"   # Raise for CoT models (Think: 4096)

JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

PROCESSED="data/processed/${DATASET}.jsonl"
SOURCE_EVAL="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}/evaluated.jsonl"
OUT_DIR="data/results/${SAMPLING_EXP}/${DATASET}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

# Build a backend config for this run
MODEL_CONFIG="$OUT_DIR/model_config.json"
if [ -n "$REVISION" ]; then
cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
else
cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
fi

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
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== Sampling experiment: $MODEL_NAME on $DATASET ==="
echo "Temperatures: $TEMPERATURES | Samples per item: $N_SAMPLES"
echo "Source evaluated.jsonl: $SOURCE_EVAL"

run_in_container python scripts/sampling_experiment.py \
    --input "$PROCESSED" \
    --existing-eval "$SOURCE_EVAL" \
    --output-dir "$OUT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --temperatures $TEMPERATURES \
    --n-samples "$N_SAMPLES" \
    --contexts $CONTEXTS \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --judge-config "$JUDGE_BACKEND"

echo "=== Done: results in $OUT_DIR ==="
