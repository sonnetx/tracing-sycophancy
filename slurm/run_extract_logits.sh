#!/bin/bash
#SBATCH --job-name=logits_probe
#SBATCH --partition=roxanad
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Extract top-K next-token log-probs under challenge for a selected set of
# items, to visualize mode sharpness at the decision point.
#
# Usage (submit four separate jobs, one per pipeline):
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct,DATASET=computational,QIDS="comp_042 comp_105",CIDS="ethos_preemptive citation_preemptive" slurm/run_extract_logits.sh

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"

HF_MODEL="${HF_MODEL:?HF_MODEL required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME required}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
CHECKPOINT="${CHECKPOINT:-instruct}"
DATASET="${DATASET:-computational}"
EXPERIMENT="${EXPERIMENT:-exp1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
TOPK="${TOPK:-20}"
QIDS="${QIDS:?QIDS required, e.g. 'comp_042 comp_105'}"
CIDS="${CIDS:-ethos_preemptive citation_preemptive}"
# Default suffix forces the model to the answer-digit decision point.
# Override via PROMPT_SUFFIX env var; use literal \n for newlines.
PROMPT_SUFFIX="${PROMPT_SUFFIX:-$'\n\nFinal answer: $'}"
export PROMPT_SUFFIX

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

PROCESSED="data/processed/${DATASET}.jsonl"
SOURCE_EVAL="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}/evaluated.jsonl"
OUT_DIR="data/results/logits_probe/${DATASET}/${MODEL_NAME}"
OUT_PATH="${OUT_DIR}/topk.jsonl"
mkdir -p "$OUT_DIR"
rm -f "$OUT_PATH"  # start fresh each run

MODEL_CONFIG="$OUT_DIR/model_config.json"
cat > "$MODEL_CONFIG" <<CONF
{"backend": "vllm", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF

TOOL=$(command -v apptainer || command -v singularity)

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
    --env "PROMPT_SUFFIX=$PROMPT_SUFFIX" \
    --pwd /workspace \
    "$SIF_IMAGE" \
    bash -c 'source /scratch_user/container_env/bin/activate && python scripts/extract_next_token_logits.py \
        --input '"$PROCESSED"' \
        --existing-eval '"$SOURCE_EVAL"' \
        --backend-config '"$MODEL_CONFIG"' \
        --model-type '"$MODEL_TYPE"' \
        --model-name '"$MODEL_NAME"' \
        --question-ids '"$QIDS"' \
        --challenge-ids '"$CIDS"' \
        --topk '"$TOPK"' \
        --prompt-suffix "$PROMPT_SUFFIX" \
        --output-path '"$OUT_PATH"

echo "Done: $OUT_PATH"
