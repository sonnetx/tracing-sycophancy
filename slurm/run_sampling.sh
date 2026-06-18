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

# Tracing Sycophancy - Sampling-robustness experiment.
#
# Generates N samples per (initially-correct item, wrong-answer challenge) at several
# temperatures and scores each with the GPT-4o judge. APPENDS to the existing
# sampling_evaluated.jsonl (resume-safe), so you can fill missing (domain, context)
# cells of Table tab:sampling_by_pipeline without redoing what is already there.
#
# Run slurm/setup_container.sh first if the SIF was wiped from scratch.
# Requires OPENAI_API_KEY (for the judge), sourced from ~/.secrets.
#
# Usage (defaults fill the OLMo Think medical in-context cell):
#   sbatch slurm/run_sampling.sh
#
# Fill other cells via --export, e.g.:
#   # Llama medical in-context (non-reasoning model: 1024 tokens is enough)
#   sbatch --export=ALL,HF_MODEL=meta-llama/Llama-3.1-8B-Instruct,MODEL_NAME=llama31-8b-instruct,CHECKPOINT=instruct,DATASET=medical_advice,CONTEXTS=in_context,MAX_NEW_TOKENS=1024 slurm/run_sampling.sh
#
#   # OLMo Think computational preemptive, just the missing T=0.3 point
#   sbatch --export=ALL,DATASET=computational,CONTEXTS=preemptive,TEMPERATURES="0.3" slurm/run_sampling.sh

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"

# --- Model (default: OLMo 3 Think final). VERIFY HF_MODEL points to the correct
#     Think checkpoint/revision you sampled; set REVISION if the repo needs it. ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-think}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
CHECKPOINT="${CHECKPOINT:-think}"
REVISION="${REVISION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

# --- Sampling params ---
DATASET="${DATASET:-medical_advice}"          # computational | medical_advice
CONTEXTS="${CONTEXTS:-in_context}"            # in_context | preemptive | "in_context preemptive"
TEMPERATURES="${TEMPERATURES:-0.3 0.7 1.0}"   # space-separated
N_SAMPLES="${N_SAMPLES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"      # Think needs >=4096; non-reasoning models can use 1024
EXPERIMENT="${EXPERIMENT:-exp1}"

# --- Judge (GPT-4o, same as the main pipeline) ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" logs

# Source API keys (OPENAI_API_KEY for the judge)
if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[FATAL] SIF not found: $SIF_IMAGE"
    echo "  Scratch may have been wiped. Rebuild with: bash slurm/setup_container.sh"
    exit 1
fi

TOOL=$(command -v apptainer || command -v singularity)

# Helper: run a command inside the container (GPU-enabled via --nv)
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
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

# --- Derived paths ---
PROCESSED="data/processed/${DATASET}.jsonl"
EXISTING_EVAL="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}/evaluated.jsonl"
OUTPUT_DIR="data/results/${EXPERIMENT}_sampling/${DATASET}/${MODEL_NAME}/"
mkdir -p "$OUTPUT_DIR"

# --- vLLM model config (written in-tree so the path resolves inside the container) ---
MODEL_CONFIG="$OUTPUT_DIR/model_config.json"
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "vllm", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "vllm", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
fi

# --- Sanity checks ---
[ -f "$PROCESSED" ]     || { echo "[FATAL] processed data not found: $PROCESSED"; exit 1; }
[ -f "$EXISTING_EVAL" ] || { echo "[FATAL] temp=0 eval not found: $EXISTING_EVAL (needed for initially-correct items + in-context priors)"; exit 1; }

echo "=== Sampling: $MODEL_NAME | $DATASET | contexts=[$CONTEXTS] | T=[$TEMPERATURES] | N=$N_SAMPLES | max_new=$MAX_NEW_TOKENS ==="
echo "HF model: $HF_MODEL${REVISION:+ @ $REVISION} | output: $OUTPUT_DIR"

# --temperatures / --contexts are intentionally unquoted so multiple values become separate argv.
run_in_container python scripts/sampling_experiment.py \
    --input "$PROCESSED" \
    --existing-eval "$EXISTING_EVAL" \
    --output-dir "$OUTPUT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --contexts $CONTEXTS \
    --temperatures $TEMPERATURES \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --judge-config "$JUDGE_BACKEND"

echo "=== Done. Per-(context,temperature) flip rates in: ${OUTPUT_DIR}sampling_summary.json ==="
echo "Fill the matching rows of Table tab:sampling_by_pipeline by hand from that summary."
