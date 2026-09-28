#!/bin/bash
#SBATCH --job-name=syco_length_robustness
#SBATCH --partition=roxanad
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Check 3: Length-matched wrong-answer robustness (medical_advice domain only).
#
# Generates a version of the medical_advice challenges where wrong answers are
# trimmed to ≤ correct-answer word count, then re-runs logprob scoring.
# If ΔLogOdds rankings are stable, length asymmetry is not driving the medical results.
#
# Usage (run once per model):
#   sbatch --export=ALL,\
#     HF_MODEL=allenai/OLMo-3-1025-7B-Instruct,\
#     MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=instruct,\
#     CHECKPOINT=instruct \
#     slurm/run_length_robustness.sh
#
# Only medical_advice is run; computational answers are naturally short / uniform.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

HF_MODEL="${HF_MODEL:-allenai/OLMo-3-1025-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
MODEL_TYPE="${MODEL_TYPE:-instruct}"
CHECKPOINT="${CHECKPOINT:-instruct}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

DATASET="medical_advice"
EXPERIMENT="${EXPERIMENT:-exp_length_robustness}"
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-config/models/gpt4o_judge.json}"
BASELINE_DIR="${BASELINE_DIR:-data/results/exp1}"

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
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== Length robustness: $DATASET | $MODEL_NAME ==="

PROCESSED_MAIN="data/processed/${DATASET}.jsonl"
LENGTH_MATCHED_FILE="data/processed/${DATASET}_length_matched.jsonl"
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

# --- Step 1: Generate length-matched challenges (reuses cached GPT-4o content) ---
if [ -f "$LENGTH_MATCHED_FILE" ] && python3 -c "
import json, sys
try:
    d = json.loads(open('$LENGTH_MATCHED_FILE').readline())
    sys.exit(0 if 'proposed_answer_length_matched' in d else 1)
except: sys.exit(1)
" 2>/dev/null; then
    echo "[Step 1] Skipping — $LENGTH_MATCHED_FILE already has length-matched challenges"
else
    echo "[Step 1] Generating length-matched challenges for $DATASET..."
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED_MAIN" \
        --output "$LENGTH_MATCHED_FILE" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual \
        --ethos "'an expert in medicine'" \
        --length-match-wrong-answers
fi

# --- Step 2: Build model config ---
MODEL_CONFIG="$RESULT_DIR/model_config.json"
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
fi

# --- Step 3: Logprob scoring on length-matched challenges ---
SCORES_FILE="$RESULT_DIR/logprob_scores.jsonl"
if [ -f "$SCORES_FILE" ] && [ "$(wc -l < "$SCORES_FILE")" -ge "$(wc -l < "$LENGTH_MATCHED_FILE")" ]; then
    echo "[Step 2] Skipping — logprob scores already complete"
else
    echo "[Step 2] Scoring logprobs on length-matched challenges..."
    run_in_container python scripts/score_logprobs.py \
        --input "$LENGTH_MATCHED_FILE" \
        --output "$SCORES_FILE" \
        --backend-config "$MODEL_CONFIG" \
        --model-name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --resume
fi

echo "=== Done: $MODEL_NAME on $DATASET (length-matched) ==="
echo ""
echo "To run the analysis across all models:"
echo "  python scripts/analyze_length_robustness.py \\"
echo "      --experiment-dir data/results/$EXPERIMENT \\"
echo "      --baseline-dir   $BASELINE_DIR \\"
echo "      --dataset        $DATASET \\"
echo "      --processed-path $LENGTH_MATCHED_FILE"
