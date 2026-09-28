#!/bin/bash
#SBATCH --job-name=syco_candidate_robustness
#SBATCH --partition=roxanad
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Check 1: Candidate-dependence robustness for ΔLogOdds.
#
# Generates 3 independent wrong-answer candidates per question, then scores each
# candidate separately to measure how much ΔLogOdds varies across candidate draws.
# Stable cross-candidate rankings bound the candidate-dependence concern.
#
# Usage (one job per model, after main processed JSONL already exists):
#   sbatch --export=ALL,\
#     HF_MODEL=allenai/OLMo-3-1025-7B-Instruct,\
#     MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=instruct,\
#     CHECKPOINT=instruct,DATASET=computational \
#     slurm/run_candidate_robustness.sh

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"

# --- Model ---
HF_MODEL="${HF_MODEL:-allenai/OLMo-3-1025-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
MODEL_TYPE="${MODEL_TYPE:-instruct}"
CHECKPOINT="${CHECKPOINT:-instruct}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"

# --- Dataset ---
DATASET="${DATASET:-computational}"
EXPERIMENT="${EXPERIMENT:-exp_candidate_robustness}"

# --- Challenge generation backend (GPT-4o) ---
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Environment ---
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

echo "=== Candidate robustness: $DATASET | $MODEL_NAME (${NUM_CANDIDATES} candidates) ==="

PROCESSED_MAIN="data/processed/${DATASET}.jsonl"
CANDIDATES_FILE="data/processed/${DATASET}_candidates.jsonl"
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

# --- Step 1: Generate 3 wrong-answer candidates ---
if [ -f "$CANDIDATES_FILE" ] && python3 -c "
import json, sys
try:
    d = json.loads(open('$CANDIDATES_FILE').readline())
    cands = d.get('proposed_answer_candidates', [])
    sys.exit(0 if len(cands) >= $NUM_CANDIDATES else 1)
except: sys.exit(1)
" 2>/dev/null; then
    echo "[Step 1] Skipping — $CANDIDATES_FILE already has ${NUM_CANDIDATES} candidates"
else
    echo "[Step 1] Generating ${NUM_CANDIDATES} wrong-answer candidates per question..."
    case "$DATASET" in
        computational)  ETHOS="an expert in math" ;;
        medical_advice) ETHOS="an expert in medicine" ;;
        *)              ETHOS="an expert" ;;
    esac
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED_MAIN" \
        --output "$CANDIDATES_FILE" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual \
        --ethos "'$ETHOS'" \
        --num-candidates "$NUM_CANDIDATES"
fi

# --- Step 2: Build model backend config ---
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

# --- Steps 3-5: Score each candidate ---
for IDX in $(seq 0 $((NUM_CANDIDATES - 1))); do
    SCORES_FILE="$RESULT_DIR/logprob_scores_c${IDX}.jsonl"
    if [ -f "$SCORES_FILE" ] && [ "$(wc -l < "$SCORES_FILE")" -ge "$(wc -l < "$CANDIDATES_FILE")" ]; then
        echo "[Step $((IDX + 3))] Skipping candidate $IDX (already scored)"
        continue
    fi
    echo "[Step $((IDX + 3))] Scoring candidate $IDX..."
    run_in_container python scripts/score_logprobs.py \
        --input "$CANDIDATES_FILE" \
        --output "$SCORES_FILE" \
        --backend-config "$MODEL_CONFIG" \
        --model-name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --candidate-idx "$IDX" \
        --resume
done

# --- Step 6: Analyze variance and rank stability ---
echo "[Step 6] Analyzing candidate robustness..."
SCORE_FILES=""
for IDX in $(seq 0 $((NUM_CANDIDATES - 1))); do
    SCORE_FILES="$SCORE_FILES $RESULT_DIR/logprob_scores_c${IDX}.jsonl"
done

run_in_container python scripts/analyze_candidate_robustness.py \
    --files $SCORE_FILES \
    --dataset "$DATASET" \
    --output "$RESULT_DIR/candidate_robustness_report.json"

echo "=== Done: $MODEL_NAME on $DATASET ==="
echo "Report: $RESULT_DIR/candidate_robustness_report.json"
