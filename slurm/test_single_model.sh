#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --partition=roxanad
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs inference + evaluation for a single model inside Apptainer container.
# Called by test_all_models.sh — not meant to be run directly.
#
# Required env vars: HF_MODEL, MODEL_NAME, MODEL_TYPE, CHECKPOINT,
#                    EXPERIMENT, TEST_INPUT, JUDGE_BACKEND, BACKEND_TYPE

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

TOOL=$(command -v apptainer || command -v singularity)

echo "============================================================"
echo "MODEL: $MODEL_NAME ($HF_MODEL) — type=$MODEL_TYPE backend=$BACKEND_TYPE"
echo "============================================================"

RESULT_DIR="data/results/${EXPERIMENT}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

RESPONSES="$RESULT_DIR/responses.jsonl"
LOGPROB_SCORES="$RESULT_DIR/logprob_scores.jsonl"
EVALUATED="$RESULT_DIR/evaluated.jsonl"

# Model config — written inside project tree so the path works in the container
MODEL_CONFIG="$RESULT_DIR/model_config.json"
cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16"}
CONF

# Helper: run inside container
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

# --- Inference (generative + log-prob) ---
echo "[Step 3+3b] Running inference (generative + log-prob)..."
run_in_container python scripts/run_inference.py \
    --input "$TEST_INPUT" \
    --output-dir "$RESULT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" 2>&1

# --- Evaluate ---
if [ -f "$RESPONSES" ]; then
    echo "[Step 4] Evaluating with judge..."
    run_in_container python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$TEST_INPUT" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND" 2>&1
else
    echo "[Step 4] Skipping evaluation (no responses generated)"
fi

# --- Validate ---
echo ""
python3 -c "
import json, sys

model = '$MODEL_NAME'
errors = []

# Generative track
try:
    evaluated = [json.loads(l) for l in open('$EVALUATED')]
    print(f'  Generative: {len(evaluated)} items evaluated')
    for item in evaluated:
        acc = item['initial'].get('metrics', {}).get('factual_accuracy', '?')
        n_ch = sum(1 for cr in item.get('challenge_responses', []) if cr.get('metrics'))
        print(f'    {item[\"question_id\"]}: accuracy={acc}  challenges={n_ch}/8')
except Exception as e:
    errors.append(f'Generative track failed: {e}')
    print(f'  Generative: FAILED ({e})')

# Log-prob track
try:
    scored = [json.loads(l) for l in open('$LOGPROB_SCORES')]
    print(f'  Log-prob:   {len(scored)} items scored')
    for item in scored:
        bl = item['baseline']
        q = item['quality']
        n_ch = len(item.get('challenge_scores', []))
        deltas = [cs['delta_log_odds'] for cs in item.get('challenge_scores', [])]
        delta_range = f'[{min(deltas):+.3f}, {max(deltas):+.3f}]' if deltas else '[]'
        print(f'    {item[\"question_id\"]}: log_odds={bl[\"log_odds\"]:+.3f}  near_random={q[\"near_random\"]}  challenges={n_ch}  deltas={delta_range}')
        if bl['correct_num_tokens'] == 0:
            errors.append(f'{item[\"question_id\"]}: correct_num_tokens=0')
        if bl['incorrect_num_tokens'] == 0:
            errors.append(f'{item[\"question_id\"]}: incorrect_num_tokens=0')
except Exception as e:
    errors.append(f'Log-prob track failed: {e}')
    print(f'  Log-prob:   FAILED ({e})')

if errors:
    for e in errors:
        print(f'  ERROR: {e}')
    print(f'  RESULT: {model} FAILED')
    sys.exit(1)
else:
    print(f'  RESULT: {model} PASSED')
" 2>&1

echo ""
echo "=== Done: $MODEL_NAME ==="
