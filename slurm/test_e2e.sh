#!/bin/bash
#SBATCH --job-name=test_e2e
#SBATCH --partition=roxanad
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quick end-to-end smoke test: runs BOTH tracks (generative + log-prob)
# on 5 items with a single OLMo model.
#
# Usage:
#   sbatch slurm/test_e2e.sh
#   sbatch --export=ALL,HF_MODEL=allenai/OLMo-7B-Instruct,MODEL_NAME=olmo-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct slurm/test_e2e.sh

# --- Modules ---
ml python/3.12.1
ml cuda/12.4.0

# --- Paths ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# --- Model (defaults to OLMo 3 base) ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-1025-7B}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-base}"
MODEL_TYPE="${MODEL_TYPE:-base}"
CHECKPOINT="${CHECKPOINT:-base}"

# --- Fixed test settings ---
N_ITEMS="${N_ITEMS:-5}"
DATASET="computational"
EXPERIMENT="test_e2e"

# --- Judge ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Activate environment ---
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

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
mkdir -p logs

echo "=== E2E smoke test: $MODEL_NAME ($MODEL_TYPE) — $N_ITEMS items ==="

# --- Model config ---
MODEL_CONFIG=$(mktemp /tmp/hf_config_XXXX.json)
cat > "$MODEL_CONFIG" <<CONF
{"backend": "transformers", "model": "$HF_MODEL", "torch_dtype": "bfloat16"}
CONF
trap "rm -f $MODEL_CONFIG" EXIT

# --- Step 1: Preprocess ---
PROCESSED="data/processed/${DATASET}.jsonl"
if [ ! -f "$PROCESSED" ]; then
    echo "[Step 1] Preprocessing $DATASET..."
    python scripts/preprocess.py \
        --dataset "$DATASET" \
        --raw-path "data/raw/$DATASET" \
        --output "$PROCESSED" \
        --sample-size 500
else
    echo "[Step 1] Skipping ($PROCESSED exists)"
fi

# --- Step 2: Generate challenges ---
if ! head -1 "$PROCESSED" | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'challenges' in d and d['challenges'] and 'PLACEHOLDER' not in d['challenges'][0].get('prompt','') else 1)" 2>/dev/null; then
    echo "[Step 2] Generating challenges..."
    python scripts/generate_challenges.py \
        --input "$PROCESSED" \
        --output "$PROCESSED" \
        --backend-config "$JUDGE_BACKEND" \
        --challenge-type factual
else
    echo "[Step 2] Skipping (challenges already present)"
fi

# --- Extract N_ITEMS for testing ---
TEST_INPUT=$(mktemp /tmp/test_input_XXXX.jsonl)
head -"$N_ITEMS" "$PROCESSED" > "$TEST_INPUT"
ACTUAL=$(wc -l < "$TEST_INPUT")
echo "Using $ACTUAL items for test"

# --- Output paths ---
RESULT_DIR="data/results/${EXPERIMENT}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"
RESPONSES="$RESULT_DIR/responses.jsonl"
LOGPROB_SCORES="$RESULT_DIR/logprob_scores.jsonl"
EVALUATED="$RESULT_DIR/evaluated.jsonl"
rm -f "$RESPONSES" "$LOGPROB_SCORES" "$EVALUATED"  # fresh run

# ===========================================
# Both tracks (single model load)
# ===========================================
echo ""
echo "=== INFERENCE (generative + log-prob) ==="

python scripts/run_inference.py \
    --input "$TEST_INPUT" \
    --output-dir "$RESULT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT"

if [ -f "$RESPONSES" ]; then
    echo "[Step 4] Evaluating with judge..."
    python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$TEST_INPUT" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND"
else
    echo "[Step 4] Skipping evaluation (no responses generated)"
fi

rm -f "$TEST_INPUT"

# ===========================================
# Validate both tracks
# ===========================================
echo ""
echo "=== VALIDATION ==="

python -c "
import json, sys

# --- Check generative track ---
print('--- Generative Track ---')
if True:
    try:
        evaluated = [json.loads(l) for l in open('$EVALUATED')]
    except FileNotFoundError:
        print('FAIL: evaluated.jsonl not created')
        sys.exit(1)

    print(f'Items evaluated: {len(evaluated)}')
    for item in evaluated:
        qid = item['question_id']
        init = item['initial']
        m = init.get('metrics', {})
        acc = m.get('factual_accuracy', '?')
        n_ch = len(item.get('challenge_responses', []))
        ch_with_metrics = sum(1 for cr in item.get('challenge_responses', []) if cr.get('metrics'))
        resp_preview = init['response'][:80].replace(chr(10), ' ')
        print(f'  {qid}: accuracy={acc}  challenges={ch_with_metrics}/{n_ch}  resp=\"{resp_preview}...\"')

# --- Check log-prob track ---
print()
print('--- Log-Prob Track ---')
try:
    scored = [json.loads(l) for l in open('$LOGPROB_SCORES')]
except FileNotFoundError:
    print('FAIL: logprob_scores.jsonl not created')
    sys.exit(1)

print(f'Items scored: {len(scored)}')
errors = []
for item in scored:
    qid = item['question_id']
    bl = item['baseline']
    q = item['quality']

    if bl['correct_num_tokens'] == 0:
        errors.append(f'{qid}: correct_num_tokens=0')
    if bl['incorrect_num_tokens'] == 0:
        errors.append(f'{qid}: incorrect_num_tokens=0')
    if bl['correct_log_prob'] > 0:
        errors.append(f'{qid}: correct_log_prob > 0')

    n_ch = len(item.get('challenge_scores', []))
    deltas = [cs['delta_log_odds'] for cs in item.get('challenge_scores', [])]
    delta_range = f'[{min(deltas):+.3f}, {max(deltas):+.3f}]' if deltas else '[]'
    print(f'  {qid}: log_odds={bl[\"log_odds\"]:+.3f}  near_random={q[\"near_random\"]}  challenges={n_ch}  deltas={delta_range}')

print()
if errors:
    for e in errors:
        print(f'ERROR: {e}')
    print('SOME CHECKS FAILED')
    sys.exit(1)
else:
    print('All checks passed.')
"

# ===========================================
# Analysis & plots
# ===========================================
echo ""
echo "=== GENERATING PLOTS ==="
ANALYSIS_DIR="data/results/${EXPERIMENT}/analysis"
python scripts/analyze.py \
    --results-dir "data/results/${EXPERIMENT}" \
    --output-dir "$ANALYSIS_DIR" 2>&1

echo ""
echo "Plots saved to: $ANALYSIS_DIR"
ls -la "$ANALYSIS_DIR"/*.png 2>/dev/null || echo "(no plots generated)"
