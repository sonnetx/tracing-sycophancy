#!/bin/bash
#SBATCH --job-name=test_logprob
#SBATCH --partition=gpu
#SBATCH --time=19:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quick smoke test for the log-probability scoring pipeline.
#
# Runs steps 1 → 2 → 3b on 5 items only, then prints a summary.
# Steps 3 (generate responses), 4 (judge), and 5 (analysis) are skipped.
#
# Usage:
#   sbatch slurm/test_logprobs.sh
#   sbatch --export=ALL,HF_MODEL=meta-llama/Llama-3.2-3B-Instruct,MODEL_NAME=llama-3.2-3b-instruct slurm/test_logprobs.sh

# --- Modules ---
ml python/3.12.1
ml cuda/12.4.0

# --- Paths ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# --- Model ---
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-3B}"
MODEL_NAME="${MODEL_NAME:-llama-3.2-3b-base}"
CHECKPOINT="${CHECKPOINT:-base}"

# --- Fixed test settings ---
N_ITEMS="${N_ITEMS:-5}"
DATASET="computational"
EXPERIMENT="test_logprobs"

# --- Activate environment ---
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== Log-prob smoke test: $MODEL_NAME ($N_ITEMS items) ==="

# --- Model config ---
MODEL_CONFIG=$(mktemp /tmp/hf_config_XXXX.json)
cat > "$MODEL_CONFIG" <<CONF
{"backend": "transformers", "model": "$HF_MODEL"}
CONF
trap "rm -f $MODEL_CONFIG" EXIT

# --- Judge backend for challenge generation ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Step 1: Preprocess (use the full processed file or create it) ---
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

# --- Step 2: Generate challenges if needed ---
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

# --- Step 3b: Score log-probabilities ---
OUTPUT="data/results/${EXPERIMENT}/${MODEL_NAME}/logprob_scores.jsonl"
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"  # fresh run for test

echo "[Step 3b] Scoring log-probabilities..."
python scripts/score_logprobs.py \
    --input "$TEST_INPUT" \
    --output "$OUTPUT" \
    --backend-config "$MODEL_CONFIG" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT"

rm -f "$TEST_INPUT"

# --- Validate output ---
echo ""
echo "=== RESULTS ==="

if [ ! -f "$OUTPUT" ]; then
    echo "FAIL: No output file created"
    exit 1
fi

SCORED=$(wc -l < "$OUTPUT")
echo "Items scored: $SCORED / $ACTUAL"

if [ "$SCORED" -eq 0 ]; then
    echo "FAIL: Zero items scored"
    exit 1
fi

# Print summary for each scored item
python -c "
import json, sys

path = '$OUTPUT'
ok = True
for line in open(path):
    d = json.loads(line)
    bl = d['baseline']
    q = d['quality']
    qid = d['question_id']

    # Checks
    errors = []
    if bl['correct_num_tokens'] == 0:
        errors.append('correct_num_tokens=0 (BPE boundary bug?)')
    if bl['incorrect_num_tokens'] == 0:
        errors.append('incorrect_num_tokens=0 (BPE boundary bug?)')
    if bl['correct_log_prob'] > 0:
        errors.append(f'correct_log_prob={bl[\"correct_log_prob\"]:.3f} (should be negative)')
    if bl['incorrect_log_prob'] > 0:
        errors.append(f'incorrect_log_prob={bl[\"incorrect_log_prob\"]:.3f} (should be negative)')

    for cs in d.get('challenge_scores', []):
        if cs['correct_num_tokens'] == 0:
            errors.append(f'{cs[\"challenge_id\"]}: correct_num_tokens=0')
        if cs['incorrect_num_tokens'] == 0:
            errors.append(f'{cs[\"challenge_id\"]}: incorrect_num_tokens=0')

    status = 'PASS' if not errors else 'FAIL'
    if errors:
        ok = False

    print(f'{status} {qid}')
    print(f'  baseline: log_odds={bl[\"log_odds\"]:+.3f}  correct={bl[\"correct_num_tokens\"]}tok  incorrect={bl[\"incorrect_num_tokens\"]}tok')
    print(f'  quality:  near_random={q[\"near_random\"]}  mean_lp={q[\"mean_log_prob\"]:.3f}')
    n_ch = len(d.get('challenge_scores', []))
    if n_ch > 0:
        deltas = [cs['delta_log_odds'] for cs in d['challenge_scores']]
        print(f'  challenges: {n_ch} scored  delta_range=[{min(deltas):+.3f}, {max(deltas):+.3f}]')
    for e in errors:
        print(f'  ERROR: {e}')
    print()

if ok:
    print('All checks passed.')
else:
    print('Some checks FAILED — see errors above.')
    sys.exit(1)
"
