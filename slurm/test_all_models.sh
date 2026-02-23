#!/bin/bash
#SBATCH --job-name=test_all
#SBATCH --partition=roxanad
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Smoke test: runs both tracks on 5 items for ALL OLMo models sequentially.
# Cleans results before each model so nothing is skipped.
#
# Usage:
#   sbatch slurm/test_all_models.sh

# --- Modules ---
ml python/3.12.1
ml cuda/12.4.0

# --- Paths ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# --- Fixed test settings ---
N_ITEMS=20
DATASET="medical_advice"
EXPERIMENT="test_all_models"
JUDGE_BACKEND="config/models/gpt4o_judge.json"

# --- All OLMo 3 models: name|hf_model_id|model_type|checkpoint ---
MODELS=(
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    # LLM360 Amber 7B (base → SFT → Safety DPO):
    # "amber-7b-base|LLM360/Amber|base|base"
    # "amber-7b-sft|LLM360/AmberChat|chat|sft"
    # "amber-7b-dpo|LLM360/AmberSafe|chat|dpo"
    # Zephyr 7B (Mistral base → SFT → DPO):
    # "zephyr-7b-base|mistralai/Mistral-7B-v0.1|base|base"
    # "zephyr-7b-sft|alignment-handbook/zephyr-7b-sft-full|chat|sft"
    # "zephyr-7b-dpo|HuggingFaceH4/zephyr-7b-beta|chat|dpo"
)

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

# Clean previous test results to avoid stale data polluting analysis
RESULT_BASE="data/results/${EXPERIMENT}"
if [ -d "$RESULT_BASE" ]; then
    echo "Clearing previous test results: $RESULT_BASE"
    rm -rf "$RESULT_BASE"
fi

echo "=== All-models smoke test: ${#MODELS[@]} models × $N_ITEMS items ==="

# --- Step 1 & 2: Preprocess + challenges (shared, run once) ---
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

# --- Extract N_ITEMS ---
TEST_INPUT=$(mktemp /tmp/test_input_XXXX.jsonl)
head -"$N_ITEMS" "$PROCESSED" > "$TEST_INPUT"
ACTUAL=$(wc -l < "$TEST_INPUT")
echo "Using $ACTUAL items per model"
echo ""

# --- Track results ---
PASSED=0
FAILED=0
FAILED_MODELS=""

for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"

    echo "============================================================"
    echo "MODEL: $MODEL_NAME ($HF_MODEL) — type=$MODEL_TYPE"
    echo "============================================================"

    # Fresh output dir
    RESULT_DIR="data/results/${EXPERIMENT}/${MODEL_NAME}"
    rm -rf "$RESULT_DIR"
    mkdir -p "$RESULT_DIR"

    RESPONSES="$RESULT_DIR/responses.jsonl"
    LOGPROB_SCORES="$RESULT_DIR/logprob_scores.jsonl"
    EVALUATED="$RESULT_DIR/evaluated.jsonl"

    # Model config
    MODEL_CONFIG=$(mktemp /tmp/hf_config_XXXX.json)
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "transformers", "model": "$HF_MODEL", "torch_dtype": "bfloat16"}
CONF

    # --- Both tracks (single model load) ---
    echo "[Step 3+3b] Running inference (generative + log-prob)..."
    python scripts/run_inference.py \
        --input "$TEST_INPUT" \
        --output-dir "$RESULT_DIR" \
        --backend-config "$MODEL_CONFIG" \
        --model-type "$MODEL_TYPE" \
        --model-name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" 2>&1

    if [ -f "$RESPONSES" ]; then
        echo "[Step 4] Evaluating with judge..."
        python scripts/evaluate.py \
            --input "$RESPONSES" \
            --questions "$TEST_INPUT" \
            --output "$EVALUATED" \
            --judge-config "$JUDGE_BACKEND" 2>&1
    else
        echo "[Step 4] Skipping evaluation (no responses generated)"
    fi

    rm -f "$MODEL_CONFIG"

    # --- Validate ---
    echo ""
    python -c "
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

    if [ $? -eq 0 ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_MODELS="$FAILED_MODELS $MODEL_NAME"
    fi
    echo ""
done

rm -f "$TEST_INPUT"

# --- Summary ---
echo "============================================================"
echo "SUMMARY: $PASSED passed, $FAILED failed out of ${#MODELS[@]} models"
if [ $FAILED -gt 0 ]; then
    echo "FAILED:$FAILED_MODELS"
fi
echo "============================================================"

# --- Step 6: Analysis & plots ---
ANALYSIS_DIR="data/results/${EXPERIMENT}/analysis"
echo ""
echo "=== GENERATING PLOTS ==="
python scripts/analyze.py \
    --results-dir "data/results/${EXPERIMENT}" \
    --output-dir "$ANALYSIS_DIR" 2>&1

echo ""
echo "Plots saved to: $ANALYSIS_DIR"
ls -la "$ANALYSIS_DIR"/*.png 2>/dev/null || echo "(no plots generated)"
