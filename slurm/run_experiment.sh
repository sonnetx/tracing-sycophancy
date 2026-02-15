#!/bin/bash
#SBATCH --job-name=syco_run
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Tracing Sycophancy - Run Experiment Pipeline
#
# Usage:
#   sbatch slurm/run_experiment.sh                              # defaults
#   sbatch --export=DATASET=medical_advice,MODEL_CONFIG=config/models/olmo_base.json slurm/run_experiment.sh

# --- Configuration (override via --export or environment) ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/tracing-sycophancy}"

DATASET="${DATASET:-computational}"
RAW_PATH="${RAW_PATH:-data/raw/$DATASET}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
EXPERIMENT="${EXPERIMENT:-exp1}"

MODEL_CONFIG="${MODEL_CONFIG:-config/models/llama_base.json}"
MODEL_NAME="${MODEL_NAME:-llama-3.1-8b-base}"
MODEL_TYPE="${MODEL_TYPE:-base}"
CHECKPOINT="${CHECKPOINT:-base}"

CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-config/models/ollama_challenge_gen.json}"
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Activate environment ---
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export HF_HOME="${HF_HOME:-/scratch/users/$USER/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/scratch/users/$USER/torch}"

cd "$PROJECT_DIR"

echo "=== Experiment: $EXPERIMENT | Dataset: $DATASET | Model: $MODEL_NAME ==="

# --- Step 1: Preprocess (if not already done) ---
PROCESSED="data/processed/${DATASET}.jsonl"
if [ ! -f "$PROCESSED" ]; then
    echo "[Step 1] Preprocessing $DATASET..."
    python scripts/preprocess.py \
        --dataset "$DATASET" \
        --raw-path "$RAW_PATH" \
        --output "$PROCESSED" \
        --sample-size "$SAMPLE_SIZE"
else
    echo "[Step 1] Skipping preprocessing ($PROCESSED exists)"
fi

# --- Step 2: Generate challenges (if not already done) ---
# Check if challenges field exists in the JSONL
if ! head -1 "$PROCESSED" | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'challenges' in d and d['challenges'] else 1)" 2>/dev/null; then
    echo "[Step 2] Generating challenges..."
    python scripts/generate_challenges.py \
        --input "$PROCESSED" \
        --output "$PROCESSED" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual
else
    echo "[Step 2] Skipping challenge generation (already present)"
fi

# --- Step 3: Generate responses ---
RESPONSES="data/results/${EXPERIMENT}/${MODEL_NAME}/responses.jsonl"
echo "[Step 3] Generating responses with $MODEL_NAME..."
python scripts/generate_responses.py \
    --input "$PROCESSED" \
    --output "$RESPONSES" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT"

# --- Step 4: Evaluate ---
EVALUATED="data/results/${EXPERIMENT}/${MODEL_NAME}/evaluated.jsonl"
echo "[Step 4] Evaluating responses..."
python scripts/evaluate.py \
    --input "$RESPONSES" \
    --questions "$PROCESSED" \
    --output "$EVALUATED" \
    --judge-config "$JUDGE_BACKEND"

echo "=== Done: $MODEL_NAME on $DATASET ==="
echo "Results: $EVALUATED"
