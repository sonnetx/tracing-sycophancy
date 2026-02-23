#!/bin/bash
#SBATCH --job-name=syco_run
#SBATCH --partition=roxanad
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Tracing Sycophancy - Run Experiment Pipeline
#
# Loads a HuggingFace model directly on GPU via transformers and runs
# the full pipeline. No separate server needed.
#
# Usage:
#   sbatch slurm/run_experiment.sh
#   sbatch --export=ALL,HF_MODEL=meta-llama/Llama-3.2-3B-Instruct,MODEL_NAME=llama-3.2-3b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct slurm/run_experiment.sh
#   sbatch --export=ALL,HF_MODEL=allenai/OLMo-7B,MODEL_NAME=olmo-7b-base,MODEL_TYPE=base,CHECKPOINT=base slurm/run_experiment.sh

# --- Modules ---
ml python/3.12.1
ml cuda/12.4.0

# --- Paths ---
VENV_DIR="${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}"
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# --- Model (HuggingFace model ID) ---
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-3B}"
MODEL_NAME="${MODEL_NAME:-llama-3.2-3b-base}"
MODEL_TYPE="${MODEL_TYPE:-base}"
CHECKPOINT="${CHECKPOINT:-base}"
REVISION="${REVISION:-}"  # HF revision (e.g. step1000-tokens4B for OLMo checkpoints)

# --- Dataset ---
DATASET="${DATASET:-computational}"
RAW_PATH="${RAW_PATH:-data/raw/$DATASET}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
EXPERIMENT="${EXPERIMENT:-exp1}"

# --- Judge & Challenge Generation ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-$JUDGE_BACKEND}"  # reuse judge model for challenge gen

# --- Activate environment ---
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Cache dirs — keep everything on scratch, not home
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
export MODEL_DIR="/scratch/users/$USER/models"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MODEL_DIR"

# Source API keys
if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== Experiment: $EXPERIMENT | Dataset: $DATASET | Model: $MODEL_NAME ==="
echo "HuggingFace model: $HF_MODEL"
echo "Model type: $MODEL_TYPE | Checkpoint: $CHECKPOINT | Revision: ${REVISION:-none}"

# Create model config for the transformers backend
MODEL_CONFIG=$(mktemp /tmp/hf_config_XXXX.json)
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "transformers", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16"}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "transformers", "model": "$HF_MODEL", "torch_dtype": "bfloat16"}
CONF
fi
trap "rm -f $MODEL_CONFIG" EXIT

# =====================================================================
# Pipeline
# =====================================================================

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
if ! head -1 "$PROCESSED" | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'challenges' in d and d['challenges'] and 'PLACEHOLDER' not in d['challenges'][0].get('prompt','') else 1)" 2>/dev/null; then
    echo "[Step 2] Generating challenges with backend..."
    python scripts/generate_challenges.py \
        --input "$PROCESSED" \
        --output "$PROCESSED" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual
else
    echo "[Step 2] Skipping challenge generation (already present)"
fi

# --- Step 3+3b: Run both tracks (single model load) ---
RESULT_DIR="data/results/${EXPERIMENT}/${MODEL_NAME}"
RESPONSES="$RESULT_DIR/responses.jsonl"
LOGPROB_SCORES="$RESULT_DIR/logprob_scores.jsonl"
echo "[Step 3+3b] Running inference (generative + log-prob) with $MODEL_NAME..."
python scripts/run_inference.py \
    --input "$PROCESSED" \
    --output-dir "$RESULT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --resume

# --- Step 4: Evaluate ---
EVALUATED="$RESULT_DIR/evaluated.jsonl"
if [ -f "$RESPONSES" ]; then
    echo "[Step 4] Evaluating responses..."
    python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$PROCESSED" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND"
else
    echo "[Step 4] Skipping evaluation (no responses generated)"
fi

# --- Step 5: Analyze ---
ANALYSIS_DIR="data/results/${EXPERIMENT}/analysis"
echo "[Step 5] Running analysis..."
python scripts/analyze.py \
    --results-dir "data/results/${EXPERIMENT}" \
    --output-dir "$ANALYSIS_DIR"

echo "=== Done: $MODEL_NAME on $DATASET ==="
echo "Results: $EVALUATED"
echo "Analysis: $ANALYSIS_DIR"
