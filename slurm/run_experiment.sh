#!/bin/bash
#SBATCH --job-name=syco_run
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:24GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Tracing Sycophancy - Run Experiment Pipeline
#
# Uses Apptainer container with vLLM for fast inference.
# Run slurm/setup_container.sh first to set up the environment.
#
# Usage:
#   sbatch slurm/run_experiment.sh
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-1025-7B,MODEL_NAME=olmo3-7b-base,MODEL_TYPE=base,CHECKPOINT=base slurm/run_experiment.sh
#   sbatch --export=ALL,BACKEND_TYPE=vllm,HF_MODEL=allenai/Olmo-3-1025-7B,MODEL_NAME=olmo3-7b-base,MODEL_TYPE=base,CHECKPOINT=base slurm/run_experiment.sh

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"

# --- Model ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-1025-7B}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-base}"
MODEL_TYPE="${MODEL_TYPE:-base}"
CHECKPOINT="${CHECKPOINT:-base}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"

# --- Dataset ---
DATASET="${DATASET:-computational}"
RAW_PATH="${RAW_PATH:-data/raw/$DATASET}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
EXPERIMENT="${EXPERIMENT:-exp1}"

# --- Judge & Challenge Generation ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-$JUDGE_BACKEND}"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
export MODEL_DIR="/scratch/users/$USER/models"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MODEL_DIR" logs

# Source API keys
if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

TOOL=$(command -v apptainer || command -v singularity)

# Helper: run a command inside the container
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

echo "=== Experiment: $EXPERIMENT | Dataset: $DATASET | Model: $MODEL_NAME ==="
echo "HuggingFace model: $HF_MODEL"
echo "Model type: $MODEL_TYPE | Checkpoint: $CHECKPOINT | Revision: ${REVISION:-none} | Backend: $BACKEND_TYPE"

# Create model config — written inside project tree so the path works in the container
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"
MODEL_CONFIG="$RESULT_DIR/model_config.json"
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16"}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16"}
CONF
fi

# =====================================================================
# Pipeline
# =====================================================================

# --- Step 1: Preprocess (if not already done) ---
PROCESSED="data/processed/${DATASET}.jsonl"
if [ ! -f "$PROCESSED" ]; then
    echo "[Step 1] Preprocessing $DATASET..."
    run_in_container python scripts/preprocess.py \
        --dataset "$DATASET" \
        --raw-path "$RAW_PATH" \
        --output "$PROCESSED" \
        --sample-size "$SAMPLE_SIZE"
else
    echo "[Step 1] Skipping preprocessing ($PROCESSED exists)"
fi

# --- Step 2: Generate challenges (if not already done) ---
if ! head -1 "$PROCESSED" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'challenges' in d and d['challenges'] and 'PLACEHOLDER' not in d['challenges'][0].get('prompt','') else 1)" 2>/dev/null; then
    echo "[Step 2] Generating challenges with backend..."
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED" \
        --output "$PROCESSED" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual
else
    echo "[Step 2] Skipping challenge generation (already present)"
fi

# --- Step 3+3b: Run both tracks (single model load) ---
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
RESPONSES="$RESULT_DIR/responses.jsonl"
LOGPROB_SCORES="$RESULT_DIR/logprob_scores.jsonl"
echo "[Step 3+3b] Running inference (generative + log-prob) with $MODEL_NAME..."
run_in_container python scripts/run_inference.py \
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
    run_in_container python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$PROCESSED" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND"
else
    echo "[Step 4] Skipping evaluation (no responses generated)"
fi

echo "=== Done: $MODEL_NAME on $DATASET ==="
echo "Results: $RESULT_DIR"
