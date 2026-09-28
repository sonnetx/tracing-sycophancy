#!/bin/bash
#SBATCH --job-name=syco_ablation
#SBATCH --partition=roxanad
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Citation confound ablation: citation_no_doi + length_control challenges.
#
# Runs on one representative model per pipeline (final-stage instruct checkpoints)
# on both datasets. The ablation JSONL is kept separate from the main processed
# file so main results are unaffected.
#
# Usage (run once per model):
#   sbatch --export=ALL,HF_MODEL=allenai/OLMo-3-1025-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=instruct,CHECKPOINT=final,DATASET=computational slurm/run_citation_ablation.sh
#   sbatch --export=ALL,HF_MODEL=allenai/OLMo-3-1025-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=instruct,CHECKPOINT=final,DATASET=medical_advice slurm/run_citation_ablation.sh
#
# Suggested models (one per pipeline, final checkpoint):
#   allenai/OLMo-3-1025-7B-Instruct   MODEL_NAME=olmo3-7b-instruct    CHECKPOINT=final
#   allenai/OLMo-3-1025-7B-SFT-Think  MODEL_NAME=olmo3-7b-think       CHECKPOINT=final  (optional)
#   meta-llama/Llama-3.1-8B-Instruct  MODEL_NAME=llama3-8b-instruct   CHECKPOINT=final
#   allenai/tulu-3-8b                 MODEL_NAME=tulu3-8b             CHECKPOINT=final

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"

# --- Model ---
HF_MODEL="${HF_MODEL:-allenai/OLMo-3-1025-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
MODEL_TYPE="${MODEL_TYPE:-instruct}"
CHECKPOINT="${CHECKPOINT:-final}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# --- Dataset ---
DATASET="${DATASET:-computational}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
EXPERIMENT="${EXPERIMENT:-exp_citation_ablation}"

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

echo "=== Citation ablation: $DATASET | $MODEL_NAME ==="

# The ablation processed file is separate from the main one so main results are unaffected.
# It reuses cached proposed_answer/justification/citation from the main file and only
# generates the new proposed_length_control field (~500 GPT-4o calls, ~$0.10).
PROCESSED_MAIN="data/processed/${DATASET}.jsonl"
PROCESSED_ABLATION="data/processed/${DATASET}_ablation.jsonl"

# --- Step 1: Generate ablation challenges (adds citation_no_doi + length_control) ---
# Input is the main processed file (reuses all cached GPT-4o content).
# Output is a separate ablation file — main file is never modified.
if [ -f "$PROCESSED_ABLATION" ] && \
   python3 -c "import sys,json; d=json.load(open('$PROCESSED_ABLATION')); sys.exit(0 if any(c['type']=='length_control' for c in d.get('challenges',[])) else 1)" 2>/dev/null; then
    echo "[Step 1] Skipping ablation challenge generation ($PROCESSED_ABLATION already has length_control challenges)"
else
    echo "[Step 1] Generating ablation challenges (length_control + citation_no_doi)..."
    case "$DATASET" in
        computational)   ETHOS="an expert in math" ;;
        medical_advice)  ETHOS="an expert in medicine" ;;
        *)               ETHOS="an expert" ;;
    esac
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED_MAIN" \
        --output "$PROCESSED_ABLATION" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual \
        --ethos "'$ETHOS'" \
        --ablations
fi

# --- Step 2: Create model config ---
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"
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

# --- Step 3: Inference on ablation challenges ---
echo "[Step 2] Running inference on ablation challenges with $MODEL_NAME..."
run_in_container python scripts/run_inference.py \
    --input "$PROCESSED_ABLATION" \
    --output-dir "$RESULT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --resume

# --- Step 4: Evaluate ---
RESPONSES="$RESULT_DIR/responses.jsonl"
EVALUATED="$RESULT_DIR/evaluated.jsonl"
if [ ! -f "$RESPONSES" ]; then
    echo "[Step 3] Skipping evaluation (no responses generated)"
elif [ -f "$EVALUATED" ] && [ "$(wc -l < "$RESPONSES")" -eq "$(wc -l < "$EVALUATED")" ]; then
    echo "[Step 3] Skipping evaluation (already complete)"
else
    echo "[Step 3] Evaluating responses..."
    run_in_container python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$PROCESSED_ABLATION" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND"
fi

echo "=== Done: $MODEL_NAME on $DATASET (ablation) ==="
echo "Results in: $RESULT_DIR"
echo "Compare citation vs citation_no_doi vs length_control in the evaluated output."
