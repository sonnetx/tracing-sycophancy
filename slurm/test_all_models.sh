#!/bin/bash
# Smoke test: runs both tracks for OLMo models in PARALLEL via SLURM.
# Submits one GPU job per model, waits for all to finish, then runs analysis.
#
# Prerequisites: run slurm/setup_container.sh first
#
# Usage:
#   bash slurm/test_all_models.sh                    # all 7 models, 20 items
#   bash slurm/test_all_models.sh --quick             # base+instruct only, 5 items
#   bash slurm/test_all_models.sh -n 10               # all models, 10 items
#   bash slurm/test_all_models.sh --quick -n 3        # 2 models, 3 items
#   bash slurm/test_all_models.sh --transformers      # use transformers backend

set -euo pipefail

# --- Parse flags ---
BACKEND_TYPE="vllm"
QUICK=false
N_ITEMS=20
for arg in "$@"; do
    case "$arg" in
        --transformers) BACKEND_TYPE="transformers" ;;
        --vllm) BACKEND_TYPE="vllm" ;;
        --quick) QUICK=true; N_ITEMS=5 ;;
        -n) NEXT_IS_N=true ;;
        *)
            if [ "${NEXT_IS_N:-}" = true ]; then
                N_ITEMS="$arg"
                NEXT_IS_N=false
            fi
            ;;
    esac
done

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"

# --- Fixed test settings ---
DATASET="medical_advice"
EXPERIMENT="test_all_models"
JUDGE_BACKEND="config/models/gpt4o_judge.json"

# --- All OLMo 3 models: name|hf_model_id|model_type|checkpoint ---
ALL_MODELS=(
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
)

# --quick: just base + final instruct (enough to see if sycophancy differs)
QUICK_MODELS=(
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
)

if [ "$QUICK" = true ]; then
    MODELS=("${QUICK_MODELS[@]}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

cd "$PROJECT_DIR"
mkdir -p logs

# --- Ensure preprocessing is done ---
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"
TOOL=$(command -v apptainer || command -v singularity)

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

run_in_container() {
    "$TOOL" exec \
        --containall \
        -B "$PROJECT_DIR:/workspace" \
        -B "/scratch/users/$USER:/scratch_user" \
        -B "/scratch/users/$USER/tmp:/tmp" \
        --home /scratch_user \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=/workspace" \
        --env "HF_HOME=/scratch_user/huggingface" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

# Clean previous test results
RESULT_BASE="data/results/${EXPERIMENT}"
if [ -d "$RESULT_BASE" ]; then
    echo "Clearing previous test results: $RESULT_BASE"
    rm -rf "$RESULT_BASE"
fi
mkdir -p "$RESULT_BASE"

PROCESSED="data/processed/${DATASET}.jsonl"
if [ ! -f "$PROCESSED" ]; then
    echo "[Prep] Preprocessing $DATASET..."
    run_in_container python scripts/preprocess.py \
        --dataset "$DATASET" \
        --raw-path "data/raw/$DATASET" \
        --output "$PROCESSED" \
        --sample-size 500
else
    echo "[Prep] Skipping preprocessing ($PROCESSED exists)"
fi

if ! head -1 "$PROCESSED" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'challenges' in d and d['challenges'] and 'PLACEHOLDER' not in d['challenges'][0].get('prompt','') else 1)" 2>/dev/null; then
    echo "[Prep] Generating challenges..."
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED" \
        --output "$PROCESSED" \
        --backend-config "$JUDGE_BACKEND" \
        --challenge-type factual
else
    echo "[Prep] Skipping challenge generation (already present)"
fi

# Extract test subset
TEST_INPUT="$RESULT_BASE/test_input.jsonl"
head -"$N_ITEMS" "$PROCESSED" > "$TEST_INPUT"
ACTUAL=$(wc -l < "$TEST_INPUT")
echo ""
echo "=== Submitting ${#MODELS[@]} parallel jobs × $ACTUAL items (backend=$BACKEND_TYPE) ==="
if [ "$QUICK" = true ]; then
    echo "    (--quick mode: base + instruct only)"
fi
echo ""

# --- Submit one SLURM job per model ---
JOB_IDS=()
for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"

    JOB_ID=$(sbatch --parsable \
        --job-name="test_${MODEL_NAME}" \
        --partition=gpu \
        --time=02:00:00 \
        --mem=64G \
        --cpus-per-task=4 \
        --gpus=1 \
        -C GPU_MEM:24GB \
        --output="logs/test_${MODEL_NAME}_%j.out" \
        --error="logs/test_${MODEL_NAME}_%j.err" \
        --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT",EXPERIMENT="$EXPERIMENT",TEST_INPUT="$TEST_INPUT",JUDGE_BACKEND="$JUDGE_BACKEND",BACKEND_TYPE="$BACKEND_TYPE" \
        slurm/test_single_model.sh)

    echo "  Submitted $MODEL_NAME → job $JOB_ID"
    JOB_IDS+=("$JOB_ID")
done

# --- Submit analysis job dependent on all model jobs ---
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
ANALYSIS_JOB=$(sbatch --parsable \
    --job-name="test_analysis" \
    --partition=normal \
    --time=00:30:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="logs/test_analysis_%j.out" \
    --error="logs/test_analysis_%j.err" \
    --dependency=afterany:"$DEPENDENCY" \
    --export=ALL,EXPERIMENT="$EXPERIMENT",RESULT_BASE="$RESULT_BASE",TEST_INPUT="$TEST_INPUT" \
    slurm/test_analyze.sh)

echo ""
echo "  Submitted analysis → job $ANALYSIS_JOB (runs after all models finish)"
echo ""
echo "=== Monitor with: squeue -u $USER ==="
echo "=== Results will be in: $RESULT_BASE/analysis/ ==="
