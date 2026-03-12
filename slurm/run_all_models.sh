#!/bin/bash
# Tracing Sycophancy - Submit FULL experiment for all models via SLURM.
#
# Submits one GPU job per (model, dataset) combination, then runs analysis.
# Each job uses the Apptainer container with vLLM for fast inference.
#
# Prerequisites: run slurm/setup_container.sh first
#
# Usage:
#   bash slurm/run_all_models.sh                    # all models
#   bash slurm/run_all_models.sh --dataset medical_advice   # single dataset

set -euo pipefail

# --- Parse flags ---
EXPERIMENT="exp1"
DATASETS=("medical_advice")
PARTITION="${PARTITION:-roxanad}"
for arg in "$@"; do
    case "$arg" in
        --dataset)  NEXT_IS_DATASET=true ;;
        *)
            if [ "${NEXT_IS_DATASET:-}" = true ]; then
                DATASETS=("$arg")
                NEXT_IS_DATASET=false
            fi
            ;;
    esac
done

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"

# --- All models: name|hf_model_id|model_type|checkpoint ---
MODELS=(
    # OLMo 3 7B — base
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    # OLMo 3 7B — Think pipeline (base → SFT → DPO → Think)
    "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    # OLMo 3 7B — Instruct pipeline (base → SFT → DPO → Instruct)
    "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    # Llama 3.1 8B — base + instruct
    "llama31-8b-base|meta-llama/Llama-3.1-8B|base|base"
    "llama31-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|chat|instruct"
    # Qwen 3.5 9B — base + instruct (requires vLLM upgrade for transformers>=5.2)
    # "qwen35-9b-base|Qwen/Qwen3.5-9B-Base|base|base"
    # "qwen35-9b-instruct|Qwen/Qwen3.5-9B|chat|instruct"
)

cd "$PROJECT_DIR"
mkdir -p logs

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

TOOL=$(command -v apptainer || command -v singularity)

# --- Ensure preprocessing is done ---
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

for DATASET in "${DATASETS[@]}"; do
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
        echo "[Prep] Generating challenges for $DATASET..."
        run_in_container python scripts/generate_challenges.py \
            --input "$PROCESSED" \
            --output "$PROCESSED" \
            --backend-config "config/models/gpt4o_judge.json" \
            --challenge-type factual
    else
        echo "[Prep] Skipping challenge generation for $DATASET (already present)"
    fi
done

# --- Submit one SLURM job per model × dataset ---
echo ""
echo "=== Submitting ${#MODELS[@]} models × ${#DATASETS[@]} datasets ==="
echo ""

JOB_IDS=()
for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"

        JOB_ID=$(sbatch --parsable \
            --job-name="syco_${MODEL_NAME}" \
            --partition="$PARTITION" \
            --time=24:00:00 \
            --mem=64G \
            --cpus-per-task=4 \
            --gpus=1 \
            -C GPU_MEM:80GB \
            --output="logs/${MODEL_NAME}_%j.out" \
            --error="logs/${MODEL_NAME}_%j.err" \
            --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT",EXPERIMENT="$EXPERIMENT",DATASET="$DATASET" \
            slurm/run_experiment.sh)

        echo "  Submitted $MODEL_NAME ($DATASET) → job $JOB_ID"
        JOB_IDS+=("$JOB_ID")
    done
done

# --- Submit analysis job per dataset, dependent on all model jobs ---
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")

for DATASET in "${DATASETS[@]}"; do
    ANALYSIS_JOB=$(sbatch --parsable \
        --job-name="syco_analysis_${DATASET}" \
        --partition=normal \
        --time=00:30:00 \
        --mem=8G \
        --cpus-per-task=2 \
        --output="logs/analysis_${DATASET}_%j.out" \
        --error="logs/analysis_${DATASET}_%j.err" \
        --dependency=afterany:"$DEPENDENCY" \
        --export=ALL \
        --wrap="
            cd $PROJECT_DIR
            TOOL=\$(command -v apptainer || command -v singularity)
            \$TOOL exec \
                --containall \
                -B '$PROJECT_DIR:/workspace' \
                -B '/scratch/users/$USER:/scratch_user' \
                --home /scratch_user \
                --env 'PYTHONNOUSERSITE=1' \
                --env 'PYTHONPATH=/workspace' \
                --pwd /workspace \
                '$SIF_IMAGE' \
                bash -c 'source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && python3 scripts/analyze.py --results-dir data/results/$EXPERIMENT/$DATASET --output-dir data/results/$EXPERIMENT/$DATASET/analysis'
        ")
    echo "  Submitted analysis ($DATASET) → job $ANALYSIS_JOB"
done

echo ""
echo "=== ${#JOB_IDS[@]} model jobs + ${#DATASETS[@]} analysis jobs submitted ==="
echo "=== Monitor with: squeue -u $USER ==="
echo "=== Results will be in: data/results/$EXPERIMENT/<dataset>/analysis/ ==="
