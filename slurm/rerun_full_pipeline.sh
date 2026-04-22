#!/bin/bash
# Rerun Full Pipeline — archives old results, regenerates challenges, reruns everything.
#
# Everything runs via SLURM — nothing blocks on the login node.
#
# Job chain:
#   1. (login node) Archive old results + strip challenges from processed data
#   2. (SLURM) Prep job: preprocess + regenerate challenges
#   3. (SLURM) Model jobs: inference + evaluation (depend on prep)
#   4. (SLURM) Analysis job (depends on all model jobs)
#
# Prerequisites: run slurm/setup_container.sh first
#
# Usage:
#   bash slurm/rerun_full_pipeline.sh
#   bash slurm/rerun_full_pipeline.sh --dataset medical_advice
#   bash slurm/rerun_full_pipeline.sh --dry-run

set -euo pipefail

# --- Parse flags ---
EXPERIMENT="exp1"
DATASETS=("computational" "medical_advice")
PARTITION="${PARTITION:-roxanad}"
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dataset)  NEXT_IS_DATASET=true ;;
        --dry-run)  DRY_RUN=true ;;
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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JUDGE_BACKEND="config/models/gpt4o_judge.json"

cd "$PROJECT_DIR"
mkdir -p logs

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

# --- All models: name|hf_model_id|model_type|checkpoint ---
MODELS=(
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    "llama31-8b-base|meta-llama/Llama-3.1-8B|base|base"
    "llama31-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|chat|instruct"
    # Tulu 3 on Llama 3.1 8B — SFT → DPO → Final (same base as above, AI2 alignment)
    "tulu3-llama31-8b-sft|allenai/Llama-3.1-Tulu-3-8B-SFT|chat|sft"
    "tulu3-llama31-8b-dpo|allenai/Llama-3.1-Tulu-3-8B-DPO|chat|dpo"
    "tulu3-llama31-8b|allenai/Llama-3.1-Tulu-3-8B|chat|instruct"
)

# =====================================================================
# Step 1 (login node): Archive old results + strip challenges
# =====================================================================
RESULT_DIR="data/results/$EXPERIMENT"
if [ -d "$RESULT_DIR" ]; then
    ARCHIVE_DIR="data/results/${EXPERIMENT}_archive_${TIMESTAMP}"
    echo "[Archive] Moving $RESULT_DIR → $ARCHIVE_DIR"
    if [ "$DRY_RUN" = false ]; then
        mv "$RESULT_DIR" "$ARCHIVE_DIR"
        echo "[Archive] Done. Old results preserved at: $ARCHIVE_DIR"
    else
        echo "[Archive] (dry-run) Would move $RESULT_DIR → $ARCHIVE_DIR"
    fi
else
    echo "[Archive] No existing results at $RESULT_DIR — nothing to archive."
fi

for DATASET in "${DATASETS[@]}"; do
    PROCESSED="data/processed/${DATASET}.jsonl"
    if [ -f "$PROCESSED" ]; then
        echo "[Prep] Stripping challenges from $PROCESSED..."
        if [ "$DRY_RUN" = false ]; then
            cp "$PROCESSED" "${PROCESSED}.bak_${TIMESTAMP}"
            python3 -c "
import json
input_path = '$PROCESSED'
lines = open(input_path).readlines()
out = []
for line in lines:
    d = json.loads(line)
    d.pop('challenges', None)
    out.append(json.dumps(d))
with open(input_path, 'w') as f:
    for line in out:
        f.write(line + '\n')
print(f'  Stripped challenges from {len(out)} items')
"
        else
            echo "[Prep] (dry-run) Would strip challenges from $PROCESSED"
        fi
    else
        echo "[Prep] $PROCESSED not found — will be created by prep job."
    fi
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[Dry-run] Would submit: 1 prep job + ${#MODELS[@]} model jobs + ${#DATASETS[@]} analysis jobs"
    echo "Done (dry-run). No changes made."
    exit 0
fi

# =====================================================================
# Step 2 (SLURM): Prep job — preprocess + generate challenges
# =====================================================================
TOOL_CMD="\$(command -v apptainer || command -v singularity)"
CONTAINER_PREFIX="
cd $PROJECT_DIR
TOOL=$TOOL_CMD
\$TOOL exec \\
    --containall \\
    -B '$PROJECT_DIR:/workspace' \\
    -B '/scratch/users/$USER:/scratch_user' \\
    -B '/scratch/users/$USER/tmp:/tmp' \\
    --home /scratch_user \\
    --env 'PYTHONNOUSERSITE=1' \\
    --env 'PYTHONPATH=/workspace' \\
    --env 'HF_HOME=/scratch_user/huggingface' \\
    --env 'HF_TOKEN=${HF_TOKEN:-}' \\
    --env 'OPENAI_API_KEY=${OPENAI_API_KEY:-}' \\
    --env 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}' \\
    --pwd /workspace \\
    '$SIF_IMAGE' \\
    bash -c 'source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace &&"

echo ""
echo "=== Submitting pipeline: ${#MODELS[@]} models × ${#DATASETS[@]} datasets ==="
echo ""

# Dataset-specific ethos statements (Fanous et al., 2025)
declare -A ETHOS_MAP
ETHOS_MAP[computational]="an expert in math"
ETHOS_MAP[medical_advice]="an expert in medicine"
PREP_CMDS=""
for DATASET in "${DATASETS[@]}"; do
    PROCESSED="data/processed/${DATASET}.jsonl"
    ETHOS="${ETHOS_MAP[$DATASET]:-an expert}"
    PREP_CMDS+="
echo \"[Prep] Preprocessing + generating challenges for $DATASET...\"
"
    # Preprocess if needed
    PREP_CMDS+="
if [ ! -f '$PROCESSED' ]; then
    echo '[Prep] Preprocessing $DATASET...'
    $CONTAINER_PREFIX python scripts/preprocess.py --dataset $DATASET --raw-path data/raw/$DATASET --output $PROCESSED --sample-size 500'
fi
"
    # Generate challenges (always, since we stripped them)
    PREP_CMDS+="
echo '[Prep] Generating challenges for $DATASET...'
$CONTAINER_PREFIX python scripts/generate_challenges.py --input $PROCESSED --output $PROCESSED --backend-config $JUDGE_BACKEND --challenge-type factual --ethos \"$ETHOS\"'
"
done

PREP_JOB=$(sbatch --parsable \
    --job-name="syco_prep" \
    --partition=normal \
    --time=00:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="logs/prep_${TIMESTAMP}_%j.out" \
    --error="logs/prep_${TIMESTAMP}_%j.err" \
    --export=ALL \
    --wrap="$PREP_CMDS")

echo "  Submitted prep job → $PREP_JOB"

# =====================================================================
# Step 3 (SLURM): Model jobs — inference + evaluation (depend on prep)
# =====================================================================
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
            --dependency=afterok:"$PREP_JOB" \
            --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT",EXPERIMENT="$EXPERIMENT",DATASET="$DATASET" \
            slurm/run_experiment.sh)

        echo "  Submitted $MODEL_NAME ($DATASET) → job $JOB_ID (after prep:$PREP_JOB)"
        JOB_IDS+=("$JOB_ID")
    done
done

# =====================================================================
# Step 4 (SLURM): Analysis job (depends on all model jobs)
# =====================================================================
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
echo "=== Job chain: prep($PREP_JOB) → ${#JOB_IDS[@]} model jobs → analysis ==="
echo "=== Monitor with: squeue -u $USER ==="
echo "=== Results will be in: data/results/$EXPERIMENT/<dataset>/analysis/ ==="
