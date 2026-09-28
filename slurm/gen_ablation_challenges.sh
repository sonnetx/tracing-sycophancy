#!/bin/bash
#SBATCH --job-name=gen_ablation
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Generate citation ablation challenges (CPU-only, no GPU needed).
# Uses the Apptainer container.
#
# Usage:
#   sbatch slurm/gen_ablation_challenges.sh
#   sbatch --export=ALL,DATASET=computational slurm/gen_ablation_challenges.sh
#
# Usage:
#   bash slurm/gen_ablation_challenges.sh
#   bash slurm/gen_ablation_challenges.sh --dataset medical_advice

set -euo pipefail

# Accept both --export env vars (sbatch) and CLI flags (bash direct)
DATASET="${DATASET:-}"
PREV=""
for arg in "$@"; do
    case "$arg" in
        --dataset) PREV="--dataset" ;;
        *)
            if [ "$PREV" = "--dataset" ]; then DATASET="$arg"; fi
            PREV=""
            ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"
JUDGE_BACKEND="config/models/gpt4o_judge.json"

export TMPDIR="/scratch/users/$USER/tmp"
mkdir -p "$TMPDIR"

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

TOOL=$(command -v apptainer || command -v singularity)

cd "$PROJECT_DIR"

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
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

run_dataset() {
    local ds="$1"
    local ethos="$2"
    local input="data/processed/${ds}.jsonl"
    local output="data/processed/${ds}_ablation.jsonl"

    if [ ! -f "$input" ]; then
        echo "ERROR: $input not found. Run the main pipeline first."
        exit 1
    fi

    if python3 -c "
import json, sys
try:
    d = json.loads(open('$output').readline())
    sys.exit(0 if any(c['type']=='length_control' for c in d.get('challenges',[])) else 1)
except: sys.exit(1)
" 2>/dev/null; then
        echo "[$ds] Skipping — $output already has length_control challenges"
        return
    fi

    echo "[$ds] Generating ablation challenges (length_control + citation_no_doi)..."
    run_in_container python scripts/generate_challenges.py \
        --input "$input" \
        --output "$output" \
        --backend-config "$JUDGE_BACKEND" \
        --challenge-type factual \
        --ethos "'$ethos'" \
        --ablations
    echo "[$ds] Done → $output"
}

if [ -z "$DATASET" ]; then
    run_dataset "computational" "an expert in math"
    run_dataset "medical_advice" "an expert in medicine"
elif [ "$DATASET" = "computational" ]; then
    run_dataset "computational" "an expert in math"
elif [ "$DATASET" = "medical_advice" ]; then
    run_dataset "medical_advice" "an expert in medicine"
else
    echo "Unknown dataset: $DATASET. Use 'computational' or 'medical_advice'."
    exit 1
fi

echo ""
echo "=== Done. Now submit inference jobs: ==="
echo "  sbatch --export=ALL,HF_MODEL=allenai/OLMo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct,DATASET=computational slurm/run_citation_ablation.sh"
echo "  sbatch --export=ALL,HF_MODEL=allenai/OLMo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct,DATASET=medical_advice slurm/run_citation_ablation.sh"
