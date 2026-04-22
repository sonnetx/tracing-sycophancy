#!/bin/bash
# Length analysis: characterize (correct_answer, proposed_answer) length differences
# and recompute log-probability summaries on the length-matched subset.
# No GPU, no API cost — just reads existing processed and logprob data.
#
# Usage:
#   bash slurm/run_length_analysis.sh
#   bash slurm/run_length_analysis.sh --threshold 0.25

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-exp1}"
THRESHOLD=0.5

NEXT_IS_THRESHOLD=false
for arg in "$@"; do
    if [ "$NEXT_IS_THRESHOLD" = true ]; then
        THRESHOLD="$arg"; NEXT_IS_THRESHOLD=false; continue
    fi
    case "$arg" in
        --threshold) NEXT_IS_THRESHOLD=true ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
EXP_DIR="data/results/${EXPERIMENT}"

cd "$PROJECT_DIR"

TOOL=$(command -v apptainer || command -v singularity)

"$TOOL" exec \
    --containall \
    -B "$PROJECT_DIR:/workspace" \
    -B "/scratch/users/$USER:/scratch_user" \
    -B "/scratch/users/$USER/tmp:/tmp" \
    --home /scratch_user \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/workspace" \
    --env "HF_HOME=/scratch_user/huggingface" \
    --pwd /workspace \
    "$SIF_IMAGE" \
    bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && python scripts/length_analysis.py --experiment-dir $EXP_DIR --threshold $THRESHOLD"
