#!/bin/bash
# Behavioural sycophancy breakdown by challenge context.
# Computes regressive / control / net rates split by in-context vs preemptive.
#
# Usage:
#   bash slurm/run_behavioral_breakdown.sh
#   bash slurm/run_behavioral_breakdown.sh --latex

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-exp1}"
EXTRA_ARGS="$*"

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
    bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && python scripts/behavioral_breakdown.py --experiment-dir $EXP_DIR $EXTRA_ARGS"
