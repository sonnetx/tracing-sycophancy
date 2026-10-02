#!/bin/bash -l
#SBATCH --job-name=syco_judge_updated
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
TASK_DIR="${VALIDATION_TASK_DIR:-$GROUP_HOME/sonnet/tracing-sycophancy/review/judge-rerun-2026-10-02}"
export TMPDIR="$L_SCRATCH_JOB/tmp"
export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch"
export HF_DATASETS_CACHE="$SCRATCH/huggingface/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10 HF_HUB_DOWNLOAD_TIMEOUT=60
TASK_WORK="$L_SCRATCH_JOB/judge-validation"
mkdir -p "$TMPDIR" "$TASK_WORK" "$TASK_DIR/results"
tar -xzf "$TASK_DIR/inputs.tar.gz" -C "$TASK_WORK"
mkdir -p "$TASK_WORK/results"
# Each array task owns seven cells. Preserve completed outputs on failure.
preserve_results() { cp -f "$TASK_WORK/results/"*.json* "$TASK_DIR/results/" 2>/dev/null || true; }
trap preserve_results EXIT
if [ -f "$HOME/.secrets" ]; then
    set -a
    source "$HOME/.secrets"
    set +a
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
export APPTAINERENV_ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_ETAG_TIMEOUT=10 APPTAINERENV_HF_HUB_DOWNLOAD_TIMEOUT=60
apptainer exec --containall \
    -B "$TASK_WORK:/task" -B "$TASK_DIR:/code:ro" \
    -B "$SCRATCH:/scratch_user" --home /scratch_user --pwd /task \
    "$SCRATCH/simg/vllm-v0.11.0.sif" \
    /scratch_user/container_env/bin/python /code/rejudge_updated_outputs.py run \
    --input-dir /task/inputs --output-dir /task/results \
    --group "$SLURM_ARRAY_TASK_ID" --groups 3
