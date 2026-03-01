#!/bin/bash
#SBATCH --job-name=test_analysis
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Final analysis step — runs after all per-model test jobs complete.
# Called by test_all_models.sh via SLURM dependency — not meant to be run directly.
#
# Required env vars: EXPERIMENT, RESULT_BASE, TEST_INPUT

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"

TOOL=$(command -v apptainer || command -v singularity)

cd "$PROJECT_DIR"

# --- Summary ---
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_MODELS=""

for MODEL_DIR in "$RESULT_BASE"/*/; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    [ "$MODEL_NAME" = "analysis" ] && continue

    EVALUATED="$MODEL_DIR/evaluated.jsonl"
    LOGPROB_SCORES="$MODEL_DIR/logprob_scores.jsonl"

    if [ -f "$EVALUATED" ] && [ -f "$LOGPROB_SCORES" ]; then
        echo "  PASSED: $MODEL_NAME"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED: $MODEL_NAME (missing output files)"
        FAILED=$((FAILED + 1))
        FAILED_MODELS="$FAILED_MODELS $MODEL_NAME"
    fi
done

echo ""
echo "TOTAL: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo "FAILED:$FAILED_MODELS"
fi
echo "============================================================"

# --- Analysis & plots ---
ANALYSIS_DIR="$RESULT_BASE/analysis"
echo ""
echo "=== GENERATING PLOTS ==="

"$TOOL" exec \
    --containall \
    -B "$PROJECT_DIR:/workspace" \
    -B "/scratch/users/$USER:/scratch_user" \
    --home /scratch_user \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/workspace" \
    --pwd /workspace \
    "$SIF_IMAGE" \
    bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && python3 scripts/analyze.py \
        --results-dir '$RESULT_BASE' \
        --output-dir '$ANALYSIS_DIR'" 2>&1

echo ""
echo "Plots saved to: $ANALYSIS_DIR"
ls -la "$ANALYSIS_DIR"/*.png 2>/dev/null || echo "(no plots generated)"

# Clean up test input
rm -f "$TEST_INPUT"
