#!/bin/bash
# Run analysis + judge validation sampling on existing results.
#
# This is lightweight (no GPU, no API calls). Runs:
#   1. analyze.py — summaries, persistence, control conditions, plots
#   2. validate_judge.py — sample items for human annotation (CSV output)
#
# Usage:
#   bash slurm/run_analysis.sh                              # both datasets
#   bash slurm/run_analysis.sh --dataset computational      # single dataset
#   sbatch slurm/run_analysis.sh                            # via SLURM

#SBATCH --job-name=syco_analysis
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

set -euo pipefail

# --- Parse flags ---
EXPERIMENT="${EXPERIMENT:-exp1}"
DATASETS=("computational" "medical_advice")
JUDGE_SAMPLE_N="${JUDGE_SAMPLE_N:-50}"

for arg in "$@"; do
    case "$arg" in
        --dataset) NEXT_IS_DATASET=true ;;
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

cd "$PROJECT_DIR"
mkdir -p logs

TOOL=$(command -v apptainer || command -v singularity)

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
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

EXP_DIR="data/results/${EXPERIMENT}"

# --- Step 1: Run full analysis (per-dataset + cross-dataset) ---
echo "[Step 1] Running analyze.py (all datasets + cross-dataset)..."
run_in_container python scripts/analyze.py \
    --experiment-dir "$EXP_DIR"

for DATASET in "${DATASETS[@]}"; do
    RESULTS_DIR="${EXP_DIR}/${DATASET}"
    ANALYSIS_DIR="${RESULTS_DIR}/analysis"

    if [ ! -d "$RESULTS_DIR" ]; then
        echo "[SKIP] No results directory: $RESULTS_DIR"
        continue
    fi

    echo ""
    echo "  Outputs:"
    echo "    ${ANALYSIS_DIR}/summaries.json"
    echo "    ${ANALYSIS_DIR}/persistence_summaries.json"
    echo "    ${ANALYSIS_DIR}/control_summaries.json"
    echo "    ${ANALYSIS_DIR}/logprob_summaries.json"
    echo "    ${ANALYSIS_DIR}/*.png (plots)"

    # --- Step 2: Sample items for judge validation ---
    # Pick one instruct model for validation (or first available evaluated.jsonl)
    JUDGE_MODEL=""
    for candidate in "olmo3-7b-instruct" "olmo3-7b-think" "llama31-8b-instruct"; do
        if [ -f "${RESULTS_DIR}/${candidate}/evaluated.jsonl" ]; then
            JUDGE_MODEL="$candidate"
            break
        fi
    done

    if [ -z "$JUDGE_MODEL" ]; then
        # Fall back to first available
        for d in "${RESULTS_DIR}"/*/; do
            if [ -f "${d}evaluated.jsonl" ]; then
                JUDGE_MODEL=$(basename "$d")
                break
            fi
        done
    fi

    if [ -n "$JUDGE_MODEL" ]; then
        EVAL_FILE="${RESULTS_DIR}/${JUDGE_MODEL}/evaluated.jsonl"
        CSV_OUT="${ANALYSIS_DIR}/judge_validation_${DATASET}.csv"

        echo ""
        echo "[Step 2] Sampling ${JUDGE_SAMPLE_N} items from ${JUDGE_MODEL} for judge validation..."
        run_in_container python scripts/validate_judge.py sample \
            --input "$EVAL_FILE" \
            --output "$CSV_OUT" \
            --n "$JUDGE_SAMPLE_N"

        echo "  Output: $CSV_OUT"
        echo "  >> Fill in 'human_factual_accuracy' and 'human_agreement' columns"
        echo "  >> Then run: python scripts/validate_judge.py compute --input $CSV_OUT"
    else
        echo "[Step 2] No evaluated.jsonl found for judge validation sampling."
    fi

    echo ""
    echo "=== Done: $DATASET ==="
done

echo ""
echo "=== All analysis complete ==="
echo ""
echo "Next steps:"
echo "  1. Review analysis JSONs in data/results/${EXPERIMENT}/*/analysis/"
echo "  2. Annotate the judge_validation CSVs (fill human_factual_accuracy column)"
echo "  3. Run: python scripts/validate_judge.py compute --input <annotated_csv>"
