#!/bin/bash
#SBATCH --job-name=syco_err_subclassify
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sub-classify erroneous challenge responses (Err. metric) via GPT-4o judge.
# Classifies each initially-correct / challenge-erroneous response as:
#   apology_capitulation | format_incoherence | truncation_refusal | other
#
# Runs on both datasets, then produces a LaTeX breakdown table.
# Resume-safe: re-running skips already-classified items.
#
# Usage:
#   sbatch slurm/run_err_subclassifier.sh
#   bash   slurm/run_err_subclassifier.sh          # local/interactive
#
# Optional env overrides:
#   EXPERIMENT=exp1 (default)
#   INCLUDE_ABLATION=1  -- also include exp_citation_ablation results

set -euo pipefail

# --- Config ---
EXPERIMENT="${EXPERIMENT:-exp1}"
INCLUDE_ABLATION="${INCLUDE_ABLATION:-0}"
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"
EXP_DIR="data/results/${EXPERIMENT}"
OUTPUT_DIR="data/results"

export TMPDIR="/scratch/users/$USER/tmp"
mkdir -p "$TMPDIR" logs

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

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
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

# --- Sub-classify: computational ---
echo "[1/3] Sub-classifying Err. responses: computational..."

EVAL_PATHS="${EXP_DIR}/computational/*/evaluated.jsonl"
if [ "${INCLUDE_ABLATION}" = "1" ]; then
    EVAL_PATHS="${EVAL_PATHS} data/results/exp_citation_ablation/computational/*/evaluated.jsonl"
fi

run_in_container python scripts/classify_err_responses.py \
    --evaluated $EVAL_PATHS \
    --questions data/processed/computational.jsonl \
    --output "${OUTPUT_DIR}/err_subclassified_computational.jsonl" \
    --judge-config "$JUDGE_BACKEND" \
    --domain computational

echo "  -> ${OUTPUT_DIR}/err_subclassified_computational.jsonl"

# --- Sub-classify: medical ---
echo "[2/3] Sub-classifying Err. responses: medical..."

EVAL_PATHS="${EXP_DIR}/medical_advice/*/evaluated.jsonl"
if [ "${INCLUDE_ABLATION}" = "1" ]; then
    EVAL_PATHS="${EVAL_PATHS} data/results/exp_citation_ablation/medical_advice/*/evaluated.jsonl"
fi

run_in_container python scripts/classify_err_responses.py \
    --evaluated $EVAL_PATHS \
    --questions data/processed/medical_advice.jsonl \
    --output "${OUTPUT_DIR}/err_subclassified_medical.jsonl" \
    --judge-config "$JUDGE_BACKEND" \
    --domain medical

echo "  -> ${OUTPUT_DIR}/err_subclassified_medical.jsonl"

# --- Analyze + emit LaTeX table ---
echo "[3/3] Analyzing sub-type breakdown and generating LaTeX table..."

run_in_container python scripts/analyze_err_subtypes.py \
    --input \
        "${OUTPUT_DIR}/err_subclassified_computational.jsonl" \
        "${OUTPUT_DIR}/err_subclassified_medical.jsonl" \
    --latex-out "${OUTPUT_DIR}/err_subtype_table.tex"

echo "  -> ${OUTPUT_DIR}/err_subtype_table.tex"
echo ""
echo "=== Done. Check logs for apology_capitulation % on medical IC rows. ==="
echo "Paste ${OUTPUT_DIR}/err_subtype_table.tex into paper appendix (app:err_examples)."
