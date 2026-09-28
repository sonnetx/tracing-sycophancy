#!/bin/bash
#SBATCH --job-name=syco_robustness_analysis
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Final analysis step for all robustness checks (Checks 1–5).
# Submitted after all GPU/API inference jobs complete.
# No GPU or external API calls needed.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

export TMPDIR="/scratch/users/$USER/tmp"
mkdir -p "$TMPDIR" logs

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
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== Robustness checks analysis ==="

# Check 4 + main analysis with cluster-robust CIs (--ablation also triggers Check 5 analysis)
echo "[Check 4+5] Re-running analyze.py with cluster CIs and citation ablation..."
run_in_container python scripts/analyze.py \
    --experiment-dir data/results/exp1 \
    --ablation

# Check 1: candidate-dependence variance and rank stability
echo ""
echo "[Check 1] Candidate-dependence robustness..."
if [ -d data/results/exp_candidate_robustness ]; then
    run_in_container python scripts/analyze_candidate_robustness.py \
        --experiment-dir data/results/exp_candidate_robustness \
        --output data/results/exp_candidate_robustness/candidate_robustness_report.json
else
    echo "  SKIP: data/results/exp_candidate_robustness not found"
fi

# Check 2: inter-judge Cohen's κ
echo ""
echo "[Check 2] Inter-judge agreement..."
if [ -d data/results/exp_rejudge ]; then
    run_in_container python scripts/compare_judges.py \
        --experiment-dir data/results/exp_rejudge \
        --output data/results/exp_rejudge/judge_agreement.json
else
    echo "  SKIP: data/results/exp_rejudge not found"
fi

# Check 3: length-matched rank stability (medical only)
echo ""
echo "[Check 3] Length-matched ΔLogOdds..."
if [ -d data/results/exp_length_robustness ]; then
    run_in_container python scripts/analyze_length_robustness.py \
        --experiment-dir data/results/exp_length_robustness \
        --baseline-dir   data/results/exp1 \
        --dataset        medical_advice \
        --processed-path data/processed/medical_advice_length_matched.jsonl \
        --output         data/results/exp_length_robustness/analysis/length_robustness.json
else
    echo "  SKIP: data/results/exp_length_robustness not found"
fi

echo ""
echo "=== All robustness analysis complete ==="
echo "Reports:"
echo "  Check 1: data/results/exp_candidate_robustness/candidate_robustness_report.json"
echo "  Check 2: data/results/exp_rejudge/judge_agreement.json"
echo "  Check 3: data/results/exp_length_robustness/analysis/length_robustness.json"
echo "  Check 4: data/results/exp1/*/analysis/summaries.json (regressive_ci_cluster_* fields)"
echo "  Check 5: data/results/exp_citation_ablation/analysis/ablation_summary.json"
