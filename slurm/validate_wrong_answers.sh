#!/bin/bash
#SBATCH --job-name=validate_wrong_answers
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Validate generated wrong answers (CPU-only, no GPU needed).
# Uses the Apptainer container.
#
# Computational: SymPy equivalence check (no API key needed).
# Medical:       GPT-4o judge classification (~500 calls, ~$0.05).
#                Resume-safe: cached_results.jsonl preserves progress on restart.
#
# Usage:
#   sbatch slurm/validate_wrong_answers.sh                                          # both datasets
#   sbatch --export=ALL,DATASET=computational slurm/validate_wrong_answers.sh
#   sbatch --export=ALL,DATASET=medical_advice slurm/validate_wrong_answers.sh
#   sbatch --export=ALL,DATASET=medical_advice,ANNOTATION_SAMPLE=50 slurm/validate_wrong_answers.sh

set -euo pipefail

# Accept both --export env vars (sbatch) and CLI flags (bash direct)
DATASET="${DATASET:-}"
ANNOTATION_SAMPLE="${ANNOTATION_SAMPLE:-0}"
PREV=""
for arg in "$@"; do
    case "$arg" in
        --dataset)           PREV="--dataset" ;;
        --annotation-sample) PREV="--annotation-sample" ;;
        *)
            if [ "$PREV" = "--dataset" ]; then DATASET="$arg"
            elif [ "$PREV" = "--annotation-sample" ]; then ANNOTATION_SAMPLE="$arg"
            fi
            PREV=""
            ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"
VENV_DIR="/scratch/users/$USER/container_env"
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

# Ensure sympy + antlr4 (needed for parse_latex) are available
echo "Checking sympy + antlr4..."
run_in_container python -c "import sympy; from sympy.parsing.latex import parse_latex" 2>/dev/null || {
    echo "Installing sympy and antlr4 into container venv..."
    run_in_container pip install --quiet 'sympy>=1.12' 'antlr4-python3-runtime==4.11.1'
}

validate_dataset() {
    local ds="$1"
    local input="data/processed/${ds}.jsonl"
    local output_dir="data/validation/${ds}"

    if [ ! -f "$input" ]; then
        echo "ERROR: $input not found."
        exit 1
    fi

    echo ""
    echo "=== Validating wrong answers: $ds ==="

    if [ "$ds" = "computational" ]; then
        run_in_container python scripts/validate_wrong_answers.py \
            --input "$input" \
            --dataset computational \
            --output-dir "$output_dir"
    else
        local sample_flag=""
        if [ "$ANNOTATION_SAMPLE" -gt 0 ]; then
            sample_flag="--annotation-sample $ANNOTATION_SAMPLE"
        fi
        run_in_container python scripts/validate_wrong_answers.py \
            --input "$input" \
            --dataset medical_advice \
            --judge-config "$JUDGE_BACKEND" \
            --output-dir "$output_dir" \
            $sample_flag
    fi

    echo "[$ds] Report written to $output_dir/"
}

if [ -z "$DATASET" ]; then
    validate_dataset "computational"
    validate_dataset "medical_advice"
elif [ "$DATASET" = "computational" ]; then
    validate_dataset "computational"
elif [ "$DATASET" = "medical_advice" ]; then
    validate_dataset "medical_advice"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi
