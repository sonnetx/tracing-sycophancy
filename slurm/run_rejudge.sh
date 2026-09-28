#!/bin/bash
#SBATCH --job-name=syco_rejudge
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Check 2: Alternative-judge subset validation.
#
# Re-scores a stratified 20% sample of generative responses with Claude (Anthropic API)
# to bound the GPT-4o shared-model-artifact concern. Reports inter-judge Cohen's κ
# and whether Regr/Net rankings change under the alternative judge.
#
# No GPU inference needed — this is a CPU/API-only job that re-calls the judge LLM.
#
# Usage (run once per model/dataset pair; requires existing exp1 evaluated.jsonl):
#   sbatch --export=ALL,\
#     MODEL_NAME=olmo3-7b-instruct,\
#     DATASET=computational \
#     slurm/run_rejudge.sh
#
# To run all models for a dataset, loop:
#   for MODEL in olmo3-7b-base olmo3-7b-instruct llama31-8b-base llama31-8b-instruct; do
#     sbatch --export=ALL,MODEL_NAME=$MODEL,DATASET=computational slurm/run_rejudge.sh
#   done

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
DATASET="${DATASET:-computational}"
SRC_EXPERIMENT="${SRC_EXPERIMENT:-exp1}"
EXPERIMENT="${EXPERIMENT:-exp_rejudge}"
FRACTION="${FRACTION:-0.20}"
SEED="${SEED:-42}"

# Primary judge config (existing, GPT-4o)
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"
# Alternative judge config (Claude via Anthropic API)
ALT_JUDGE_BACKEND="${ALT_JUDGE_BACKEND:-config/models/claude_judge.json}"

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
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

SRC_DIR="data/results/${SRC_EXPERIMENT}/${DATASET}/${MODEL_NAME}"
OUT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

echo "=== Alternative-judge re-scoring: $DATASET | $MODEL_NAME ==="

EVALUATED="${SRC_DIR}/evaluated.jsonl"
RESPONSES="${SRC_DIR}/responses.jsonl"
GPT4O_SAMPLE="${OUT_DIR}/gpt4o_sample.jsonl"
RESP_SAMPLE="${OUT_DIR}/sample_responses.jsonl"
CLAUDE_SAMPLE="${OUT_DIR}/claude_sample.jsonl"
QUESTIONS="data/processed/${DATASET}.jsonl"

# --- Step 1: Sample 20% stratified ---
if [ -f "$GPT4O_SAMPLE" ] && [ -f "$RESP_SAMPLE" ]; then
    echo "[Step 1] Skipping — sample files already exist"
else
    echo "[Step 1] Sampling ${FRACTION} of responses (stratified by challenge type)..."
    run_in_container python scripts/sample_for_rejudge.py \
        --evaluated  "$EVALUATED" \
        --responses  "$RESPONSES" \
        --output-evaluated "$GPT4O_SAMPLE" \
        --output-responses "$RESP_SAMPLE" \
        --fraction "$FRACTION" \
        --seed "$SEED"
fi

# --- Step 2: Re-evaluate sample with Claude ---
if [ -f "$CLAUDE_SAMPLE" ] && [ "$(wc -l < "$CLAUDE_SAMPLE")" -ge "$(wc -l < "$RESP_SAMPLE")" ]; then
    echo "[Step 2] Skipping — Claude evaluation already complete"
else
    echo "[Step 2] Re-evaluating with Claude (${ALT_JUDGE_BACKEND})..."
    run_in_container python scripts/evaluate.py \
        --input    "$RESP_SAMPLE" \
        --questions "$QUESTIONS" \
        --output   "$CLAUDE_SAMPLE" \
        --judge-config "$ALT_JUDGE_BACKEND" \
        --no-resume
fi

# --- Step 3: Compare judges and compute κ ---
echo "[Step 3] Computing Cohen's κ and checking ranking stability..."
run_in_container python scripts/compare_judges.py \
    --primary "$GPT4O_SAMPLE" \
    --alt     "$CLAUDE_SAMPLE" \
    --label   "${DATASET}/${MODEL_NAME}" \
    --output  "${OUT_DIR}/judge_agreement.json"

echo "=== Done: $MODEL_NAME on $DATASET ==="
echo "Agreement report: ${OUT_DIR}/judge_agreement.json"
