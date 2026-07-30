#!/bin/bash
#SBATCH --job-name=syco_paraphrase_robustness
#SBATCH --partition=roxanad,gpu
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Paraphrase-dependence robustness for ΔLogOdds (reviewer request).
#
# GPT-4o paraphrases the correct and wrong answers for a subset of medical
# items (content preserved, surface form changed). Each variant swaps ONE
# candidate's surface form while challenges stay fixed, so the check isolates
# how much the reported ΔLogOdds depends on candidate phrasing:
#   c0_w0 (originals), c1_w0/c2_w0 (correct paraphrased), c0_w1/c0_w2 (wrong paraphrased)
#
# Paraphrase generation (~800 GPT-4o calls, ~$4-5) runs once and is skipped
# when the file exists. Submit ONE job first (it generates the paraphrases),
# then the remaining models after Step 1 completes, e.g.:
#
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,CHECKPOINT=instruct \
#     slurm/run_paraphrase_robustness.sh
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Think,MODEL_NAME=olmo3-7b-think,CHECKPOINT=main,MAX_MODEL_LEN=12288 \
#     slurm/run_paraphrase_robustness.sh
#   sbatch --export=ALL,HF_MODEL=meta-llama/Llama-3.1-8B-Instruct,MODEL_NAME=llama31-8b-instruct,CHECKPOINT=instruct \
#     slurm/run_paraphrase_robustness.sh
#   sbatch --export=ALL,HF_MODEL=allenai/Llama-3.1-Tulu-3-8B,MODEL_NAME=tulu3-llama31-8b,CHECKPOINT=instruct \
#     slurm/run_paraphrase_robustness.sh
#
# After all four finish:
#   python scripts/analyze_paraphrase_robustness.py \
#     --experiment-dir data/results/exp_paraphrase_robustness/medical_advice

set -euo pipefail

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_IMAGE="/scratch/users/$USER/simg/vllm-v0.11.0.sif"

# --- Model ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-instruct}"
CHECKPOINT="${CHECKPOINT:-instruct}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

# --- Experiment ---
DATASET="${DATASET:-medical_advice}"
EXPERIMENT="${EXPERIMENT:-exp_paraphrase_robustness}"
N_ITEMS="${N_ITEMS:-200}"
N_PARAPHRASES="${N_PARAPHRASES:-2}"
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-config/models/gpt4o_judge.json}"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" logs

if [ -f ~/.secrets ]; then
    set -a; source ~/.secrets; set +a
fi

cd "$PROJECT_DIR"

TOOL=$(command -v apptainer || command -v singularity)

run_in_container() {
    "$TOOL" exec --nv \
        --containall \
        -B "$PROJECT_DIR:/workspace" \
        -B "/scratch/users/$USER:/scratch_user" \
        -B "/scratch/users/$USER/tmp:/tmp" \
        --home /scratch_user \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=/workspace" \
        --env "HF_HOME=/scratch_user/huggingface" \
        --env "HF_DATASETS_CACHE=/scratch_user/huggingface/datasets" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== Paraphrase robustness: $DATASET | $MODEL_NAME (${N_ITEMS} items, ${N_PARAPHRASES} paraphrases per candidate) ==="

PROCESSED_MAIN="data/processed/${DATASET}.jsonl"
PARAPHRASE_FILE="data/processed/${DATASET}_paraphrase_subset.jsonl"
VARIANT_DIR="data/processed/${DATASET}_paraphrase_variants"
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"

# --- Step 1: Generate paraphrases (GPT-4o; runs once, resumes if partial) ---
if [ -f "$PARAPHRASE_FILE" ] && [ "$(wc -l < "$PARAPHRASE_FILE")" -ge "$N_ITEMS" ]; then
    echo "[Step 1] Skipping — $PARAPHRASE_FILE already complete"
else
    echo "[Step 1] Generating paraphrases (GPT-4o)..."
    run_in_container python scripts/generate_paraphrases.py \
        --input "$PROCESSED_MAIN" \
        --output "$PARAPHRASE_FILE" \
        --backend-config "$CHALLENGE_BACKEND" \
        --n-items "$N_ITEMS" \
        --n-paraphrases "$N_PARAPHRASES" \
        --seed 42
fi

# --- Step 2: Materialize scoring variants ---
FIRST_VARIANT="$VARIANT_DIR/variant_c0_w0.jsonl"
if [ -f "$FIRST_VARIANT" ] && [ "$(wc -l < "$FIRST_VARIANT")" -ge "$N_ITEMS" ]; then
    echo "[Step 2] Skipping — variants already materialized"
else
    echo "[Step 2] Materializing variants..."
    run_in_container python scripts/make_paraphrase_variants.py \
        --input "$PARAPHRASE_FILE" \
        --output-dir "$VARIANT_DIR"
fi

# --- Step 3: Model backend config ---
MODEL_CONFIG="$RESULT_DIR/model_config.json"
cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF

# --- Step 4: Score each variant ---
VARIANTS="c0_w0"
for i in $(seq 1 "$N_PARAPHRASES"); do VARIANTS="$VARIANTS c${i}_w0"; done
for i in $(seq 1 "$N_PARAPHRASES"); do VARIANTS="$VARIANTS c0_w${i}"; done

for V in $VARIANTS; do
    SCORES_FILE="$RESULT_DIR/logprob_scores_${V}.jsonl"
    VARIANT_FILE="$VARIANT_DIR/variant_${V}.jsonl"
    if [ -f "$SCORES_FILE" ] && [ "$(wc -l < "$SCORES_FILE")" -ge "$(wc -l < "$VARIANT_FILE")" ]; then
        echo "[Step 4] Skipping variant $V (already scored)"
        continue
    fi
    echo "[Step 4] Scoring variant $V..."
    run_in_container python scripts/score_logprobs.py \
        --input "$VARIANT_FILE" \
        --output "$SCORES_FILE" \
        --backend-config "$MODEL_CONFIG" \
        --model-name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --resume
done

# --- Step 5: Analyze (aggregates whichever models are done; idempotent) ---
echo "[Step 5] Analyzing paraphrase robustness..."
run_in_container python scripts/analyze_paraphrase_robustness.py \
    --experiment-dir "data/results/${EXPERIMENT}/${DATASET}"

echo "=== Done: $MODEL_NAME on $DATASET ==="
