#!/bin/bash -l
#SBATCH --job-name=syco_samp_ctrl
#SBATCH --partition=roxanad,gpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Controlled sampling run (scripts/sampling_controlled.py). Uses the main greedy run's
# initially correct items and its 4,096-token cap, samples the preemptive wrong-answer
# challenges plus the bare-question baseline at T=1, and records finish reasons.
#
# Usage:
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,DATASET=computational slurm/run_sampling_controlled.sh

set -euo pipefail
PROJECT_DIR=/home/groups/roxanad/sonnet/tracing-sycophancy
SIF_IMAGE=/scratch/users/$USER/simg/vllm-v0.11.0.sif
: "${HF_MODEL:?}" "${MODEL_NAME:?}" "${DATASET:?}"
TEMPERATURE="${TEMPERATURE:-1.0}"
N_SAMPLES="${N_SAMPLES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
EVAL_EXP="${EVAL_EXP:-exp1_rerun}"
OUT_EXP="${OUT_EXP:-exp1_sampling_controlled}"

export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
[ -f ~/.secrets ] && { set -a; source ~/.secrets; set +a; }
cd "$PROJECT_DIR"
mkdir -p logs

OUT_DIR="data/results/$OUT_EXP/$DATASET/$MODEL_NAME"
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/model_config.json" <<CONF
{"backend": "vllm", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": 8192, "gpu_memory_utilization": 0.80}
CONF

echo "=== Controlled sampling: $MODEL_NAME | $DATASET | T=$TEMPERATURE | N=$N_SAMPLES | max_new=$MAX_NEW_TOKENS | items from $EVAL_EXP ==="
apptainer exec --nv --containall \
    -B "$PROJECT_DIR:/workspace" -B "/scratch/users/$USER:/scratch_user" -B "/scratch/users/$USER/tmp:/tmp" \
    --home /scratch_user \
    --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/workspace \
    --env HF_HOME=/scratch_user/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_ETAG_TIMEOUT=10 --env HF_HUB_DOWNLOAD_TIMEOUT=60 \
    --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
    --pwd /workspace "$SIF_IMAGE" \
    bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && \
      python scripts/sampling_controlled.py \
        --input data/processed/$DATASET.jsonl \
        --existing-eval data/results/$EVAL_EXP/$DATASET/$MODEL_NAME/evaluated.jsonl \
        --output-dir $OUT_DIR --backend-config $OUT_DIR/model_config.json \
        --judge-config config/models/gpt4o_judge.json --model-name $MODEL_NAME \
        --temperature $TEMPERATURE --n-samples $N_SAMPLES --max-new-tokens $MAX_NEW_TOKENS"
echo "=== Done: $OUT_DIR ==="
