#!/bin/bash
# Tracing Sycophancy - Submit jobs for all models in an experiment
#
# Usage:
#   bash slurm/run_all_models.sh
#
# Submits one Slurm job per (model, dataset) combination.
# Each job loads a HuggingFace model via transformers and runs the pipeline.
# Edit the arrays below to configure your experiment.

EXPERIMENT="exp1"
DATASETS=("computational" "medical_advice")

# Define models: name|hf_model_id|model_type|checkpoint
MODELS=(
    # OLMo 3 7B — base
    # "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    # OLMo 3 7B — Think pipeline (base → SFT → DPO → Think)
    # "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    # "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    # "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    # OLMo 3 7B — Instruct pipeline (base → SFT → DPO → Instruct)
    # "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    # "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    # "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    # Llama 3.2 3B:
    # "llama-3.2-3b-base|meta-llama/Llama-3.2-3B|base|base"
    # "llama-3.2-3b-instruct|meta-llama/Llama-3.2-3B-Instruct|chat|instruct"
    # Llama 3.2 1B:
    # "llama-3.2-1b-base|meta-llama/Llama-3.2-1B|base|base"
    # "llama-3.2-1b-instruct|meta-llama/Llama-3.2-1B-Instruct|chat|instruct"
    # Ministral 3B:
    # "ministral-3-3b-base|mistralai/Ministral-3-3B-Base-2512|base|base"
    # "ministral-3-3b-instruct|mistralai/Ministral-3-3B-Instruct-2512|chat|instruct"
    # "ministral-3-3b-reasoning|mistralai/Ministral-3-3B-Reasoning-2512|chat|reasoning"
    # Ministral 8B:
    # "ministral-3-8b-base|mistralai/Ministral-3-8B-Base-2512|base|base"
    # "ministral-3-8b-instruct|mistralai/Ministral-3-8B-Instruct-2512|chat|instruct"
    # "ministral-3-8b-reasoning|mistralai/Ministral-3-8B-Reasoning-2512|chat|reasoning"
    # Ministral 14B:
    # "ministral-3-14b-base|mistralai/Ministral-3-14B-Base-2512|base|base"
    # "ministral-3-14b-instruct|mistralai/Ministral-3-14B-Instruct-2512|chat|instruct"
    # "ministral-3-14b-reasoning|mistralai/Ministral-3-14B-Reasoning-2512|chat|reasoning"
    # LLM360 Amber 7B (base → SFT → Safety DPO):
    "amber-7b-base|LLM360/Amber|base|base"
    "amber-7b-sft|LLM360/AmberChat|chat|sft"
    "amber-7b-dpo|LLM360/AmberSafe|chat|dpo"
    # Zephyr 7B (Mistral base → SFT → DPO):
    # "zephyr-7b-base|mistralai/Mistral-7B-v0.1|base|base"
    # "zephyr-7b-sft|alignment-handbook/zephyr-7b-sft-full|chat|sft"
    # "zephyr-7b-dpo|HuggingFaceH4/zephyr-7b-beta|chat|dpo"
)

# OLMo training checkpoints — reads revisions from config file.
# Set to true to submit one job per checkpoint; false to skip.
RUN_OLMO_CHECKPOINTS=false
OLMO_CHECKPOINT_FILE="config/olmo_checkpoints.txt"

mkdir -p logs

JOB_IDS=()

# --- Standard models ---
for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"

        echo "Submitting: $MODEL_NAME on $DATASET"
        JOB_ID=$(sbatch --parsable \
            --job-name="syco_${MODEL_NAME}_${DATASET}" \
            --export="ALL,DATASET=$DATASET,HF_MODEL=$HF_MODEL,MODEL_NAME=$MODEL_NAME,MODEL_TYPE=$MODEL_TYPE,CHECKPOINT=$CHECKPOINT,EXPERIMENT=$EXPERIMENT" \
            slurm/run_experiment.sh)
        JOB_IDS+=("$JOB_ID")
    done
done

# --- OLMo training checkpoints ---
if [ "$RUN_OLMO_CHECKPOINTS" = true ] && [ -f "$OLMO_CHECKPOINT_FILE" ]; then
    while IFS= read -r REVISION; do
        # Skip comments and blank lines
        [[ "$REVISION" =~ ^#.*$ || -z "$REVISION" ]] && continue

        for DATASET in "${DATASETS[@]}"; do
            MODEL_NAME="olmo-7b-${REVISION}"
            echo "Submitting: $MODEL_NAME on $DATASET"
            JOB_ID=$(sbatch --parsable \
                --job-name="syco_${MODEL_NAME}_${DATASET}" \
                --export="ALL,DATASET=$DATASET,HF_MODEL=allenai/OLMo-7B,MODEL_NAME=$MODEL_NAME,MODEL_TYPE=base,CHECKPOINT=$REVISION,REVISION=$REVISION,EXPERIMENT=$EXPERIMENT" \
                slurm/run_experiment.sh)
            JOB_IDS+=("$JOB_ID")
        done
    done < "$OLMO_CHECKPOINT_FILE"
fi

# --- Analysis job (runs after all model jobs finish) ---
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
    echo "Submitting analysis job (depends on ${#JOB_IDS[@]} jobs)..."
    sbatch \
        --job-name="syco_analysis_${EXPERIMENT}" \
        --dependency="afterany:${DEP_STR}" \
        --partition=normal \
        --time=01:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --output=logs/analysis_%j.out \
        --error=logs/analysis_%j.err \
        --wrap="
            ml python/3.12.1
            source ${VENV_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy/sycophancy_env}/bin/activate
            cd ${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}
            export PYTHONPATH=\$PWD:\$PYTHONPATH
            python scripts/analyze.py \
                --results-dir data/results/${EXPERIMENT} \
                --output-dir data/results/${EXPERIMENT}/analysis
        "
fi

echo "All jobs submitted. Monitor with: squeue -u \$USER"
