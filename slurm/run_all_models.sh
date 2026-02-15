#!/bin/bash
# Tracing Sycophancy - Submit jobs for all models in an experiment
#
# Usage:
#   bash slurm/run_all_models.sh
#
# Submits one Slurm job per (model, dataset) combination.
# Edit the arrays below to configure your experiment.

EXPERIMENT="exp1"
DATASETS=("computational" "medical_advice")

# Define model configs: name|config_path|model_type|checkpoint
MODELS=(
    "llama-3.1-8b-base|config/models/llama_base.json|base|base"
    "llama-3.1-8b-instruct|config/models/llama_instruct.json|chat|instruct"
    # Add more models here:
    # "olmo-7b-step1000|config/models/olmo_step1000.json|base|step1000"
    # "olmo-7b-final|config/models/olmo_final.json|chat|final"
    # "arcee-trinity-truebase|config/models/arcee_truebase.json|base|truebase"
    # "arcee-trinity-base|config/models/arcee_base.json|base|base"
)

mkdir -p logs

for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME MODEL_CONFIG MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"

        echo "Submitting: $MODEL_NAME on $DATASET"
        sbatch \
            --job-name="syco_${MODEL_NAME}_${DATASET}" \
            --export="ALL,DATASET=$DATASET,MODEL_CONFIG=$MODEL_CONFIG,MODEL_NAME=$MODEL_NAME,MODEL_TYPE=$MODEL_TYPE,CHECKPOINT=$CHECKPOINT,EXPERIMENT=$EXPERIMENT" \
            slurm/run_experiment.sh
    done
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
