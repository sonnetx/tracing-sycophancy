#!/bin/bash
#SBATCH --job-name=syco_ifctrl
#SBATCH --partition=roxanad
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Instruction-following controls: belief / command / truth_orthogonal challenges that
# hold the wrong answer fixed while varying only the framing, to separate
# sycophancy-proper from compliance. run_inference.py produces BOTH the generative
# responses and the log-prob (ΔLogOdds) scores; evaluate.py then judges flips.
# Results go under a separate experiment dir; the existing exp1 results are untouched,
# and the ethos/justification/citation comparison comes from exp1 for the same model.
#
# Reuses the cached proposed_answer from data/processed/<dataset>.jsonl, so challenge
# generation makes NO new GPT-4o calls. Only the 6 control challenges are kept (the main
# types already exist in exp1), so inference is ~initial + 6 challenges per item.
#
# Usage (run once per checkpoint x dataset). Use the SAME model identifiers as your
# original exp1 / sampling runs so results are comparable:
#   sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Think,MODEL_NAME=olmo3-7b-think,MODEL_TYPE=chat,CHECKPOINT=think,DATASET=computational slurm/run_if_controls.sh

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_STORE}/vllm-v0.11.0.sif"

# --- Model (override via --export; defaults are the OLMo Think final checkpoint) ---
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b-think}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
CHECKPOINT="${CHECKPOINT:-think}"
REVISION="${REVISION:-}"
BACKEND_TYPE="${BACKEND_TYPE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# --- Dataset / experiment ---
DATASET="${DATASET:-computational}"          # computational | medical_advice
EXPERIMENT="${EXPERIMENT:-exp_if_controls}"
TRUTH_ORTHO="${TRUTH_ORTHO:-Answer in all capital letters.}"

# --- Judge & challenge generation ---
JUDGE_BACKEND="${JUDGE_BACKEND:-config/models/gpt4o_judge.json}"
CHALLENGE_BACKEND="${CHALLENGE_BACKEND:-$JUDGE_BACKEND}"

# --- Environment ---
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" logs

if [ -f ~/.secrets ]; then set -a; source ~/.secrets; set +a; fi

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
        --pwd /workspace \
        "$SIF_IMAGE" \
        bash -c "source /scratch_user/container_env/bin/activate && export PATH=\$VIRTUAL_ENV/bin:\$PATH && export PYTHONPATH=/workspace && $*"
}

echo "=== IF-controls: $DATASET | $MODEL_NAME | $CHECKPOINT ==="

PROCESSED_MAIN="data/processed/${DATASET}.jsonl"
PROCESSED_IFCTRL="data/processed/${DATASET}_ifcontrol.jsonl"

[ -f "$PROCESSED_MAIN" ] || { echo "[FATAL] $PROCESSED_MAIN not found (run the main pipeline first)"; exit 1; }

# --- Step 1: Generate IF-control challenges, then keep only the control types ---
if [ -f "$PROCESSED_IFCTRL" ] && \
   python3 -c "import sys,json; d=json.load(open('$PROCESSED_IFCTRL')); sys.exit(0 if any(c['type']=='command' for c in d.get('challenges',[])) else 1)" 2>/dev/null; then
    echo "[Step 1] Skipping challenge generation ($PROCESSED_IFCTRL already has IF-control challenges)"
else
    case "$DATASET" in
        computational)  ETHOS="an expert in math" ;;
        medical_advice) ETHOS="an expert in medicine" ;;
        *)              ETHOS="an expert" ;;
    esac
    echo "[Step 1] Generating IF-control challenges (belief / command / truth_orthogonal)..."
    run_in_container python scripts/generate_challenges.py \
        --input "$PROCESSED_MAIN" \
        --output "$PROCESSED_IFCTRL" \
        --backend-config "$CHALLENGE_BACKEND" \
        --challenge-type factual \
        --ethos "'$ETHOS'" \
        --if-controls \
        --truth-orthogonal-instruction "'$TRUTH_ORTHO'"
    # Drop the main challenge types (they already exist in exp1) to save inference cost.
    run_in_container python3 -c "
import json
keep={'belief','command','truth_orthogonal'}
rows=[json.loads(l) for l in open('$PROCESSED_IFCTRL')]
for r in rows:
    r['challenges']=[c for c in r.get('challenges',[]) if c['type'] in keep]
open('$PROCESSED_IFCTRL','w').write(''.join(json.dumps(r)+chr(10) for r in rows))
print('Kept', sum(len(r['challenges']) for r in rows), 'control challenges across', len(rows), 'items')
"
fi

# --- Step 2: Model config ---
RESULT_DIR="data/results/${EXPERIMENT}/${DATASET}/${MODEL_NAME}"
mkdir -p "$RESULT_DIR"
MODEL_CONFIG="$RESULT_DIR/model_config.json"
if [ -n "$REVISION" ]; then
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "revision": "$REVISION", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
else
    cat > "$MODEL_CONFIG" <<CONF
{"backend": "$BACKEND_TYPE", "model": "$HF_MODEL", "torch_dtype": "bfloat16", "max_model_len": $MAX_MODEL_LEN, "gpu_memory_utilization": $GPU_MEM_UTIL}
CONF
fi

# --- Step 3: Inference (generative + log-prob) ---
echo "[Step 2] Running inference (generative + log-prob)..."
run_in_container python scripts/run_inference.py \
    --input "$PROCESSED_IFCTRL" \
    --output-dir "$RESULT_DIR" \
    --backend-config "$MODEL_CONFIG" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --resume

# --- Step 4: Evaluate (generative flip judgments) ---
RESPONSES="$RESULT_DIR/responses.jsonl"
EVALUATED="$RESULT_DIR/evaluated.jsonl"
if [ ! -f "$RESPONSES" ]; then
    echo "[Step 3] Skipping evaluation (no responses)"
elif [ -f "$EVALUATED" ] && [ "$(wc -l < "$RESPONSES")" -eq "$(wc -l < "$EVALUATED")" ]; then
    echo "[Step 3] Skipping evaluation (already complete)"
else
    echo "[Step 3] Evaluating responses..."
    run_in_container python scripts/evaluate.py \
        --input "$RESPONSES" \
        --questions "$PROCESSED_IFCTRL" \
        --output "$EVALUATED" \
        --judge-config "$JUDGE_BACKEND"
fi

echo "=== Done: $MODEL_NAME | $DATASET (IF-controls) ==="
echo "Results in: $RESULT_DIR  (responses.jsonl, logprob_scores.jsonl, evaluated.jsonl)"
