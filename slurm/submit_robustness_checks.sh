#!/bin/bash
# Submit all 5 robustness checks with proper SLURM dependencies.
#
# Dependency graph:
#
#   [immediate] gen_ablation_challenges  ─────┐
#   [immediate] check1 jobs (GPU)             │
#   [immediate] check2 jobs (CPU/API)         │
#   [immediate] check3 jobs (GPU)             │
#                                             ▼
#                                   check5 inference (GPU)
#                                             │
#   ◄────────────────────────────────────────┘
#   all_done ─► robustness_analysis (CPU)
#
# Usage:
#   bash slurm/submit_robustness_checks.sh             # submit everything
#   bash slurm/submit_robustness_checks.sh --dry-run   # print commands only

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=true; done

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/tracing-sycophancy}"
cd "$PROJECT_DIR"
mkdir -p logs

if [ -f ~/.secrets ]; then set -a; source ~/.secrets; set +a; fi

# submit <sbatch-args...> — returns job ID; prints "[DRY-RUN]" in dry-run mode
submit() {
    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "99999"
    else
        sbatch --parsable "$@"
    fi
}

# dependency <jid1> [jid2 ...] — formats SLURM afterok dependency string
dependency() { local ids; ids=$(IFS=:; echo "$*"); echo "afterok:${ids}"; }

# ── Prerequisites ──────────────────────────────────────────────────────────────

if [ ! -f config/models/claude_judge.json ]; then
    echo "[Setup] Creating config/models/claude_judge.json..."
    cat > config/models/claude_judge.json <<'EOF'
{
    "backend": "anthropic",
    "model": "claude-sonnet-4-6",
    "temperature": 0
}
EOF
fi

# ── Model lists ────────────────────────────────────────────────────────────────

# All models — used for Checks 2 (all) and 3 (all × medical)
ALL_MODELS=(
    "olmo3-7b-base|allenai/Olmo-3-1025-7B|base|base"
    "olmo3-7b-think-sft|allenai/Olmo-3-7B-Think-SFT|chat|sft"
    "olmo3-7b-think-dpo|allenai/Olmo-3-7B-Think-DPO|chat|dpo"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    "olmo3-7b-instruct-sft|allenai/Olmo-3-7B-Instruct-SFT|chat|sft"
    "olmo3-7b-instruct-dpo|allenai/Olmo-3-7B-Instruct-DPO|chat|dpo"
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    "llama31-8b-base|meta-llama/Llama-3.1-8B|base|base"
    "llama31-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|chat|instruct"
    "tulu3-llama31-8b-sft|allenai/Llama-3.1-Tulu-3-8B-SFT|chat|sft"
    "tulu3-llama31-8b-dpo|allenai/Llama-3.1-Tulu-3-8B-DPO|chat|dpo"
    "tulu3-llama31-8b|allenai/Llama-3.1-Tulu-3-8B|chat|instruct"
)

# Representative final-stage models — used for Checks 1 and 5
REP_MODELS=(
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|chat|instruct"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|chat|think"
    "llama31-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|chat|instruct"
    "tulu3-llama31-8b|allenai/Llama-3.1-Tulu-3-8B|chat|instruct"
)

DATASETS=("computational" "medical_advice")

# Accumulates all job IDs that the final analysis must wait for
ALL_JIDS=()

echo "========================================================"
echo " Robustness checks submission"
echo "========================================================"

# ── Check 5, Step 1: Generate ablation challenges (CPU, no GPU) ───────────────
echo ""
echo "── Check 5 / Step 1: gen_ablation_challenges ─────────────"
GEN_ABL_JID=$(submit \
    --job-name=gen_ablation \
    --partition=normal \
    --time=01:00:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output=logs/gen_ablation_%j.out \
    --error=logs/gen_ablation_%j.err \
    slurm/gen_ablation_challenges.sh)
echo "  gen_ablation_challenges → job $GEN_ABL_JID"

# ── Check 1: Candidate dependence (rep models × 2 datasets, GPU) ─────────────
echo ""
echo "── Check 1: Candidate dependence ─────────────────────────"
for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${REP_MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"
        JID=$(submit \
            --job-name="c1_${MODEL_NAME}_${DATASET:0:4}" \
            --partition=roxanad \
            --time=08:00:00 \
            --mem=64G \
            --cpus-per-task=8 \
            --gpus=1 \
            -C GPU_MEM:80GB \
            --output="logs/c1_${MODEL_NAME}_${DATASET}_%j.out" \
            --error="logs/c1_${MODEL_NAME}_${DATASET}_%j.err" \
            --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT",DATASET="$DATASET" \
            slurm/run_candidate_robustness.sh)
        echo "  Check 1 | $MODEL_NAME / $DATASET → job $JID"
        ALL_JIDS+=("$JID")
    done
done

# ── Check 2: Alternative judge (all models × 2 datasets, CPU/API) ────────────
echo ""
echo "── Check 2: Alternative judge ─────────────────────────────"
for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${ALL_MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"
        JID=$(submit \
            --job-name="c2_${MODEL_NAME}_${DATASET:0:4}" \
            --partition=normal \
            --time=02:00:00 \
            --mem=16G \
            --cpus-per-task=4 \
            --output="logs/c2_${MODEL_NAME}_${DATASET}_%j.out" \
            --error="logs/c2_${MODEL_NAME}_${DATASET}_%j.err" \
            --export=ALL,MODEL_NAME="$MODEL_NAME",DATASET="$DATASET" \
            slurm/run_rejudge.sh)
        echo "  Check 2 | $MODEL_NAME / $DATASET → job $JID"
        ALL_JIDS+=("$JID")
    done
done

# ── Check 3: Length-matched wrong answers (all models × medical_advice, GPU) ──
echo ""
echo "── Check 3: Length-matched wrong answers (medical_advice) ─"
for MODEL_ENTRY in "${ALL_MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"
    JID=$(submit \
        --job-name="c3_${MODEL_NAME}" \
        --partition=roxanad \
        --time=08:00:00 \
        --mem=64G \
        --cpus-per-task=8 \
        --gpus=1 \
        -C GPU_MEM:80GB \
        --output="logs/c3_${MODEL_NAME}_%j.out" \
        --error="logs/c3_${MODEL_NAME}_%j.err" \
        --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT" \
        slurm/run_length_robustness.sh)
    echo "  Check 3 | $MODEL_NAME / medical_advice → job $JID"
    ALL_JIDS+=("$JID")
done

# ── Check 5, Step 2: Citation ablation inference (after gen_ablation, GPU) ────
echo ""
echo "── Check 5 / Step 2: Citation ablation inference ──────────"
for DATASET in "${DATASETS[@]}"; do
    for MODEL_ENTRY in "${REP_MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME HF_MODEL MODEL_TYPE CHECKPOINT <<< "$MODEL_ENTRY"
        JID=$(submit \
            --job-name="c5_${MODEL_NAME}_${DATASET:0:4}" \
            --partition=roxanad \
            --time=12:00:00 \
            --mem=64G \
            --cpus-per-task=8 \
            --gpus=1 \
            -C GPU_MEM:80GB \
            --output="logs/c5_${MODEL_NAME}_${DATASET}_%j.out" \
            --error="logs/c5_${MODEL_NAME}_${DATASET}_%j.err" \
            --dependency="$(dependency "$GEN_ABL_JID")" \
            --export=ALL,HF_MODEL="$HF_MODEL",MODEL_NAME="$MODEL_NAME",MODEL_TYPE="$MODEL_TYPE",CHECKPOINT="$CHECKPOINT",DATASET="$DATASET" \
            slurm/run_citation_ablation.sh)
        echo "  Check 5 | $MODEL_NAME / $DATASET → job $JID (after gen_ablation $GEN_ABL_JID)"
        ALL_JIDS+=("$JID")
    done
done

# ── Final analysis (after ALL above jobs complete) ────────────────────────────
echo ""
echo "── Final analysis (all checks) ────────────────────────────"
DEP=$(dependency "${ALL_JIDS[@]}")
ANALYSIS_JID=$(submit \
    --job-name=robustness_analysis \
    --partition=normal \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output=logs/robustness_analysis_%j.out \
    --error=logs/robustness_analysis_%j.err \
    --dependency="$DEP" \
    slurm/run_robustness_analysis.sh)
echo "  Analysis → job $ANALYSIS_JID (after ${#ALL_JIDS[@]} inference jobs)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Submitted $((${#ALL_JIDS[@]} + 2)) jobs total"
echo "   gen_ablation   : $GEN_ABL_JID"
echo "   Check 1 (GPU)  : ${#REP_MODELS[@]} models × 2 datasets = $((${#REP_MODELS[@]} * 2)) jobs"
echo "   Check 2 (CPU)  : ${#ALL_MODELS[@]} models × 2 datasets = $((${#ALL_MODELS[@]} * 2)) jobs"
echo "   Check 3 (GPU)  : ${#ALL_MODELS[@]} models × medical   = ${#ALL_MODELS[@]} jobs"
echo "   Check 5 (GPU)  : ${#REP_MODELS[@]} models × 2 datasets = $((${#REP_MODELS[@]} * 2)) jobs (blocked on gen_ablation)"
echo "   Analysis (CPU) : $ANALYSIS_JID"
echo ""
echo " Monitor: squeue -u \$USER"
echo " Logs:    tail -f logs/robustness_analysis_*.out"
echo "========================================================"
