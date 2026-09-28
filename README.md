# Receptivity and Resistance: Factual Sycophancy Across Post-Training

Code for our paper, accepted at NeurIPS 2026.

We trace factual sycophancy across 12 checkpoints from four post-training pipelines built on two base models: OLMo 3 7B Think and Instruct, Llama 3.1 8B Instruct, and Tulu 3. We report two evaluation dimensions. **Receptivity** is the challenge-induced change in the wrong-minus-correct mean-per-token log-probability score (ΔLogOdds). **Resistance** is retention of an initially correct answer under challenge, measured through coherent-response flip rates, with erroneous responses reported separately. These are measurement labels, not claims about internal beliefs or mechanisms.

On matched items, flip rates fall while preemptive ΔLogOdds grows for OLMo Instruct on both domains, OLMo Think on math, and Tulu 3 on medical questions. Llama Instruct reduces both quantities on matched items; OLMo Think shows no significant medical behavioral improvement. Final-checkpoint sampling reveals additional differences: flip rates remain comparatively stable or decrease for OLMo Instruct and Tulu 3, with some cell-specific increases, but increase for Llama on math and for OLMo Think, especially medical questions. Sampling does not establish that candidate-score shifts cause errors or test matched base-to-final training effects. The paper reports both dimensions and their limitations separately.

## Pipeline

```
Raw data
  │
  ▼
[1. Preprocess]              →  data/processed/{dataset}.jsonl
  │
  ▼
[2. Generate challenges]     →  adds 8 challenges per question (via GPT-4o)
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
[3. Generate responses]                   [3b. Score log-probs]
  │   (generative track)                    │   (log-prob track)
  ▼                                         │
[4. Evaluate]                               │
  │   GPT-4o judge + hedging/refusal        │
  │                                         │
  ├─────────────────────────────────────────┘
  ▼
[5. Analyze]                 →  summaries, statistical tests, plots
```

## Setup

```bash
pip install -e .
```

Set the API key used for challenge generation and the GPT-4o judge.

```bash
export OPENAI_API_KEY=...
```

## Reproduce

The full per-checkpoint pipeline (preprocess, generate challenges, run inference, score log-probabilities, judge evaluation) is orchestrated by `slurm/run_experiment.sh`, written for SLURM plus Apptainer and launched once per model checkpoint.

```bash
sbatch --export=ALL,HF_MODEL=allenai/Olmo-3-7B-Instruct,MODEL_NAME=olmo3-7b-instruct,MODEL_TYPE=chat,CHECKPOINT=instruct,DATASET=medical_advice slurm/run_experiment.sh
```

Analysis, statistics, and figures come from `scripts/analyze.py` and the `plot_*.py` scripts. To run outside SLURM, adapt the in-container `python scripts/...` commands inside `slurm/run_experiment.sh`.
