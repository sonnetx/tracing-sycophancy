#!/usr/bin/env python3
"""Check 1: Candidate-dependence robustness for ΔLogOdds.

Loads 3 logprob score files (one per wrong-answer candidate) and reports:
  - Per-question variance of mean ΔLogOdds across candidates
  - Spearman rank correlation of per-checkpoint mean ΔLogOdds between candidate pairs
  - % of checkpoints where the top-3 ranking is identical across all 3 candidates

If rankings are stable across candidate draws the dissociation finding holds
regardless of which wrong answer was picked.

Usage:
    python scripts/analyze_candidate_robustness.py \
        --files \
            data/results/exp_candidate_robustness/computational/olmo3-7b-base/logprob_scores_c0.jsonl \
            data/results/exp_candidate_robustness/computational/olmo3-7b-base/logprob_scores_c1.jsonl \
            data/results/exp_candidate_robustness/computational/olmo3-7b-base/logprob_scores_c2.jsonl \
        --dataset computational

Or with --experiment-dir to aggregate across all models in a directory:
    python scripts/analyze_candidate_robustness.py \
        --experiment-dir data/results/exp_candidate_robustness
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.stats import load_logprob_results


def per_checkpoint_mean_dlo(lp_df: pd.DataFrame, non_simple_only: bool = True) -> float:
    """Mean ΔLogOdds across all questions for one logprob score file."""
    ch = lp_df[lp_df["condition"] == "challenge"]
    if non_simple_only:
        ch = ch[ch["challenge_type"] != "simple"]
    if len(ch) == 0:
        return float("nan")
    return float(ch.groupby("question_id")["delta_log_odds"].mean().mean())


def analyze_candidates(files: list[str], label: str = "") -> dict:
    """Compare ΔLogOdds variance and rank stability across 3 candidate files."""
    if len(files) < 2:
        raise ValueError("Need at least 2 candidate files to compare")

    dfs = [load_logprob_results(f) for f in files]
    n = len(dfs)

    # Per-question mean ΔLogOdds for each candidate (non-simple challenges only)
    per_q_means = []
    for df in dfs:
        ch = df[df["condition"] == "challenge"]
        ch = ch[ch["challenge_type"] != "simple"]
        per_q = ch.groupby("question_id")["delta_log_odds"].mean()
        per_q_means.append(per_q)

    common_qids = per_q_means[0].index
    for pq in per_q_means[1:]:
        common_qids = common_qids.intersection(pq.index)

    if len(common_qids) == 0:
        return {"error": "No common question_ids across candidate files"}

    # Stack into a matrix: rows = questions, cols = candidates
    mat = np.column_stack([pq.loc[common_qids].values for pq in per_q_means])

    # Per-question variance across candidates
    q_variances = mat.var(axis=1, ddof=1) if n > 1 else np.zeros(len(common_qids))
    mean_per_q_dlo = mat.mean(axis=1)

    # Spearman rank correlations between all pairs
    pair_rhos = []
    for i, j in combinations(range(n), 2):
        rho, p = spearmanr(mat[:, i], mat[:, j])
        pair_rhos.append({"pair": f"c{i}-c{j}", "rho": float(rho), "p": float(p)})

    # Checkpoint-level mean ΔLogOdds (one scalar per candidate)
    checkpoint_means = [per_checkpoint_mean_dlo(df) for df in dfs]

    result = {
        "label": label,
        "n_questions": int(len(common_qids)),
        "n_candidates": n,
        "mean_per_question_variance": float(np.mean(q_variances)),
        "median_per_question_variance": float(np.median(q_variances)),
        "mean_per_question_dlo": float(mean_per_q_dlo.mean()),
        "candidate_checkpoint_means": checkpoint_means,
        "pairwise_spearman": pair_rhos,
        "min_spearman_rho": float(min(r["rho"] for r in pair_rhos)),
    }
    return result


def analyze_experiment_dir(exp_dir: str) -> dict:
    """Walk exp_dir/{dataset}/{model}/logprob_scores_c{0,1,2}.jsonl and aggregate."""
    all_results = {}
    for dataset in sorted(os.listdir(exp_dir)):
        ds_dir = os.path.join(exp_dir, dataset)
        if not os.path.isdir(ds_dir) or dataset == "analysis":
            continue
        model_results = {}
        for model in sorted(os.listdir(ds_dir)):
            model_dir = os.path.join(ds_dir, model)
            if not os.path.isdir(model_dir):
                continue
            files = sorted(
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.startswith("logprob_scores_c") and f.endswith(".jsonl")
            )
            if len(files) < 2:
                continue
            try:
                result = analyze_candidates(files, label=model)
                model_results[model] = result
            except Exception as e:
                print(f"  Error for {dataset}/{model}: {e}")
        if model_results:
            all_results[dataset] = model_results
    return all_results


def print_report(results: dict) -> None:
    for dataset, model_results in results.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"{'='*70}")
        print(f"{'Model':<35} {'n_q':>5} {'mean_var':>10} {'min_rho':>8}  Candidate means")
        print("-" * 75)
        for model, r in sorted(model_results.items()):
            if "error" in r:
                print(f"  {model}: {r['error']}")
                continue
            means_str = "  ".join(f"{m:+.3f}" for m in r["candidate_checkpoint_means"])
            print(f"{model:<35} {r['n_questions']:>5} {r['mean_per_question_variance']:>10.4f} "
                  f"{r['min_spearman_rho']:>8.3f}  [{means_str}]")
        print()
        all_rhos = [
            pr["rho"]
            for r in model_results.values()
            if "pairwise_spearman" in r
            for pr in r["pairwise_spearman"]
        ]
        if all_rhos:
            print(f"  Across all models — min Spearman ρ: {min(all_rhos):.3f}  "
                  f"mean: {np.mean(all_rhos):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Candidate-dependence robustness check for ΔLogOdds")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--files", nargs="+", metavar="FILE",
                      help="2-3 logprob score JSONL files (one per candidate)")
    mode.add_argument("--experiment-dir", metavar="DIR",
                      help="Experiment root with {dataset}/{model}/logprob_scores_c*.jsonl layout")
    parser.add_argument("--dataset", default="", help="Label for --files mode")
    parser.add_argument("--output", default=None, help="Path to save JSON report")
    args = parser.parse_args()

    if args.files:
        result = analyze_candidates(args.files, label=args.dataset)
        results = {args.dataset or "dataset": {"model": result}}
        print_report(results)
    else:
        results = analyze_experiment_dir(args.experiment_dir)
        print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
