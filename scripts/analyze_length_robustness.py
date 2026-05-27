#!/usr/bin/env python3
"""Check 3: Length-matched wrong-answer robustness for ΔLogOdds.

Compares ΔLogOdds results between the original (unmatched) and length-matched
(wrong answers trimmed to ≤ correct-answer word count) conditions.

Reports:
  - Pre/post token-count ratio (wrong/correct) to confirm matching worked
  - Spearman rank correlation of per-checkpoint mean ΔLogOdds between conditions
  - Whether the direction and relative ordering of model rankings is preserved

Usage:
    python scripts/analyze_length_robustness.py \
        --experiment-dir data/results/exp_length_robustness \
        --baseline-dir   data/results/exp1 \
        --dataset        medical_advice \
        [--output data/results/exp_length_robustness/analysis/length_robustness.json]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.stats import load_logprob_results
from src.utils import read_jsonl


def compute_word_count_ratio(processed_path: str) -> dict:
    """Compute mean wrong/correct word count ratio from a processed JSONL."""
    items = read_jsonl(processed_path)
    ratios_orig = []
    ratios_matched = []
    for item in items:
        correct = item.get("correct_answer", "")
        wrong_orig = item.get("proposed_answer", "")
        wrong_matched = item.get("proposed_answer_length_matched", wrong_orig)
        n_correct = len(correct.split())
        if n_correct == 0:
            continue
        ratios_orig.append(len(wrong_orig.split()) / n_correct)
        ratios_matched.append(len(wrong_matched.split()) / n_correct)
    return {
        "n_items": len(ratios_orig),
        "mean_ratio_original": float(np.mean(ratios_orig)) if ratios_orig else float("nan"),
        "mean_ratio_matched": float(np.mean(ratios_matched)) if ratios_matched else float("nan"),
        "pct_trimmed": float(np.mean([r > 1 for r in ratios_orig])) if ratios_orig else 0.0,
    }


def per_model_mean_dlo(results_dir: str, non_simple_only: bool = True) -> dict[str, float]:
    """Return {model_name: mean_ΔLogOdds} for all models in a results dir."""
    out = {}
    for model in sorted(os.listdir(results_dir)):
        lp_path = os.path.join(results_dir, model, "logprob_scores.jsonl")
        if not os.path.isfile(lp_path):
            continue
        df = load_logprob_results(lp_path)
        ch = df[df["condition"] == "challenge"]
        if non_simple_only:
            ch = ch[ch["challenge_type"] != "simple"]
        if len(ch) == 0:
            continue
        out[model] = float(ch.groupby("question_id")["delta_log_odds"].mean().mean())
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Length-matched wrong-answer robustness check for ΔLogOdds")
    parser.add_argument("--experiment-dir", required=True,
                        help="Results dir for length-matched condition "
                             "(e.g. data/results/exp_length_robustness)")
    parser.add_argument("--baseline-dir", required=True,
                        help="Results dir for original (unmatched) condition "
                             "(e.g. data/results/exp1)")
    parser.add_argument("--dataset", default="medical_advice",
                        help="Dataset subdirectory to compare (default: medical_advice)")
    parser.add_argument("--processed-path", default=None,
                        help="Processed JSONL with proposed_answer_length_matched field "
                             "(to compute pre/post word-count ratios). "
                             "If omitted, ratio stats are skipped.")
    parser.add_argument("--output", default=None, help="Path to save JSON report")
    args = parser.parse_args()

    exp_ds_dir = os.path.join(args.experiment_dir, args.dataset)
    base_ds_dir = os.path.join(args.baseline_dir, args.dataset)

    matched_means = per_model_mean_dlo(exp_ds_dir)
    orig_means = per_model_mean_dlo(base_ds_dir)

    common = sorted(set(matched_means) & set(orig_means))
    if not common:
        print("No common models found between experiment and baseline directories.")
        return

    orig_vec = np.array([orig_means[m] for m in common])
    matched_vec = np.array([matched_means[m] for m in common])
    rho, p = spearmanr(orig_vec, matched_vec) if len(common) >= 3 else (float("nan"), float("nan"))

    print(f"\nLength-matched ΔLogOdds robustness — {args.dataset}")
    print(f"{'Model':<35} {'orig ΔLO':>10} {'matched ΔLO':>12}")
    print("-" * 60)
    for m in common:
        print(f"{m:<35} {orig_means[m]:>10.3f} {matched_means[m]:>12.3f}")
    print(f"\nSpearman ρ (orig vs matched): {rho:.3f}  (p={p:.3f}, n={len(common)} models)")

    report: dict = {
        "dataset": args.dataset,
        "n_models": len(common),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "model_means": {m: {"original": orig_means[m], "matched": matched_means[m]}
                        for m in common},
    }

    if args.processed_path and os.path.isfile(args.processed_path):
        ratio_stats = compute_word_count_ratio(args.processed_path)
        report["word_count_ratios"] = ratio_stats
        print(f"\nWord-count ratio (wrong/correct):")
        print(f"  Before matching: {ratio_stats['mean_ratio_original']:.2f}  "
              f"({100*ratio_stats['pct_trimmed']:.0f}% of items trimmed)")
        print(f"  After matching:  {ratio_stats['mean_ratio_matched']:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
