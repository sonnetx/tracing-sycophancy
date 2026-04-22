#!/usr/bin/env python3
"""Length analysis for log-probability candidate completions.

Two outputs per dataset:
  1. length_stats.json       — distribution of lengths and length differences
                                between correct_answer and proposed (wrong) answer.
  2. logprob_summaries_length_matched.json — per-model log-prob summary
                                re-computed restricted to items where
                                |len(correct) - len(proposed)| / max(len) < threshold.


Usage:
    python scripts/length_analysis.py --experiment-dir data/results/exp1
    python scripts/length_analysis.py --experiment-dir data/results/exp1 --threshold 0.25
"""

import argparse
import json
import os

import pandas as pd

from src.analysis.stats import compute_logprob_summary, load_logprob_results
from src.utils import read_jsonl


def compute_length_stats(processed_path: str) -> tuple[dict, pd.DataFrame]:
    """Character-length statistics for (correct, proposed-wrong) pairs."""
    items = read_jsonl(processed_path)
    rows = []
    for it in items:
        qid = it.get("id") or it.get("question_id")
        c = it.get("correct_answer", "") or ""
        p = it.get("proposed_answer", "") or ""
        lc, lp = len(c), len(p)
        m = max(lc, lp)
        rel = abs(lc - lp) / m if m > 0 else 0.0
        rows.append({
            "question_id": qid,
            "len_correct": lc,
            "len_proposed": lp,
            "abs_diff": abs(lc - lp),
            "rel_diff": rel,
        })
    df = pd.DataFrame(rows)
    stats = {
        "n_items": int(len(df)),
        "mean_len_correct_chars": float(df["len_correct"].mean()),
        "mean_len_proposed_chars": float(df["len_proposed"].mean()),
        "median_len_correct_chars": float(df["len_correct"].median()),
        "median_len_proposed_chars": float(df["len_proposed"].median()),
        "mean_abs_diff_chars": float(df["abs_diff"].mean()),
        "median_abs_diff_chars": float(df["abs_diff"].median()),
        "mean_rel_diff": float(df["rel_diff"].mean()),
        "median_rel_diff": float(df["rel_diff"].median()),
        "frac_rel_diff_lt_0.25": float((df["rel_diff"] < 0.25).mean()),
        "frac_rel_diff_lt_0.50": float((df["rel_diff"] < 0.50).mean()),
    }
    return stats, df


def analyze_dataset(dataset_dir: str, processed_path: str, threshold: float) -> None:
    print(f"\n=== {dataset_dir} ===")
    if not os.path.isfile(processed_path):
        print(f"[SKIP] processed JSONL not found: {processed_path}")
        return

    stats, length_df = compute_length_stats(processed_path)
    print(f"Items: {stats['n_items']}")
    print(f"Mean chars  correct={stats['mean_len_correct_chars']:.1f}  "
          f"proposed={stats['mean_len_proposed_chars']:.1f}")
    print(f"|Δlen| chars  mean={stats['mean_abs_diff_chars']:.1f}  "
          f"median={stats['median_abs_diff_chars']:.1f}")
    print(f"Relative Δlen  mean={stats['mean_rel_diff']:.3f}  "
          f"median={stats['median_rel_diff']:.3f}")
    print(f"Fraction rel_diff<0.25: {stats['frac_rel_diff_lt_0.25']:.3f}")
    print(f"Fraction rel_diff<0.50: {stats['frac_rel_diff_lt_0.50']:.3f}")

    analysis_dir = os.path.join(dataset_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, "length_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    matched_qids = set(length_df[length_df["rel_diff"] < threshold]["question_id"])
    print(f"\nLength-matched subset (rel_diff < {threshold}): "
          f"{len(matched_qids)}/{stats['n_items']} items")

    matched_summaries: dict = {}
    print(f"\n{'model':>24s}  {'full ΔLO':>10s}  {'matched ΔLO':>12s}  {'Δ':>7s}  {'n_full':>7s}  {'n_matched':>9s}")
    for entry in sorted(os.listdir(dataset_dir)):
        lp_path = os.path.join(dataset_dir, entry, "logprob_scores.jsonl")
        if not os.path.isfile(lp_path):
            continue
        lp_df = load_logprob_results(lp_path)
        full_summary = compute_logprob_summary(lp_df)
        matched_df = lp_df[lp_df["question_id"].isin(matched_qids)]
        matched_summary = compute_logprob_summary(matched_df)
        matched_summaries[entry] = {
            "n_matched_items": len(matched_qids),
            "full": full_summary,
            "length_matched": matched_summary,
        }
        full_mean = full_summary.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0.0)
        matched_mean = matched_summary.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0.0)
        n_full = full_summary.get("total_questions", 0)
        n_matched = matched_summary.get("total_questions", 0)
        print(f"{entry:>24s}  {full_mean:>10.3f}  {matched_mean:>12.3f}  "
              f"{(matched_mean - full_mean):>+7.3f}  {n_full:>7d}  {n_matched:>9d}")

    out_path = os.path.join(analysis_dir, "logprob_summaries_length_matched.json")
    with open(out_path, "w") as f:
        json.dump(matched_summaries, f, indent=2)
    print(f"\nSaved length_stats.json and logprob_summaries_length_matched.json to {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True,
                        help="Experiment directory (e.g. data/results/exp1)")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing {dataset}.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Max |Δlen|/max(len) for length-matched subset (default 0.5)")
    args = parser.parse_args()

    for entry in sorted(os.listdir(args.experiment_dir)):
        ds_dir = os.path.join(args.experiment_dir, entry)
        if not os.path.isdir(ds_dir):
            continue
        if entry in ("cross_dataset", "paper_figures"):
            continue
        processed_path = os.path.join(args.processed_dir, f"{entry}.jsonl")
        analyze_dataset(ds_dir, processed_path, args.threshold)


if __name__ == "__main__":
    main()
