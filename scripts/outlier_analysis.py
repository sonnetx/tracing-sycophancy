#!/usr/bin/env python3
"""Outlier analysis for per-item ΔLogOdds.

Heavy-tailed ΔLogOdds distributions mean a handful of items can drag the mean
far from the median. This script quantifies how much of each checkpoint's mean
ΔLogOdds is contributed by the top-k most sycophantic items, and writes out
those items joined with question / challenge text for qualitative inspection.

Outputs per dataset into {dataset_dir}/analysis/:
  1. outlier_contribution.json    — top-k contribution to mean per checkpoint
                                    (mean, mean_without_top_k, share, trimmed means).
  2. outlier_items.csv            — top-k items per checkpoint with question text,
                                    challenge text, correct/proposed answer,
                                    ΔLogOdds, and context/type tags.
  3. outlier_characteristics.json — aggregate stats on outlier items:
                                    challenge-type distribution, context distribution,
                                    length stats relative to non-outliers.

Usage:
    python scripts/outlier_analysis.py --experiment-dir data/results/exp1
    python scripts/outlier_analysis.py --experiment-dir data/results/exp1 --top-k 10
"""

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from src.analysis.stats import load_logprob_results
from src.utils import read_jsonl


def load_question_metadata(processed_path: str) -> dict:
    """Load processed JSONL into {question_id: {question, correct_answer,
    proposed_answer, challenges: {challenge_id: {text, type, context}}}}."""
    meta = {}
    for it in read_jsonl(processed_path):
        qid = it.get("id") or it.get("question_id")
        challenges = {}
        for ch in it.get("challenges", []) or []:
            cid = ch.get("challenge_id") or ch.get("id")
            if cid is None:
                continue
            challenges[cid] = {
                "text": ch.get("text") or ch.get("challenge") or "",
                "type": ch.get("challenge_type") or ch.get("type"),
                "context": ch.get("challenge_context") or ch.get("context"),
            }
        meta[qid] = {
            "question": it.get("question", "") or it.get("prompt", "") or "",
            "correct_answer": it.get("correct_answer", "") or "",
            "proposed_answer": it.get("proposed_answer", "") or "",
            "len_question": len(it.get("question", "") or ""),
            "len_correct": len(it.get("correct_answer", "") or ""),
            "len_proposed": len(it.get("proposed_answer", "") or ""),
            "challenges": challenges,
        }
    return meta


def checkpoint_outlier_stats(dlo: pd.Series, top_k: int) -> dict:
    """How much does the top-k (by ΔLogOdds, descending) drive the mean?"""
    s = pd.Series(dlo).dropna()
    n = len(s)
    if n == 0:
        return {}
    mean = float(s.mean())
    median = float(s.median())
    k = min(top_k, n)
    top = s.nlargest(k)
    rest = s.drop(top.index)
    mean_without_top = float(rest.mean()) if len(rest) > 0 else 0.0
    # What share of the total sum comes from the top-k?
    total_sum = float(s.sum())
    top_sum = float(top.sum())
    share = (top_sum / total_sum) if total_sum != 0 else float("nan")
    # Symmetric trimmed mean (drops top & bottom 5% and 10%)
    trimmed_05 = float(s[(s >= s.quantile(0.05)) & (s <= s.quantile(0.95))].mean()) if n >= 20 else float("nan")
    trimmed_10 = float(s[(s >= s.quantile(0.10)) & (s <= s.quantile(0.90))].mean()) if n >= 10 else float("nan")
    return {
        "n": int(n),
        "mean": mean,
        "median": median,
        "top_k": int(k),
        "mean_without_top_k": mean_without_top,
        "shift_from_dropping_top_k": mean - mean_without_top,
        "top_k_share_of_sum": share,
        "trimmed_mean_5pct": trimmed_05,
        "trimmed_mean_10pct": trimmed_10,
    }


def collect_top_outliers(lp_df: pd.DataFrame, meta: dict, model: str,
                          dataset: str, top_k: int) -> list:
    """Per-checkpoint, return top-k challenge rows joined with question metadata."""
    out = []
    challenges = lp_df[lp_df["condition"] == "challenge"].copy()
    for ckpt, grp in challenges.groupby("checkpoint"):
        top = grp.nlargest(top_k, "delta_log_odds")
        for _, row in top.iterrows():
            qid = row["question_id"]
            m = meta.get(qid, {})
            ch_meta = m.get("challenges", {}).get(row["challenge_id"], {})
            out.append({
                "dataset": dataset,
                "model": model,
                "checkpoint": ckpt,
                "question_id": qid,
                "challenge_id": row["challenge_id"],
                "challenge_type": row.get("challenge_type") or ch_meta.get("type"),
                "challenge_context": row.get("challenge_context") or ch_meta.get("context"),
                "delta_log_odds": float(row["delta_log_odds"]),
                "baseline_log_odds": float(row["log_odds"]) - float(row["delta_log_odds"]),
                "challenged_log_odds": float(row["log_odds"]),
                "question": (m.get("question", "") or "")[:300],
                "correct_answer": (m.get("correct_answer", "") or "")[:200],
                "proposed_answer": (m.get("proposed_answer", "") or "")[:200],
                "challenge_text": (ch_meta.get("text", "") or "")[:300],
                "len_question": m.get("len_question", 0),
                "len_correct": m.get("len_correct", 0),
                "len_proposed": m.get("len_proposed", 0),
            })
    return out


def characterize_outliers(outlier_rows: list, lp_df_all: pd.DataFrame,
                           meta: dict) -> dict:
    """Compare outlier items to non-outlier population on type/context/length."""
    if not outlier_rows:
        return {}
    outlier_df = pd.DataFrame(outlier_rows)

    # Challenge-type and context distributions among outliers
    type_counts = Counter(outlier_df["challenge_type"].dropna())
    ctx_counts = Counter(outlier_df["challenge_context"].dropna())

    # Baseline distribution across all challenges in the dataset (for comparison)
    all_ch = lp_df_all[lp_df_all["condition"] == "challenge"]
    all_type = Counter(all_ch["challenge_type"].dropna())
    all_ctx = Counter(all_ch["challenge_context"].dropna())

    def norm(c):
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}

    # Length stats: are outliers longer / shorter than the median item?
    all_qids = set(meta.keys())
    outlier_qids = set(outlier_df["question_id"])
    lens_outlier_q = [meta[q]["len_question"] for q in outlier_qids if q in meta]
    lens_all_q = [meta[q]["len_question"] for q in all_qids]
    lens_outlier_c = [meta[q]["len_correct"] for q in outlier_qids if q in meta]
    lens_all_c = [meta[q]["len_correct"] for q in all_qids]
    lens_outlier_p = [meta[q]["len_proposed"] for q in outlier_qids if q in meta]
    lens_all_p = [meta[q]["len_proposed"] for q in all_qids]

    def mm(xs):
        if not xs:
            return {"mean": None, "median": None, "n": 0}
        return {"mean": float(np.mean(xs)), "median": float(np.median(xs)), "n": len(xs)}

    return {
        "n_outlier_rows": len(outlier_df),
        "n_unique_outlier_questions": len(outlier_qids),
        "challenge_type_share_outliers": norm(type_counts),
        "challenge_type_share_all": norm(all_type),
        "challenge_context_share_outliers": norm(ctx_counts),
        "challenge_context_share_all": norm(all_ctx),
        "length_question_outliers": mm(lens_outlier_q),
        "length_question_all": mm(lens_all_q),
        "length_correct_outliers": mm(lens_outlier_c),
        "length_correct_all": mm(lens_all_c),
        "length_proposed_outliers": mm(lens_outlier_p),
        "length_proposed_all": mm(lens_all_p),
    }


def analyze_dataset(dataset_dir: str, processed_path: str, top_k: int) -> None:
    dataset = os.path.basename(os.path.normpath(dataset_dir))
    print(f"\n=== {dataset} ===")
    if not os.path.isfile(processed_path):
        print(f"[SKIP] processed JSONL not found: {processed_path}")
        return

    meta = load_question_metadata(processed_path)
    print(f"Loaded metadata for {len(meta)} questions from {processed_path}")

    contributions = {}  # model -> {checkpoint -> stats}
    all_outlier_rows = []
    combined_challenges = []  # for characterize_outliers

    print(f"\n{'model/ckpt':>36s}  {'mean':>8s}  {'median':>8s}  "
          f"{'no-top'+str(top_k):>8s}  {'share':>6s}  {'trim10':>8s}  {'n':>5s}")

    for entry in sorted(os.listdir(dataset_dir)):
        lp_path = os.path.join(dataset_dir, entry, "logprob_scores.jsonl")
        if not os.path.isfile(lp_path):
            continue
        lp_df = load_logprob_results(lp_path)
        challenges = lp_df[lp_df["condition"] == "challenge"]
        combined_challenges.append(challenges)

        per_ckpt = {}
        for ckpt, grp in challenges.groupby("checkpoint"):
            stats = checkpoint_outlier_stats(grp["delta_log_odds"], top_k=top_k)
            per_ckpt[str(ckpt)] = stats
            print(f"{(entry + '/' + str(ckpt)):>36s}  "
                  f"{stats['mean']:>+8.3f}  {stats['median']:>+8.3f}  "
                  f"{stats['mean_without_top_k']:>+8.3f}  "
                  f"{stats['top_k_share_of_sum']:>6.2%}  "
                  f"{stats['trimmed_mean_10pct']:>+8.3f}  "
                  f"{stats['n']:>5d}")
        contributions[entry] = per_ckpt

        rows = collect_top_outliers(lp_df, meta, model=entry,
                                     dataset=dataset, top_k=top_k)
        all_outlier_rows.extend(rows)

    analysis_dir = os.path.join(dataset_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. contribution stats
    contrib_path = os.path.join(analysis_dir, "outlier_contribution.json")
    with open(contrib_path, "w") as f:
        json.dump({"top_k": top_k, "per_model": contributions}, f, indent=2)

    # 2. per-item outlier CSV
    csv_path = os.path.join(analysis_dir, "outlier_items.csv")
    if all_outlier_rows:
        fields = list(all_outlier_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in all_outlier_rows:
                writer.writerow(row)

    # 3. aggregate characteristics
    char_path = os.path.join(analysis_dir, "outlier_characteristics.json")
    lp_df_all = pd.concat(combined_challenges, ignore_index=True) if combined_challenges else pd.DataFrame()
    chars = characterize_outliers(all_outlier_rows, lp_df_all, meta)
    with open(char_path, "w") as f:
        json.dump(chars, f, indent=2)

    print(f"\nSaved outlier_contribution.json, outlier_items.csv, "
          f"outlier_characteristics.json to {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True,
                        help="Experiment directory (e.g. data/results/exp1)")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing {dataset}.jsonl")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top-|ΔLogOdds| items to surface per checkpoint (default 10)")
    args = parser.parse_args()

    for entry in sorted(os.listdir(args.experiment_dir)):
        ds_dir = os.path.join(args.experiment_dir, entry)
        if not os.path.isdir(ds_dir):
            continue
        if entry in ("cross_dataset", "paper_figures"):
            continue
        processed_path = os.path.join(args.processed_dir, f"{entry}.jsonl")
        analyze_dataset(ds_dir, processed_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
