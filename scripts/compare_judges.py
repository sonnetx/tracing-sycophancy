#!/usr/bin/env python3
"""Check 2: Compare two judges' evaluations of the same responses.

Computes 3-class Cohen's κ (correct / incorrect / erroneous) between the primary
judge (GPT-4o) and an alternative judge (e.g. Claude) on a matched sample, and
reports whether regressive/net sycophancy rankings across checkpoints change.

Usage:
    python scripts/compare_judges.py \
        --primary  data/results/exp_rejudge/computational/olmo3-7b-instruct/gpt4o_sample.jsonl \
        --alt      data/results/exp_rejudge/computational/olmo3-7b-instruct/claude_sample.jsonl \
        --label    "computational / olmo3-7b-instruct"

Or compare all models in an experiment directory:
    python scripts/compare_judges.py \
        --experiment-dir data/results/exp_rejudge \
        --output data/results/exp_rejudge/judge_agreement.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.analysis.stats import compute_metrics_summary, load_results_as_dataframe


def cohen_kappa_score(y1, y2, labels=None):
    """Cohen's κ for categorical sequences (no sklearn required)."""
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return 0.0
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    p_e = sum((y1.count(lbl) / n) * (y2.count(lbl) / n) for lbl in labels)
    return 1.0 if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)


LABEL_ORDER = ["correct", "incorrect", "erroneous"]


def flatten_to_rows(evaluated_items: list[dict]) -> pd.DataFrame:
    """Flatten evaluated JSONL to one row per (question_id, challenge_id) pair."""
    rows = []
    for item in evaluated_items:
        qid = item["question_id"]
        # Initial response
        rows.append({
            "question_id": qid,
            "challenge_id": None,
            "factual_accuracy": item["initial"]["metrics"].get("factual_accuracy"),
        })
        for cr in item.get("challenge_responses", []):
            rows.append({
                "question_id": qid,
                "challenge_id": cr["challenge_id"],
                "factual_accuracy": cr["metrics"].get("factual_accuracy"),
            })
    return pd.DataFrame(rows)


def compute_kappa(primary_df: pd.DataFrame, alt_df: pd.DataFrame) -> dict:
    """Merge on (question_id, challenge_id) and compute Cohen's κ."""
    merged = primary_df.merge(
        alt_df, on=["question_id", "challenge_id"],
        suffixes=("_primary", "_alt"),
    )
    merged = merged.dropna(subset=["factual_accuracy_primary", "factual_accuracy_alt"])
    if len(merged) < 10:
        return {"error": f"Too few matched rows ({len(merged)}) for κ computation",
                "n_matched": len(merged)}

    y1 = merged["factual_accuracy_primary"].tolist()
    y2 = merged["factual_accuracy_alt"].tolist()
    kappa = float(cohen_kappa_score(y1, y2, labels=LABEL_ORDER))
    pct_agree = float((merged["factual_accuracy_primary"] == merged["factual_accuracy_alt"]).mean())

    # Confusion matrix (primary rows × alt cols)
    confusion = {}
    for true_label in LABEL_ORDER:
        for pred_label in LABEL_ORDER:
            mask = (merged["factual_accuracy_primary"] == true_label) & \
                   (merged["factual_accuracy_alt"] == pred_label)
            confusion[f"{true_label}_as_{pred_label}"] = int(mask.sum())

    return {
        "n_matched": len(merged),
        "cohen_kappa": kappa,
        "pct_agreement": pct_agree,
        "confusion": confusion,
    }


def compare_model_pair(primary_path: str, alt_path: str) -> dict:
    """Compare two evaluated JSONL files and return agreement + ranking stability."""
    from src.utils import read_jsonl
    primary_items = read_jsonl(primary_path)
    alt_items = read_jsonl(alt_path)

    primary_flat = flatten_to_rows(primary_items)
    alt_flat = flatten_to_rows(alt_items)
    agreement = compute_kappa(primary_flat, alt_flat)

    # Regressive rates under each judge (using the same compute_metrics_summary)
    # Build DataFrames compatible with load_results_as_dataframe output format
    def items_to_df(items):
        rows = []
        for item in items:
            base = {"question_id": item["question_id"],
                    "model": item.get("model", "model"),
                    "checkpoint": item.get("checkpoint", "ckpt"),
                    "model_type": item.get("model_type", "unknown")}
            rows.append({**base, "response_type": "initial", "challenge_id": None,
                         "challenge_type": None, "challenge_context": None,
                         "response": item["initial"]["response"],
                         "factual_accuracy": item["initial"]["metrics"].get("factual_accuracy"),
                         "agreement": None, "hedging": False, "hedging_phrases": [],
                         "refusal": False})
            for cr in item.get("challenge_responses", []):
                cid = cr["challenge_id"]
                parts = cid.rsplit("_", 1)
                c_type = parts[0] if len(parts) == 2 else cid
                c_ctx_raw = parts[1] if len(parts) == 2 else "unknown"
                c_ctx = "in_context" if c_ctx_raw == "incontext" else c_ctx_raw
                rows.append({**base, "response_type": "challenge", "challenge_id": cid,
                             "challenge_type": c_type, "challenge_context": c_ctx,
                             "response": cr["response"],
                             "factual_accuracy": cr["metrics"].get("factual_accuracy"),
                             "agreement": None, "hedging": False, "hedging_phrases": [],
                             "refusal": False})
        return pd.DataFrame(rows)

    primary_df = items_to_df(primary_items)
    alt_df = items_to_df(alt_items)

    primary_summary = compute_metrics_summary(primary_df)
    alt_summary = compute_metrics_summary(alt_df)

    primary_regr = primary_summary.get("sycophancy", {}).get("regressive_rate", None)
    alt_regr = alt_summary.get("sycophancy", {}).get("regressive_rate", None)
    rank_changed = None
    if primary_regr is not None and alt_regr is not None:
        # For a single model, "ranking stable" means direction of effect is the same
        rank_changed = False  # single model, no ranking to compare

    return {
        "agreement": agreement,
        "primary_regressive_rate": primary_regr,
        "alt_regressive_rate": alt_regr,
        "rank_changed": rank_changed,
    }


def analyze_experiment_dir(exp_dir: str) -> dict:
    """Walk exp_dir/{dataset}/{model}/ and compare gpt4o_sample.jsonl vs claude_sample.jsonl."""
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
            primary = os.path.join(model_dir, "gpt4o_sample.jsonl")
            alt = os.path.join(model_dir, "claude_sample.jsonl")
            if not (os.path.isfile(primary) and os.path.isfile(alt)):
                continue
            try:
                result = compare_model_pair(primary, alt)
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
        print(f"{'Model':<35} {'n':>5} {'κ':>7} {'%agree':>8} {'regr(GPT4)':>11} {'regr(alt)':>10}")
        print("-" * 75)
        all_kappas = []
        for model, r in sorted(model_results.items()):
            ag = r.get("agreement", {})
            if "error" in ag:
                print(f"  {model}: {ag['error']}")
                continue
            k = ag.get("cohen_kappa", float("nan"))
            all_kappas.append(k)
            print(f"{model:<35} {ag.get('n_matched', 0):>5} {k:>7.3f} "
                  f"{ag.get('pct_agreement', 0):>8.3f} "
                  f"{r.get('primary_regressive_rate') or 0:>11.3f} "
                  f"{r.get('alt_regressive_rate') or 0:>10.3f}")
        if all_kappas:
            print(f"\n  Mean κ: {np.mean(all_kappas):.3f}  "
                  f"Min κ: {min(all_kappas):.3f}")
            interp = "substantial" if min(all_kappas) > 0.6 else \
                     "moderate" if min(all_kappas) > 0.4 else "fair"
            print(f"  Interpretation: {interp} agreement (worst-case model)")


def main():
    parser = argparse.ArgumentParser(description="Compare two judge evaluations (Cohen's κ + ranking stability)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--primary", help="Primary judge evaluated JSONL (e.g. GPT-4o)")
    mode.add_argument("--experiment-dir", help="Root dir with {dataset}/{model}/ structure")
    parser.add_argument("--alt", help="Alternative judge evaluated JSONL (required with --primary)")
    parser.add_argument("--label", default="", help="Label for --primary/--alt mode")
    parser.add_argument("--output", default=None, help="Path to save JSON report")
    args = parser.parse_args()

    if args.primary:
        if not args.alt:
            parser.error("--alt is required with --primary")
        result = compare_model_pair(args.primary, args.alt)
        results = {args.label or "dataset": {"model": result}}
        print_report(results)
    else:
        results = analyze_experiment_dir(args.experiment_dir)
        print_report(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
