#!/usr/bin/env python3
"""Validate GPT-4o LLM-as-a-Judge accuracy via human annotation.

Two modes:
  1. SAMPLE: Extract random items from evaluated.jsonl into a CSV for human review.
  2. COMPUTE: Read the annotated CSV and compute agreement + beta distribution stats.

Usage:
    # Step 1 — generate annotation sheet
    python scripts/validate_judge.py sample \
        --input data/results/exp1/computational/olmo3-7b-instruct/evaluated.jsonl \
        --output judge_validation_computational.csv \
        --n 50

    # Step 2 — after filling in 'human_label' column in the CSV
    python scripts/validate_judge.py compute \
        --input judge_validation_computational.csv

Follows the beta-distribution validation approach from Fanous et al. (2025) — SycEval.
"""

import argparse
import csv
import json
import random
import sys

from src.analysis.stats import beta_distribution_ci
from src.utils import read_jsonl


def _collect_rows(item: dict, model_name: str = "",
                  qid_to_meta: dict | None = None) -> list[dict]:
    qid = item["question_id"]
    meta = (qid_to_meta or {}).get(qid, {})
    question = meta.get("question", "")
    correct = meta.get("correct_answer", "")
    initial = item["initial"]
    i_metrics = initial.get("metrics", {})
    rows = [{
        "question_id": qid,
        "model": model_name,
        "condition": "initial",
        "challenge_id": "",
        "question": question,
        "correct_answer": correct,
        "model_response": initial["response"],
        "judge_factual_accuracy": i_metrics.get("factual_accuracy", ""),
        "judge_agreement": str(i_metrics.get("agreement", "")),
        "human_factual_accuracy": "",
        "human_agreement": "",
    }]
    for cr in item.get("challenge_responses", []):
        cr_metrics = cr.get("metrics", {})
        if cr_metrics.get("skipped"):
            continue
        rows.append({
            "question_id": qid,
            "model": model_name,
            "condition": "challenge",
            "challenge_id": cr["challenge_id"],
            "question": question,
            "correct_answer": correct,
            "model_response": cr["response"],
            "judge_factual_accuracy": cr_metrics.get("factual_accuracy", ""),
            "judge_agreement": str(cr_metrics.get("agreement", "")),
            "human_factual_accuracy": "",
            "human_agreement": "",
        })
    return rows


def _load_qid_to_meta(processed_path: str) -> dict:
    out = {}
    for item in read_jsonl(processed_path):
        qid = item.get("id") or item.get("question_id")
        if not qid:
            continue
        out[qid] = {
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
        }
    return out


def sample_for_annotation(input_path: str, output_path: str, n: int,
                          seed: int = 42, stratify: bool = False,
                          results_dir: str | None = None,
                          processed: str | None = None):
    import os

    qid_to_meta = _load_qid_to_meta(processed) if processed else {}
    if processed:
        print(f"Loaded {len(qid_to_meta)} (question, correct_answer) entries from {processed}")

    rows: list[dict] = []
    if results_dir:
        for entry in sorted(os.listdir(results_dir)):
            ev_path = os.path.join(results_dir, entry, "evaluated.jsonl")
            if not os.path.isfile(ev_path):
                continue
            for item in read_jsonl(ev_path):
                rows.extend(_collect_rows(item, model_name=entry, qid_to_meta=qid_to_meta))
    else:
        for item in read_jsonl(input_path):
            rows.extend(_collect_rows(item, qid_to_meta=qid_to_meta))

    if not rows:
        print(f"No items found.")
        return

    random.seed(seed)
    if stratify:
        # Stratify by judge_factual_accuracy label to ensure coverage of
        # minority classes (incorrect / erroneous).
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            label = r["judge_factual_accuracy"] or "missing"
            buckets.setdefault(label, []).append(r)
        per_bucket = max(1, n // len(buckets))
        sampled: list[dict] = []
        for label, bucket in buckets.items():
            take = min(per_bucket, len(bucket))
            sampled.extend(random.sample(bucket, take))
            print(f"  stratum '{label}': sampled {take} / {len(bucket)}")
        # Top up to n with any remaining
        if len(sampled) < n:
            sampled_ids = {(r["question_id"], r["challenge_id"], r.get("model", "")) for r in sampled}
            remaining = [r for r in rows
                         if (r["question_id"], r["challenge_id"], r.get("model", "")) not in sampled_ids]
            extra = random.sample(remaining, min(n - len(sampled), len(remaining)))
            sampled.extend(extra)
        random.shuffle(sampled)
    else:
        sampled = random.sample(rows, min(n, len(rows)))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sampled[0].keys())
        writer.writeheader()
        writer.writerows(sampled)

    print(f"\nWrote {len(sampled)} items to {output_path}")
    print("Fill in 'human_factual_accuracy' (correct/incorrect/erroneous)")
    print("and 'human_agreement' (true/false/null) columns, then run:")
    print(f"  python scripts/validate_judge.py compute --input {output_path}")


def compute_validation(input_path: str):
    """Read annotated CSV and compute agreement stats + beta distribution."""
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to rows with human annotations
    annotated = [r for r in rows if r.get("human_factual_accuracy", "").strip()]
    if not annotated:
        print("No human annotations found. Fill in 'human_factual_accuracy' column first.")
        return

    print(f"\n=== Judge Validation ({len(annotated)} annotated items) ===\n")

    # --- Factual accuracy agreement ---
    acc_matches = sum(
        1 for r in annotated
        if r["judge_factual_accuracy"].strip() == r["human_factual_accuracy"].strip()
    )
    acc_mismatches = len(annotated) - acc_matches

    acc_stats = beta_distribution_ci(acc_matches, acc_mismatches)
    print("Factual Accuracy Judge Agreement:")
    print(f"  Matches: {acc_matches}/{len(annotated)} ({acc_stats['accuracy']:.1%})")
    print(f"  Beta posterior mean: {acc_stats['beta_mean']:.3f}")
    print(f"  95% credible interval: [{acc_stats['ci_low']:.3f}, {acc_stats['ci_high']:.3f}]")

    # --- Agreement label agreement (where applicable) ---
    agree_rows = [
        r for r in annotated
        if r.get("human_agreement", "").strip()
        and r["human_agreement"].strip().lower() not in ("", "null", "none", "na")
        and r["judge_agreement"].strip().lower() not in ("", "null", "none", "na")
    ]

    if agree_rows:
        agree_matches = sum(
            1 for r in agree_rows
            if r["judge_agreement"].strip().lower() == r["human_agreement"].strip().lower()
        )
        agree_mismatches = len(agree_rows) - agree_matches
        agree_stats = beta_distribution_ci(agree_matches, agree_mismatches)
        print(f"\nAgreement Label Judge Agreement:")
        print(f"  Matches: {agree_matches}/{len(agree_rows)} ({agree_stats['accuracy']:.1%})")
        print(f"  Beta posterior mean: {agree_stats['beta_mean']:.3f}")
        print(f"  95% credible interval: [{agree_stats['ci_low']:.3f}, {agree_stats['ci_high']:.3f}]")

    # --- Confusion matrix for factual accuracy ---
    labels = ["correct", "incorrect", "erroneous"]
    cm = {jl: {hl: 0 for hl in labels} for jl in labels}
    for r in annotated:
        jl, hl = r["judge_factual_accuracy"].strip(), r["human_factual_accuracy"].strip()
        if jl in cm and hl in cm[jl]:
            cm[jl][hl] += 1
    print(f"\nConfusion Matrix (rows=judge, cols=human):")
    print(f"{'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels))
    for jl in labels:
        print(f"{jl:>12s}  " + "  ".join(f"{cm[jl][hl]:>10d}" for hl in labels))

    # --- Per-class precision and recall ---
    per_class = {}
    print(f"\nPer-class precision / recall:")
    print(f"{'label':>12s}  {'prec':>6s}  {'recall':>6s}  {'n_judge':>8s}  {'n_human':>8s}")
    for lbl in labels:
        tp = cm[lbl][lbl]
        judge_total = sum(cm[lbl][hl] for hl in labels)             # row sum
        human_total = sum(cm[jl][lbl] for jl in labels)             # col sum
        prec = tp / judge_total if judge_total else 0.0
        rec = tp / human_total if human_total else 0.0
        per_class[lbl] = {"precision": prec, "recall": rec,
                          "tp": tp, "judge_total": judge_total, "human_total": human_total}
        print(f"{lbl:>12s}  {prec:>6.3f}  {rec:>6.3f}  {judge_total:>8d}  {human_total:>8d}")

    # --- Cohen's kappa ---
    n = len(annotated)
    po = acc_matches / n if n else 0.0
    pe = 0.0
    for lbl in labels:
        judge_total = sum(cm[lbl][hl] for hl in labels)
        human_total = sum(cm[jl][lbl] for jl in labels)
        pe += (judge_total / n) * (human_total / n) if n else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    print(f"\nCohen's kappa: {kappa:.3f}  (p_o={po:.3f}, p_e={pe:.3f})")

    # Save results
    results = {
        "total_annotated": len(annotated),
        "factual_accuracy": acc_stats,
        "per_class": per_class,
        "cohens_kappa": kappa,
        "confusion_matrix": cm,
    }
    if agree_rows:
        results["agreement_label"] = agree_stats

    out_path = input_path.replace(".csv", "_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLM-as-a-Judge accuracy")
    sub = parser.add_subparsers(dest="command")

    sp_sample = sub.add_parser("sample", help="Sample items for human annotation")
    sp_sample.add_argument("--input", help="Path to a single evaluated.jsonl")
    sp_sample.add_argument("--results-dir",
                           help="Pool responses across all models in this dataset directory")
    sp_sample.add_argument("--output", required=True, help="Output CSV path")
    sp_sample.add_argument("--n", type=int, default=50, help="Number of items to sample")
    sp_sample.add_argument("--seed", type=int, default=42, help="Random seed")
    sp_sample.add_argument("--stratify", action="store_true",
                           help="Stratify sampling by judge factual-accuracy label")
    sp_sample.add_argument("--processed",
                           help=("Path to data/processed/{dataset}.jsonl. "
                                 "Required to populate question and correct_answer "
                                 "columns, since evaluated.jsonl does not store them."))

    sp_compute = sub.add_parser("compute", help="Compute agreement from annotated CSV")
    sp_compute.add_argument("--input", required=True, help="Path to annotated CSV")

    args = parser.parse_args()

    if args.command == "sample":
        if not args.input and not args.results_dir:
            parser.error("Provide either --input or --results-dir")
        sample_for_annotation(args.input, args.output, args.n, args.seed,
                              stratify=args.stratify, results_dir=args.results_dir,
                              processed=args.processed)
    elif args.command == "compute":
        compute_validation(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
