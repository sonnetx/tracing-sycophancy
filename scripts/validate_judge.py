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


def sample_for_annotation(input_path: str, output_path: str, n: int, seed: int = 42):
    """Sample n items from evaluated.jsonl and write a CSV for human annotation."""
    items = read_jsonl(input_path)
    if not items:
        print(f"No items found in {input_path}")
        return

    # Flatten: pick initial + a random challenge response per item
    rows = []
    for item in items:
        qid = item["question_id"]
        initial = item["initial"]
        i_metrics = initial.get("metrics", {})

        # Include the initial response
        rows.append({
            "question_id": qid,
            "condition": "initial",
            "challenge_id": "",
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
            "model_response": initial["response"][:500],
            "judge_factual_accuracy": i_metrics.get("factual_accuracy", ""),
            "judge_agreement": str(i_metrics.get("agreement", "")),
            "human_factual_accuracy": "",
            "human_agreement": "",
        })

        # Include challenge responses
        for cr in item.get("challenge_responses", []):
            cr_metrics = cr.get("metrics", {})
            if cr_metrics.get("skipped"):
                continue
            rows.append({
                "question_id": qid,
                "condition": "challenge",
                "challenge_id": cr["challenge_id"],
                "question": item.get("question", ""),
                "correct_answer": item.get("correct_answer", ""),
                "model_response": cr["response"][:500],
                "judge_factual_accuracy": cr_metrics.get("factual_accuracy", ""),
                "judge_agreement": str(cr_metrics.get("agreement", "")),
                "human_factual_accuracy": "",
                "human_agreement": "",
            })

    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sampled[0].keys())
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Wrote {len(sampled)} items to {output_path}")
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
    print(f"\nConfusion Matrix (rows=judge, cols=human):")
    print(f"{'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels))
    for jl in labels:
        counts = []
        for hl in labels:
            c = sum(1 for r in annotated if r["judge_factual_accuracy"].strip() == jl and r["human_factual_accuracy"].strip() == hl)
            counts.append(c)
        print(f"{jl:>12s}  " + "  ".join(f"{c:>10d}" for c in counts))

    # Save results
    results = {
        "total_annotated": len(annotated),
        "factual_accuracy": acc_stats,
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
    sp_sample.add_argument("--input", required=True, help="Path to evaluated.jsonl")
    sp_sample.add_argument("--output", required=True, help="Output CSV path")
    sp_sample.add_argument("--n", type=int, default=50, help="Number of items to sample")
    sp_sample.add_argument("--seed", type=int, default=42, help="Random seed")

    sp_compute = sub.add_parser("compute", help="Compute agreement from annotated CSV")
    sp_compute.add_argument("--input", required=True, help="Path to annotated CSV")

    args = parser.parse_args()

    if args.command == "sample":
        sample_for_annotation(args.input, args.output, args.n, args.seed)
    elif args.command == "compute":
        compute_validation(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
