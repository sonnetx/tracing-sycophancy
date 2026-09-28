#!/usr/bin/env python3
"""Validate generated wrong answers for quality.

For AMPS (computational): uses SymPy to detect accidentally-equivalent answers.
For MedQuAD (medical_advice): uses a GPT-4o judge to classify each proposed
wrong answer as clearly_wrong / ambiguous / accidentally_correct.

Outputs:
  <output_dir>/report.json          — machine-readable summary
  <output_dir>/report.txt           — human-readable report
  <output_dir>/flagged_items.csv    — items needing manual review
  <output_dir>/cached_results.jsonl — per-item classifications (resume support)

Usage:
    # Computational (SymPy only, no API key needed)
    python scripts/validate_wrong_answers.py \
        --input data/processed/computational.jsonl \
        --dataset computational \
        --output-dir data/validation/computational

    # Medical (GPT-4o judge)
    python scripts/validate_wrong_answers.py \
        --input data/processed/medical_advice.jsonl \
        --dataset medical_advice \
        --judge-config config/models/gpt4o_judge.json \
        --output-dir data/validation/medical_advice

    # Medical with human annotation sample (50 items stratified by qtype)
    python scripts/validate_wrong_answers.py \
        --input data/processed/medical_advice.jsonl \
        --dataset medical_advice \
        --judge-config config/models/gpt4o_judge.json \
        --output-dir data/validation/medical_advice \
        --annotation-sample 50
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Computational: SymPy equivalence check
# ---------------------------------------------------------------------------

def _validate_computational(items: list[dict], cached: dict) -> list[dict]:
    from src.validation.sympy_checker import check_equivalence

    results = []
    for item in tqdm(items, desc="SymPy equivalence check"):
        qid = item["id"]
        if qid in cached:
            results.append(cached[qid])
            continue
        correct = item.get("correct_answer", "")
        proposed = item.get("proposed_answer", "")
        if not proposed or "PLACEHOLDER" in proposed:
            result = {
                "id": qid, "category": item.get("category", ""),
                "question": item["question"],
                "correct_answer": correct,
                "proposed_answer": proposed,
                "status": "skip", "detail": "no proposed answer",
            }
        else:
            check = check_equivalence(correct, proposed)
            result = {
                "id": qid, "category": item.get("category", ""),
                "question": item["question"],
                "correct_answer": correct,
                "proposed_answer": proposed,
                "status": check["status"],
                "detail": check["detail"],
            }
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Medical: GPT-4o judge classification
# ---------------------------------------------------------------------------

def _validate_medical(items: list[dict], judge, cached: dict) -> list[dict]:
    from src.validation.wrong_answer_judge import classify

    results = []
    for item in tqdm(items, desc="Judge classification"):
        qid = item["id"]
        if qid in cached:
            results.append(cached[qid])
            continue
        proposed = item.get("proposed_answer", "")
        if not proposed or "PLACEHOLDER" in proposed:
            result = {
                "id": qid, "category": item.get("category", ""),
                "question": item["question"],
                "correct_answer": item.get("correct_answer", ""),
                "proposed_answer": proposed,
                "classification": "skip",
                "reasoning": "no proposed answer",
            }
        else:
            cls = classify(item["question"], item["correct_answer"], proposed, judge)
            result = {
                "id": qid, "category": item.get("category", ""),
                "question": item["question"],
                "correct_answer": item.get("correct_answer", ""),
                "proposed_answer": proposed,
                "classification": cls["classification"],
                "reasoning": cls["reasoning"],
            }
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _compute_stats_computational(results: list[dict]) -> dict:
    valid = [r for r in results if r["status"] != "skip"]
    counts = {"equivalent": 0, "not_equivalent": 0, "parse_error": 0}
    by_category: dict[str, dict] = {}
    for r in valid:
        counts[r["status"]] += 1
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, {"equivalent": 0, "not_equivalent": 0, "parse_error": 0})
        by_category[cat][r["status"]] += 1
    n = len(valid)
    parse_errors = [r for r in valid if r["status"] == "parse_error"]
    multipart_count = sum(
        1 for r in parse_errors
        if "multi" in r.get("detail", "") or "structured" in r.get("detail", "")
    )
    return {
        "n_total": len(results), "n_validated": n,
        "counts": counts,
        "rates": {k: v / n if n else 0.0 for k, v in counts.items()},
        "by_category": by_category,
        "parse_error_multipart": multipart_count,
        "parse_error_unparseable": len(parse_errors) - multipart_count,
        "flagged": [r for r in valid if r["status"] in ("equivalent", "parse_error")],
    }


def _compute_stats_medical(results: list[dict]) -> dict:
    valid = [r for r in results if r["classification"] != "skip"]
    counts = {"clearly_wrong": 0, "ambiguous": 0, "accidentally_correct": 0}
    by_category: dict[str, dict] = {}
    for r in valid:
        counts[r["classification"]] += 1
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, {"clearly_wrong": 0, "ambiguous": 0, "accidentally_correct": 0})
        by_category[cat][r["classification"]] += 1
    n = len(valid)
    return {
        "n_total": len(results), "n_validated": n,
        "counts": counts,
        "rates": {k: v / n if n else 0.0 for k, v in counts.items()},
        "by_category": by_category,
        "flagged": [r for r in valid if r["classification"] in ("ambiguous", "accidentally_correct")],
    }


def _write_report_txt(stats: dict, dataset: str, output_dir: Path):
    lines = [
        f"Wrong-Answer Validation Report — {dataset}",
        "=" * 60,
        f"Total items:     {stats['n_total']}",
        f"Validated items: {stats['n_validated']}",
        "",
    ]
    if dataset == "computational":
        counts = stats["counts"]
        rates = stats["rates"]
        n = stats["n_validated"]
        multipart = stats.get("parse_error_multipart", 0)
        unparseable = stats.get("parse_error_unparseable", 0)
        lines += [
            "SymPy equivalence results:",
            f"  not_equivalent  : {counts['not_equivalent']:>4d} / {n}  ({rates['not_equivalent']:.1%})  — clearly wrong ✓",
            f"  equivalent      : {counts['equivalent']:>4d} / {n}  ({rates['equivalent']:.1%})  — FLAGGED: accidentally correct",
            f"  parse_error     : {counts['parse_error']:>4d} / {n}  ({rates['parse_error']:.1%})",
            f"    of which multi-part/structured (expected): {multipart}",
            f"    of which truly unparseable (manual review): {unparseable}",
            "",
            "Breakdown by category:",
        ]
        for cat, c in sorted(stats["by_category"].items()):
            tot = sum(c.values())
            lines.append(f"  {cat:<40s}  eq={c['equivalent']}  ne={c['not_equivalent']}  err={c['parse_error']}  (n={tot})")
    else:
        counts = stats["counts"]
        rates = stats["rates"]
        n = stats["n_validated"]
        lines += [
            "Judge classification results:",
            f"  clearly_wrong        : {counts['clearly_wrong']:>4d} / {n}  ({rates['clearly_wrong']:.1%})  — valid challenges ✓",
            f"  ambiguous            : {counts['ambiguous']:>4d} / {n}  ({rates['ambiguous']:.1%})  — FLAGGED: partially correct",
            f"  accidentally_correct : {counts['accidentally_correct']:>4d} / {n}  ({rates['accidentally_correct']:.1%})  — FLAGGED: generation failure",
            "",
            "Breakdown by qtype:",
        ]
        for cat, c in sorted(stats["by_category"].items()):
            tot = sum(c.values())
            lines.append(
                f"  {cat:<25s}  cw={c['clearly_wrong']}  amb={c['ambiguous']}  acc={c['accidentally_correct']}  (n={tot})"
            )

    flagged = stats["flagged"]
    lines += ["", f"Total flagged items: {len(flagged)}", ""]
    if flagged:
        lines.append("Flagged items written to flagged_items.csv for manual review.")

    txt = "\n".join(lines)
    (output_dir / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)


def _write_flagged_csv(flagged: list[dict], dataset: str, output_dir: Path):
    if not flagged:
        return
    if dataset == "computational":
        fieldnames = ["id", "category", "status", "detail", "correct_answer", "proposed_answer", "question"]
    else:
        fieldnames = ["id", "category", "classification", "reasoning", "correct_answer", "proposed_answer", "question"]
    with open(output_dir / "flagged_items.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flagged)


def _write_annotation_sample(results: list[dict], n: int, seed: int, output_dir: Path):
    """Stratified sample for human review: balanced across qtypes and classification labels."""
    valid = [r for r in results if r.get("classification") not in ("skip", None)]
    random.seed(seed)

    # Stratify by (category, classification)
    buckets: dict[tuple, list] = {}
    for r in valid:
        key = (r.get("category", "unknown"), r["classification"])
        buckets.setdefault(key, []).append(r)

    per_bucket = max(1, n // len(buckets))
    sampled = []
    for key, bucket in sorted(buckets.items()):
        take = min(per_bucket, len(bucket))
        sampled.extend(random.sample(bucket, take))
    if len(sampled) < n:
        sampled_ids = {r["id"] for r in sampled}
        remaining = [r for r in valid if r["id"] not in sampled_ids]
        extra = random.sample(remaining, min(n - len(sampled), len(remaining)))
        sampled.extend(extra)
    random.shuffle(sampled)

    fieldnames = ["id", "category", "classification", "reasoning",
                  "correct_answer", "proposed_answer", "question", "human_verdict"]
    with open(output_dir / "annotation_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sampled:
            writer.writerow({**r, "human_verdict": ""})
    print(f"\nAnnotation sample ({len(sampled)} items) written to {output_dir / 'annotation_sample.csv'}")
    print("Fill in 'human_verdict' column (clearly_wrong / ambiguous / accidentally_correct),")
    print("then re-run with --compute-annotation-agreement to report inter-rater agreement.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate generated wrong answers")
    parser.add_argument("--input", required=True,
                        help="Processed JSONL with proposed_answer fields")
    parser.add_argument("--dataset", required=True,
                        choices=["computational", "medical_advice"],
                        help="Dataset type determines validation method")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for report and flagged-items output")
    parser.add_argument("--judge-config", default=None,
                        help="Backend config JSON for GPT-4o judge (medical only)")
    parser.add_argument("--annotation-sample", type=int, default=0,
                        help="If >0, write a stratified annotation CSV of this many items (medical only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(args.input)
    print(f"Loaded {len(items)} items from {args.input}")

    # Resume support: load previously cached results
    cache_path = output_dir / "cached_results.jsonl"
    cached = {}
    if cache_path.exists():
        for r in read_jsonl(str(cache_path)):
            cached[r["id"]] = r
        print(f"Resuming from cache: {len(cached)} items already classified")

    if args.dataset == "computational":
        results = _validate_computational(items, cached)
        stats = _compute_stats_computational(results)
    else:
        if not args.judge_config:
            print("--judge-config is required for medical_advice dataset", file=sys.stderr)
            sys.exit(1)
        from src.utils import load_backend
        judge = load_backend(args.judge_config)
        results = _validate_medical(items, judge, cached)
        stats = _compute_stats_medical(results)

    # Save full results cache
    write_jsonl(results, str(cache_path))

    # Write outputs
    _write_report_txt(stats, args.dataset, output_dir)

    report_json = {k: v for k, v in stats.items() if k != "flagged"}
    report_json["n_flagged"] = len(stats["flagged"])
    (output_dir / "report.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )

    _write_flagged_csv(stats["flagged"], args.dataset, output_dir)

    if args.annotation_sample > 0 and args.dataset == "medical_advice":
        _write_annotation_sample(results, args.annotation_sample, args.seed, output_dir)

    print(f"\nOutputs written to {output_dir}/")


if __name__ == "__main__":
    main()
