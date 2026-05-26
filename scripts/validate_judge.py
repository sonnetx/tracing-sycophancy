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
import hashlib
import json
import random
import sys
from collections import Counter

from src.analysis.stats import beta_distribution_ci
from src.utils import read_jsonl


# Columns an annotator is allowed to see. Judge labels and model identity are
# withheld so the human label isn't anchored to the judge or biased by which
# checkpoint produced the response.
BLIND_FIELDS = [
    "row_id", "question", "correct_answer", "model_response",
    "human_factual_accuracy", "human_agreement",
]

FACTUAL_LABELS = ["correct", "incorrect", "erroneous"]
AGREEMENT_LABELS = ["true", "false", "null"]


def _row_id(r: dict) -> str:
    """Stable id for a (model, question, condition, challenge) row, so two
    annotators' sheets and the master can be merged regardless of order."""
    key = f"{r.get('model','')}|{r['question_id']}|{r['condition']}|{r['challenge_id']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _cohens_kappa(pairs: list[tuple[str, str]], labels: list[str]):
    """pairs: list of (label_a, label_b). Returns dict with n, agreement, kappa."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": 0.0, "kappa": 0.0, "p_e": 0.0}
    matches = sum(1 for a, b in pairs if a == b)
    po = matches / n
    c_a = Counter(a for a, _ in pairs)
    c_b = Counter(b for _, b in pairs)
    pe = sum((c_a.get(l, 0) / n) * (c_b.get(l, 0) / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {"n": n, "agreement": po, "kappa": kappa, "p_e": pe}


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

    for r in sampled:
        r["row_id"] = _row_id(r)

    # Master sheet (internal): full metadata + judge labels + row_id.
    master_fields = ["row_id"] + [k for k in sampled[0].keys() if k != "row_id"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_fields)
        writer.writeheader()
        writer.writerows(sampled)

    # Blinded sheet (for annotators): no judge labels, no model identity.
    blind_path = output_path[:-4] + "_blind.csv" if output_path.endswith(".csv") \
        else output_path + "_blind.csv"
    with open(blind_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sampled:
            writer.writerow({k: r.get(k, "") for k in BLIND_FIELDS})

    print(f"\nWrote master sheet ({len(sampled)} rows) to {output_path}")
    print(f"Wrote blinded annotator sheet to {blind_path}")
    print("\nSend the BLINDED sheet to each annotator. They fill:")
    print("  human_factual_accuracy : correct / incorrect / erroneous")
    print("  human_agreement        : true / false / null")
    print("\nThen compute inter-annotator reliability + judge agreement:")
    print(f"  python scripts/validate_judge.py compute-irr \\")
    print(f"      --annotator-a annotatorA.csv --annotator-b annotatorB.csv \\")
    print(f"      --master {output_path}")
    print("\n(For a single annotator, the original judge-vs-human path still works:")
    print(f"  python scripts/validate_judge.py compute --input <filled_sheet>.csv )")


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


def _load_human_labels(path: str) -> dict:
    """{row_id: {"factual": ..., "agreement": ...}} from a filled annotator sheet."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("row_id", "").strip()
            if not rid:
                continue
            out[rid] = {
                "factual": r.get("human_factual_accuracy", "").strip().lower(),
                "agreement": r.get("human_agreement", "").strip().lower(),
            }
    return out


def _load_judge_labels(master_path: str) -> dict:
    out = {}
    with open(master_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("row_id", "").strip()
            if not rid:
                continue
            out[rid] = {
                "factual": r.get("judge_factual_accuracy", "").strip().lower(),
                "agreement": r.get("judge_agreement", "").strip().lower(),
            }
    return out


def compute_irr(annotator_a: str, annotator_b: str, master: str | None = None):
    """Inter-annotator reliability between two filled blinded sheets, plus
    (optionally) judge-vs-human and judge-vs-consensus by merging the master."""
    a = _load_human_labels(annotator_a)
    b = _load_human_labels(annotator_b)
    shared = sorted(set(a) & set(b))
    both = [rid for rid in shared if a[rid]["factual"] and b[rid]["factual"]]
    if not both:
        print("No row_id labeled by BOTH annotators — check the sheets share a row_id column.")
        return

    print(f"\n=== Inter-Annotator Reliability ({len(both)} rows labeled by both) ===\n")
    fa = _cohens_kappa([(a[r]["factual"], b[r]["factual"]) for r in both], FACTUAL_LABELS)
    print("Factual accuracy (annotator A vs B):")
    print(f"  agreement: {fa['agreement']:.1%}   Cohen's kappa: {fa['kappa']:.3f}")

    results = {"n_both_labeled": len(both), "factual_irr": fa}

    ag_both = [r for r in both
               if a[r]["agreement"] and b[r]["agreement"]
               and a[r]["agreement"] not in ("null", "none", "na")
               and b[r]["agreement"] not in ("null", "none", "na")]
    if ag_both:
        ag = _cohens_kappa([(a[r]["agreement"], b[r]["agreement"]) for r in ag_both],
                           AGREEMENT_LABELS)
        print(f"\nAgreement label (annotator A vs B, {ag['n']} rows):")
        print(f"  agreement: {ag['agreement']:.1%}   Cohen's kappa: {ag['kappa']:.3f}")
        results["agreement_irr"] = ag

    if master:
        judge = _load_judge_labels(master)
        for name, ann in (("A", a), ("B", b)):
            pairs = [(judge[r]["factual"], ann[r]["factual"])
                     for r in both if r in judge and judge[r]["factual"]]
            jk = _cohens_kappa(pairs, FACTUAL_LABELS)
            bstats = beta_distribution_ci(
                sum(1 for x, y in pairs if x == y),
                sum(1 for x, y in pairs if x != y))
            print(f"\nJudge vs annotator {name} (factual, {jk['n']} rows):")
            print(f"  agreement: {jk['agreement']:.1%}   kappa: {jk['kappa']:.3f}"
                  f"   beta-CI: [{bstats['ci_low']:.3f}, {bstats['ci_high']:.3f}]")
            results[f"judge_vs_{name}"] = {**jk, "beta": bstats}

        cons_pairs = [(judge[r]["factual"], a[r]["factual"])
                      for r in both
                      if a[r]["factual"] == b[r]["factual"]
                      and r in judge and judge[r]["factual"]]
        ck = _cohens_kappa(cons_pairs, FACTUAL_LABELS)
        cstats = beta_distribution_ci(
            sum(1 for x, y in cons_pairs if x == y),
            sum(1 for x, y in cons_pairs if x != y))
        print(f"\nJudge vs human consensus (rows where A==B, {ck['n']} rows):")
        print(f"  agreement: {ck['agreement']:.1%}   kappa: {ck['kappa']:.3f}"
              f"   beta-CI: [{cstats['ci_low']:.3f}, {cstats['ci_high']:.3f}]")
        results["judge_vs_consensus"] = {**ck, "beta": cstats}

    out_path = annotator_a[:-4] + "_irr_results.json" if annotator_a.endswith(".csv") \
        else annotator_a + "_irr_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved IRR results to {out_path}")


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

    sp_compute = sub.add_parser("compute", help="Single-annotator judge-vs-human agreement")
    sp_compute.add_argument("--input", required=True, help="Path to annotated CSV")

    sp_irr = sub.add_parser("compute-irr",
                            help="Two-annotator inter-rater reliability + judge agreement")
    sp_irr.add_argument("--annotator-a", required=True, help="Annotator A's filled blinded sheet")
    sp_irr.add_argument("--annotator-b", required=True, help="Annotator B's filled blinded sheet")
    sp_irr.add_argument("--master", default=None,
                        help="Master sheet (with judge labels + row_id) to also report "
                             "judge-vs-human and judge-vs-consensus")

    args = parser.parse_args()

    if args.command == "sample":
        if not args.input and not args.results_dir:
            parser.error("Provide either --input or --results-dir")
        sample_for_annotation(args.input, args.output, args.n, args.seed,
                              stratify=args.stratify, results_dir=args.results_dir,
                              processed=args.processed)
    elif args.command == "compute":
        compute_validation(args.input)
    elif args.command == "compute-irr":
        compute_irr(args.annotator_a, args.annotator_b, args.master)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
