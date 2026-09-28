#!/usr/bin/env python3
"""Judge-vs-human-consensus metrics for the n=150 validation samples.

Joins judge labels (extracted from evaluated.jsonl by row_id) to the
multi-annotator consensus CSVs and, on rows with a majority consensus label,
reports per domain: judge accuracy with a Beta(matches+1, mismatches+1)
posterior mean and 95% credible interval, 3-class Cohen's kappa, per-class
precision/recall, and the confusion matrix.

Usage:
    python scripts/compute_judge_vs_consensus.py \
        --judge-labels judge_labels.csv \
        --consensus computational_n150_three_annotator_consensus.csv \
                    medical_n150_four_annotator_consensus.csv \
        --output judge_vs_consensus.json
"""

import argparse
import csv
import json
from collections import Counter

from scipy.stats import beta

LABELS = ["correct", "incorrect", "erroneous"]


def cohens_kappa(pairs):
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-labels", required=True)
    parser.add_argument("--consensus", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    judge = {}
    with open(args.judge_labels) as f:
        for r in csv.DictReader(f):
            judge[r["row_id"]] = r["judge_factual_accuracy"].strip().lower()

    report = {}
    for path in args.consensus:
        rows = list(csv.DictReader(open(path)))
        domain = rows[0]["domain"]
        pairs, missing_judge = [], 0
        for r in rows:
            status = r.get("consensus_status", "").strip().lower()
            consensus = r.get("majority_consensus_factual_accuracy", "").strip().lower()
            if consensus not in LABELS or "no majority" in status or "no_majority" in status:
                continue
            j = judge.get(r["row_id"])
            if j not in LABELS:
                missing_judge += 1
                continue
            pairs.append((j, consensus))

        n = len(pairs)
        matches = sum(1 for j, c in pairs if j == c)
        a, b = matches + 1, (n - matches) + 1
        entry = {
            "n_majority_rows_scored": n,
            "rows_missing_judge_label": missing_judge,
            "judge_accuracy": matches / n,
            "beta_posterior_mean": a / (a + b),
            "beta_95_ci": [float(beta.ppf(0.025, a, b)), float(beta.ppf(0.975, a, b))],
            "cohens_kappa": cohens_kappa(pairs),
            "per_class": {},
            "confusion_judge_rows_consensus_cols": {},
        }
        for l in LABELS:
            tp = sum(1 for j, c in pairs if j == l and c == l)
            pred = sum(1 for j, _ in pairs if j == l)
            actual = sum(1 for _, c in pairs if c == l)
            entry["per_class"][l] = {
                "precision": tp / pred if pred else None,
                "recall": tp / actual if actual else None,
                "n_consensus": actual,
            }
        for j in LABELS:
            entry["confusion_judge_rows_consensus_cols"][j] = {
                c: sum(1 for jj, cc in pairs if jj == j and cc == c) for c in LABELS}
        report[domain] = entry

    out = args.output or "judge_vs_consensus.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}\n")
    for domain, e in report.items():
        print(f"== {domain}: n={e['n_majority_rows_scored']}, "
              f"acc={e['judge_accuracy']:.3f} "
              f"(Beta mean {e['beta_posterior_mean']:.3f}, "
              f"95% CI [{e['beta_95_ci'][0]:.3f}, {e['beta_95_ci'][1]:.3f}]), "
              f"kappa={e['cohens_kappa']:.3f}")
        for l, d in e["per_class"].items():
            p = f"{d['precision']:.2f}" if d["precision"] is not None else "NA"
            r = f"{d['recall']:.2f}" if d["recall"] is not None else "NA"
            print(f"   {l:>10}: precision {p}  recall {r}  (n={d['n_consensus']})")


if __name__ == "__main__":
    main()
