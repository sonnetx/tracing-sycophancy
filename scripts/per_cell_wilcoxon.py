#!/usr/bin/env python3
"""Per-cell Wilcoxon signed-rank tests for ΔLogOdds.

For every (model, challenge type, context) cell, tests the per-item
delta_log_odds distribution against a zero-shift null (two-sided,
zero_method="wilcox"), backing the per-cell claims in Finding 1 with
significance results (reviewer request: the main text called cells
"near zero" / "uniformly negative" without per-cell tests).

Outputs a JSON summary and a LaTeX-ready significance map keyed the same
way as the type x context table (tab:dlo_type_ctx_comp): for each cell,
mean, median, n, Wilcoxon p, and a marker (* p<0.05 after Holm
correction within each model x domain family of 8 cells).

Usage:
    python scripts/per_cell_wilcoxon.py \
        --results-dir data/results/exp1/computational \
        --output data/results/exp1/computational/analysis/per_cell_wilcoxon.json

Run once per domain directory. Requires the per-model logprob_scores.jsonl
files (on Sherlock under the exp1 results tree if not synced locally).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from src.utils import read_jsonl

CHALLENGE_TYPES = ["simple", "ethos", "justification", "citation"]
CONTEXTS = ["in_context", "preemptive"]


def find_logprob_files(results_dir: str) -> dict[str, str]:
    models = {}
    for entry in sorted(os.listdir(results_dir)):
        lp_path = os.path.join(results_dir, entry, "logprob_scores.jsonl")
        if os.path.isfile(lp_path):
            models[entry] = lp_path
    return models


def collect_cells(lp_path: str) -> dict[tuple[str, str], list[float]]:
    """Per-item delta_log_odds grouped by (challenge_type, challenge_context)."""
    cells = defaultdict(list)
    for row in read_jsonl(lp_path):
        for ch in row.get("challenge_scores", []):
            ctype = ch.get("challenge_type")
            cctx = ch.get("challenge_context")
            dlo = ch.get("delta_log_odds")
            if ctype in CHALLENGE_TYPES and cctx in CONTEXTS and dlo is not None:
                cells[(ctype, cctx)].append(float(dlo))
    return cells


def holm_correct(pvals: dict) -> dict:
    """Holm-Bonferroni within one model's family of cells. Keys with p=None pass through."""
    testable = {k: v for k, v in pvals.items() if v is not None}
    m = len(testable)
    ordered = sorted(testable.items(), key=lambda kv: kv[1])
    adjusted, running_max = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    for k in pvals:
        adjusted.setdefault(k, None)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="Domain results dir containing per-model subdirs")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <results-dir>/analysis/per_cell_wilcoxon.json)")
    args = parser.parse_args()

    lp_files = find_logprob_files(args.results_dir)
    if not lp_files:
        raise SystemExit(f"No logprob_scores.jsonl found under {args.results_dir}")

    results = {}
    for model, lp_path in lp_files.items():
        cells = collect_cells(lp_path)
        pvals, summary = {}, {}
        for ctype in CHALLENGE_TYPES:
            for cctx in CONTEXTS:
                key = f"{ctype}/{cctx}"
                vals = np.asarray(cells.get((ctype, cctx), []), dtype=float)
                entry = {"n": int(vals.size), "mean": None, "median": None,
                         "wilcoxon_p": None}
                if vals.size:
                    entry["mean"] = float(np.mean(vals))
                    entry["median"] = float(np.median(vals))
                    nonzero = vals[vals != 0]
                    if nonzero.size >= 10:
                        _, p = wilcoxon(vals, alternative="two-sided",
                                        zero_method="wilcox")
                        entry["wilcoxon_p"] = float(p)
                pvals[key] = entry["wilcoxon_p"]
                summary[key] = entry
        adjusted = holm_correct(pvals)
        for key, entry in summary.items():
            entry["holm_p"] = adjusted[key]
            entry["significant"] = (adjusted[key] is not None and adjusted[key] < 0.05)
        results[model] = summary

    out_path = args.output or os.path.join(args.results_dir, "analysis",
                                           "per_cell_wilcoxon.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    # Console view: one line per model, star = Holm-significant at 0.05
    header = ["model"] + [f"{t[:4]}/{c[:2].upper()}" for t in CHALLENGE_TYPES for c in CONTEXTS]
    print("  ".join(f"{h:>14}" for h in header))
    for model, summary in results.items():
        row = [model[:14]]
        for ctype in CHALLENGE_TYPES:
            for cctx in CONTEXTS:
                e = summary[f"{ctype}/{cctx}"]
                if e["mean"] is None:
                    row.append("--")
                else:
                    star = "*" if e["significant"] else " "
                    row.append(f"{e['mean']:+.3f}{star}")
        print("  ".join(f"{c:>14}" for c in row))


if __name__ == "__main__":
    main()
