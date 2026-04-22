#!/usr/bin/env python3
"""Breakdown of ΔLogOdds by challenge context and type.

Reads the per-dataset logprob_summaries.json produced by scripts/analyze.py
and prints two tables:
  1. Per (model, context) with context in {in_context, preemptive, overall}
  2. Per (model, type) with type in {simple, ethos, justification, citation}

Usage (requires python3 — f-strings and UTF-8 source):
    python3 scripts/logprob_breakdown.py \\
        --summaries data/results/exp1/computational/analysis/logprob_summaries.json
    python3 scripts/logprob_breakdown.py \\
        --summaries data/results/exp1/computational/analysis/logprob_summaries.json \\
        --latex
"""

import argparse
import json
import os


MODEL_ORDER = [
    "olmo3-7b-base",
    "olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think",
    "olmo3-7b-instruct-sft", "olmo3-7b-instruct-dpo", "olmo3-7b-instruct",
    "llama31-8b-base", "llama31-8b-instruct",
    "tulu3-llama31-8b-sft", "tulu3-llama31-8b-dpo", "tulu3-llama31-8b",
]
DISPLAY = {
    "olmo3-7b-base": "OLMo 3 Base",
    "olmo3-7b-think-sft": "OLMo Think SFT",
    "olmo3-7b-think-dpo": "OLMo Think DPO",
    "olmo3-7b-think":     "OLMo Think",
    "olmo3-7b-instruct-sft": "OLMo Instruct SFT",
    "olmo3-7b-instruct-dpo": "OLMo Instruct DPO",
    "olmo3-7b-instruct":     "OLMo Instruct",
    "llama31-8b-base":     "Llama 3.1 Base",
    "llama31-8b-instruct": "Llama 3.1 Instruct",
    "tulu3-llama31-8b-sft": "Tulu 3 SFT",
    "tulu3-llama31-8b-dpo": "Tulu 3 DPO",
    "tulu3-llama31-8b":     "Tulu 3",
}


def fmt(x):
    if x is None:
        return "   .  "
    return f"{x:+.3f}"


def _pick(stats: dict, which: str):
    return stats.get(which)


def print_context_table(summaries: dict, latex: bool) -> None:
    ctx_order = ["in_context", "preemptive", "overall"]
    rows = []
    for mk in MODEL_ORDER:
        if mk not in summaries:
            continue
        ch = summaries[mk].get("challenges", {})
        row = {"model": DISPLAY[mk]}
        for ctx in ctx_order:
            stats = ch.get(ctx) or ch.get("overall") if ctx == "overall" else ch.get(ctx)
            stats = ch.get(ctx) if ctx != "overall" else ch.get("overall")
            if stats:
                row[f"{ctx}_mean"]   = _pick(stats, "mean_delta_log_odds")
                row[f"{ctx}_median"] = _pick(stats, "median_delta_log_odds")
            else:
                row[f"{ctx}_mean"]   = None
                row[f"{ctx}_median"] = None
        rows.append(row)

    if latex:
        print("% --- ΔLogOdds by challenge context ---")
        print("\\begin{tabular}{l cc cc cc}")
        print("  \\toprule")
        print("  & \\multicolumn{2}{c}{\\textbf{In-context}} "
              "& \\multicolumn{2}{c}{\\textbf{Preemptive}} "
              "& \\multicolumn{2}{c}{\\textbf{Overall}} \\\\")
        print("  \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
        print("  \\textbf{Model} & Mean & Median & Mean & Median & Mean & Median \\\\")
        print("  \\midrule")
        for r in rows:
            print(f"  {r['model']:20s} & "
                  f"{fmt(r['in_context_mean']):>7s} & {fmt(r['in_context_median']):>7s} & "
                  f"{fmt(r['preemptive_mean']):>7s} & {fmt(r['preemptive_median']):>7s} & "
                  f"{fmt(r['overall_mean']):>7s} & {fmt(r['overall_median']):>7s} \\\\")
        print("  \\bottomrule")
        print("\\end{tabular}")
    else:
        print(f"\n{'Model':24s} | {'in-ctx mean':>11s} {'med':>7s} | "
              f"{'preempt mean':>12s} {'med':>7s} | {'overall mean':>12s} {'med':>7s}")
        print("-" * 100)
        for r in rows:
            print(f"{r['model']:24s} | "
                  f"{fmt(r['in_context_mean']):>11s} {fmt(r['in_context_median']):>7s} | "
                  f"{fmt(r['preemptive_mean']):>12s} {fmt(r['preemptive_median']):>7s} | "
                  f"{fmt(r['overall_mean']):>12s} {fmt(r['overall_median']):>7s}")


def print_type_table(summaries: dict, latex: bool) -> None:
    types = ["simple", "ethos", "justification", "citation"]
    rows = []
    for mk in MODEL_ORDER:
        if mk not in summaries:
            continue
        ch = summaries[mk].get("challenges", {})
        row = {"model": DISPLAY[mk]}
        for t in types:
            stats = ch.get(f"type_{t}")
            row[f"{t}_mean"]   = _pick(stats, "mean_delta_log_odds") if stats else None
            row[f"{t}_median"] = _pick(stats, "median_delta_log_odds") if stats else None
        rows.append(row)

    if latex:
        print("\n% --- ΔLogOdds mean by challenge type ---")
        print("\\begin{tabular}{l cccc}")
        print("  \\toprule")
        print("  \\textbf{Model} & \\textbf{Simple} & \\textbf{Ethos} "
              "& \\textbf{Justification} & \\textbf{Citation} \\\\")
        print("  \\midrule")
        for r in rows:
            print(f"  {r['model']:20s} & "
                  f"{fmt(r['simple_mean']):>7s} & {fmt(r['ethos_mean']):>7s} & "
                  f"{fmt(r['justification_mean']):>7s} & {fmt(r['citation_mean']):>7s} \\\\")
        print("  \\bottomrule")
        print("\\end{tabular}")
    else:
        print(f"\n{'Model':24s} | {'simple':>7s} | {'ethos':>7s} | {'justif.':>7s} | {'citation':>8s}  (mean ΔLO)")
        print("-" * 76)
        for r in rows:
            print(f"{r['model']:24s} | "
                  f"{fmt(r['simple_mean']):>7s} | {fmt(r['ethos_mean']):>7s} | "
                  f"{fmt(r['justification_mean']):>7s} | {fmt(r['citation_mean']):>8s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True,
                        help="Path to logprob_summaries.json")
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX tabular instead of plain text")
    args = parser.parse_args()

    with open(args.summaries, "r") as f:
        summaries = json.load(f)

    label = os.path.dirname(args.summaries)
    print(f"=== {label} ===")
    print_context_table(summaries, args.latex)
    print_type_table(summaries, args.latex)


if __name__ == "__main__":
    main()
