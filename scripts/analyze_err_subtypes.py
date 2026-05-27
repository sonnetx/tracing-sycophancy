#!/usr/bin/env python3
"""Analyze Err. sub-type breakdown from classify_err_responses.py output.

Prints a breakdown table grouped by (model, checkpoint, domain, challenge_context)
and emits a LaTeX tabular snippet for the paper appendix.

Usage:
    python scripts/analyze_err_subtypes.py \\
        --input data/results/err_subclassified_medical.jsonl \\
                data/results/err_subclassified_computational.jsonl \\
        --highlight "llama-3.1-8b-instruct,instruct,medical,in_context" \\
                    "olmo-think,dpo,medical,in_context"
"""

import argparse
from collections import defaultdict

from src.utils import read_jsonl

SUBTYPES = ["apology_capitulation", "format_incoherence", "truncation_refusal", "other"]
SUBTYPE_LABELS = {
    "apology_capitulation": "Apol.-Cap.",
    "format_incoherence": "Format Inc.",
    "truncation_refusal": "Trunc./Ref.",
    "other": "Other",
}


def main():
    parser = argparse.ArgumentParser(description="Analyze Err. sub-type breakdown")
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more err_subclassified.jsonl files")
    parser.add_argument("--highlight", nargs="*", default=[],
                        help="Comma-separated keys to highlight: model,checkpoint,domain,context")
    parser.add_argument("--latex-out", default=None,
                        help="Optional path to write LaTeX table (prints to stdout if omitted)")
    args = parser.parse_args()

    highlight_keys = set()
    for h in args.highlight:
        parts = [p.strip() for p in h.split(",")]
        if len(parts) == 4:
            highlight_keys.add(tuple(parts))

    # counts[(model, checkpoint, domain, context)][subtype] = int
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in args.input:
        for rec in read_jsonl(path):
            key = (rec["model"], rec["checkpoint"], rec["domain"], rec["challenge_context"])
            counts[key][rec["err_subtype"]] += 1

    # Sort by domain, model, checkpoint, context
    sorted_keys = sorted(counts.keys(), key=lambda k: (k[2], k[0], k[1], k[3]))

    # --- Console table ---
    header = f"{'Model':<35} {'Ckpt':<12} {'Domain':<14} {'Ctx':<10} {'N':>5}  " + \
             "  ".join(f"{SUBTYPE_LABELS[s]:>10}" for s in SUBTYPES)
    print(header)
    print("-" * len(header))

    for key in sorted_keys:
        model, checkpoint, domain, context = key
        sub = counts[key]
        n_total = sum(sub.values())
        ctx_display = "IC" if context == "in_context" else "PE"
        flag = " **" if key in highlight_keys else ""
        row = (
            f"{model:<35} {checkpoint:<12} {domain:<14} {ctx_display:<10} {n_total:>5}  "
            + "  ".join(
                f"{sub.get(s, 0) / n_total:>9.1%}" if n_total > 0 else f"{'—':>10}"
                for s in SUBTYPES
            )
            + flag
        )
        print(row)

    # --- LaTeX table ---
    latex_lines = [
        r"\begin{table}[h]",
        r"  \caption{Err.\ sub-type breakdown by model, domain, and challenge context. "
        r"Sub-classification via GPT-4o judge shown challenge context (see \S\ref{sec:gen_track}). "
        r"Apol.-Cap.\ = apology-capitulation (behaviorally sycophantic); "
        r"Format Inc.\ = format incoherence (measurement artifact, predominant in base models); "
        r"Trunc./Ref.\ = truncation or refusal. "
        r"Computational rows with $N < 10$ are included for completeness; percentages are unreliable at that sample size.}",
        r"  \label{tab:err_subtype_breakdown}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llllr rrrr}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Ckpt} & \textbf{Domain} & \textbf{Ctx}"
        r" & $N$ & \textbf{Apol.-Cap.} & \textbf{Format Inc.} & \textbf{Trunc./Ref.} & \textbf{Other} \\",
        r"    \midrule",
    ]

    prev_domain = None
    for key in sorted_keys:
        model, checkpoint, domain, context = key
        sub = counts[key]
        n_total = sum(sub.values())
        ctx_display = r"IC" if context == "in_context" else r"PE"

        if prev_domain is not None and domain != prev_domain:
            latex_lines.append(r"    \midrule")
        prev_domain = domain

        pcts = []
        for s in SUBTYPES:
            if n_total > 0:
                pct = sub.get(s, 0) / n_total * 100
                pcts.append(f"{pct:.0f}\\%")
            else:
                pcts.append("---")

        row = (
            f"    {model} & {checkpoint} & {domain} & {ctx_display}"
            f" & {n_total} & " + " & ".join(pcts) + r" \\"
        )
        if key in highlight_keys:
            row = row + r"  % <-- cited in Discussion"
        latex_lines.append(row)

    latex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    latex_output = "\n".join(latex_lines)
    if args.latex_out:
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(latex_output)
        print(f"\nLaTeX table written to {args.latex_out}")
    else:
        print("\n--- LaTeX ---")
        print(latex_output)


if __name__ == "__main__":
    main()
