#!/usr/bin/env python3

import argparse
import json
import os
from typing import Iterable

from src.analysis.stats import load_results_as_dataframe


MODEL_ORDER = [
    "olmo3-7b-base",
    "olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think",
    "olmo3-7b-instruct-sft", "olmo3-7b-instruct-dpo", "olmo3-7b-instruct",
    "llama31-8b-base", "llama31-8b-instruct",
    "tulu3-llama31-8b-sft", "tulu3-llama31-8b-dpo", "tulu3-llama31-8b",
]
DISPLAY = {
    "olmo3-7b-base":          "OLMo 3 Base",
    "olmo3-7b-think-sft":     "OLMo Think SFT",
    "olmo3-7b-think-dpo":     "OLMo Think DPO",
    "olmo3-7b-think":         "OLMo Think",
    "olmo3-7b-instruct-sft":  "OLMo Instruct SFT",
    "olmo3-7b-instruct-dpo":  "OLMo Instruct DPO",
    "olmo3-7b-instruct":      "OLMo Instruct",
    "llama31-8b-base":        "Llama 3.1 Base",
    "llama31-8b-instruct":    "Llama 3.1 Instruct",
    "tulu3-llama31-8b-sft":   "Tulu 3 SFT",
    "tulu3-llama31-8b-dpo":   "Tulu 3 DPO",
    "tulu3-llama31-8b":       "Tulu 3",
}

WRONG_TYPES = ("simple", "ethos", "justification", "citation")


def compute_context_breakdown(df) -> dict:
    """For each context in {in_context, preemptive}, compute Regr, Ctrl, Net."""
    initial = df[df["response_type"] == "initial"]
    challenges = df[df["response_type"] == "challenge"]
    initially_correct = set(
        initial[initial["factual_accuracy"] == "correct"]["question_id"]
    )
    if len(initially_correct) == 0 or len(challenges) == 0:
        return {}

    out = {}
    for ctx in ("in_context", "preemptive"):
        ctx_ch = challenges[challenges["challenge_context"] == ctx]
        if len(ctx_ch) == 0:
            continue

        wrong_all = ctx_ch[
            ctx_ch["challenge_type"].isin(WRONG_TYPES) &
            ctx_ch["question_id"].isin(initially_correct)
        ]
        wrong = wrong_all[wrong_all["factual_accuracy"].isin(["correct", "incorrect"])]
        regr_count = int((wrong["factual_accuracy"] == "incorrect").sum())
        regr_total = int(len(wrong))
        regr_total_raw = int(len(wrong_all))
        regr_rate = regr_count / regr_total if regr_total > 0 else 0.0
        regr_err = int((wrong_all["factual_accuracy"] == "erroneous").sum())
        regr_err_rate = regr_err / regr_total_raw if regr_total_raw > 0 else 0.0

        # Correct-answer control (same context, coherent-only denominator)
        ctrl_all = ctx_ch[
            (ctx_ch["challenge_type"] == "correct") &
            ctx_ch["question_id"].isin(initially_correct)
        ]
        ctrl = ctrl_all[ctrl_all["factual_accuracy"].isin(["correct", "incorrect"])]
        ctrl_count = int((ctrl["factual_accuracy"] == "incorrect").sum())
        ctrl_total = int(len(ctrl))
        ctrl_total_raw = int(len(ctrl_all))
        ctrl_rate = ctrl_count / ctrl_total if ctrl_total > 0 else 0.0
        ctrl_err = int((ctrl_all["factual_accuracy"] == "erroneous").sum())
        ctrl_err_rate = ctrl_err / ctrl_total_raw if ctrl_total_raw > 0 else 0.0

        out[ctx] = {
            "regressive_rate": regr_rate,
            "regressive_count": regr_count,
            "regressive_total": regr_total,
            "regressive_total_raw": regr_total_raw,
            "regressive_err_rate": regr_err_rate,
            "control_rate": ctrl_rate,
            "control_count": ctrl_count,
            "control_total": ctrl_total,
            "control_total_raw": ctrl_total_raw,
            "control_err_rate": ctrl_err_rate,
            "net_rate": regr_rate - ctrl_rate,
        }

    return out


def fmt_pct(x):
    if x is None:
        return "  .  "
    return f"{x * 100:5.1f}"


def print_dataset(dataset_dir: str, dataset_label: str, latex: bool) -> None:
    rows = []
    for model_key in MODEL_ORDER:
        ev_path = os.path.join(dataset_dir, model_key, "evaluated.jsonl")
        if not os.path.isfile(ev_path):
            continue
        df = load_results_as_dataframe(ev_path)
        breakdown = compute_context_breakdown(df)
        if not breakdown:
            continue
        rows.append((model_key, breakdown))

    if not rows:
        return

    if latex:
        print(f"\n% === {dataset_label} ===")
        print("\\begin{tabular}{l cccc cccc}")
        print("  \\toprule")
        print("  & \\multicolumn{4}{c}{\\textbf{In-context}} "
              "& \\multicolumn{4}{c}{\\textbf{Preemptive}} \\\\")
        print("  \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
        print("  \\textbf{Model} & Regr. & Ctrl. & Net & Err. & "
              "Regr. & Ctrl. & Net & Err. \\\\")
        print("  \\midrule")
        for model_key, b in rows:
            ic = b.get("in_context", {})
            pe = b.get("preemptive", {})
            print(f"  {DISPLAY[model_key]:20s} & "
                  f"{fmt_pct(ic.get('regressive_rate')):>5s} & "
                  f"{fmt_pct(ic.get('control_rate')):>5s} & "
                  f"{fmt_pct(ic.get('net_rate')):>5s} & "
                  f"{fmt_pct(ic.get('regressive_err_rate')):>5s} & "
                  f"{fmt_pct(pe.get('regressive_rate')):>5s} & "
                  f"{fmt_pct(pe.get('control_rate')):>5s} & "
                  f"{fmt_pct(pe.get('net_rate')):>5s} & "
                  f"{fmt_pct(pe.get('regressive_err_rate')):>5s} \\\\")
        print("  \\bottomrule")
        print("\\end{tabular}")
    else:
        print(f"\n=== {dataset_label} ===")
        print(f"{'Model':22s} | {'IC Regr':>7s} {'Ctrl':>6s} {'Net':>6s} {'Err':>6s} | "
              f"{'PE Regr':>7s} {'Ctrl':>6s} {'Net':>6s} {'Err':>6s}  (percent)")
        print("-" * 100)
        for model_key, b in rows:
            ic = b.get("in_context", {})
            pe = b.get("preemptive", {})
            print(f"{DISPLAY[model_key]:22s} | "
                  f"{fmt_pct(ic.get('regressive_rate')):>7s} "
                  f"{fmt_pct(ic.get('control_rate')):>6s} "
                  f"{fmt_pct(ic.get('net_rate')):>6s} "
                  f"{fmt_pct(ic.get('regressive_err_rate')):>6s} | "
                  f"{fmt_pct(pe.get('regressive_rate')):>7s} "
                  f"{fmt_pct(pe.get('control_rate')):>6s} "
                  f"{fmt_pct(pe.get('net_rate')):>6s} "
                  f"{fmt_pct(pe.get('regressive_err_rate')):>6s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True,
                        help="e.g. data/results/exp1")
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX tabular rather than plain text")
    args = parser.parse_args()

    for entry in sorted(os.listdir(args.experiment_dir)):
        ds_dir = os.path.join(args.experiment_dir, entry)
        if not os.path.isdir(ds_dir) or entry in ("cross_dataset", "paper_figures"):
            continue
        print_dataset(ds_dir, entry, args.latex)


if __name__ == "__main__":
    main()
