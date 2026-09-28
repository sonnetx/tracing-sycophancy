#!/usr/bin/env python3
"""In-context vs preemptive dLogOdds across training (paper Figure 3).

Reads <experiment-dir>/<domain>/analysis/logprob_summaries.json, as written by
scripts/analyze.py --experiment-dir, and writes one full-width figure with four
panels (domain x context). Each point is the mean per-item dLogOdds over the
ethos, justification, and citation challenges. Llama 3.1 has only base and final
checkpoints.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.analysis.plots import PIPELINE_COLORS, PIPELINE_LINESTYLES, PIPELINE_MARKERS  # noqa: E402

DOMAINS = [("computational", "Computational"), ("medical_advice", "Medical")]
CONTEXTS = [("in_context", "in-context"), ("preemptive", "preemptive")]
PIPES = [
    ("Instruct", "OLMo 3 Instruct", ["olmo3-7b-base", "olmo3-7b-instruct-sft", "olmo3-7b-instruct-dpo", "olmo3-7b-instruct"]),
    ("Think", "OLMo 3 Think", ["olmo3-7b-base", "olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think"]),
    ("Tulu 3", "Tulu 3", ["llama31-8b-base", "tulu3-llama31-8b-sft", "tulu3-llama31-8b-dpo", "tulu3-llama31-8b"]),
    ("Llama 3.1", "Llama 3.1 Instruct", ["llama31-8b-base", None, None, "llama31-8b-instruct"]),
]
STAGES = ["Base", "SFT", "DPO", "Final"]
REF_GRAY = "#9a9a96"

plt.rcParams.update({
    # Printed at 1:1 on a 5.5 in text width.
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment-dir", default=os.path.join(ROOT, "data", "results", "exp1"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "paper", "figures"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.05), sharey=True)
    for d, (dom_dir, dom_label) in enumerate(DOMAINS):
        with open(os.path.join(args.experiment_dir, dom_dir, "analysis", "logprob_summaries.json")) as f:
            summ = json.load(f)
        for c, (ctx, ctx_label) in enumerate(CONTEXTS):
            ax = axes[2 * d + c]
            ax.axhline(0, color=REF_GRAY, linewidth=0.8, zorder=1)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
            for key, _, models in PIPES:
                pts = [(x, summ[m]["challenges"][ctx]["mean_delta_log_odds"])
                       for x, m in enumerate(models) if m is not None]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PIPELINE_COLORS[key],
                        linestyle=PIPELINE_LINESTYLES[key], linewidth=1.3,
                        marker=PIPELINE_MARKERS[key], markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(STAGES, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.3, len(STAGES) - 0.7)
            ax.set_title(f"{dom_label}\n{ctx_label}", linespacing=1.1)
    axes[0].set_ylabel(r"$\Delta$LogOdds")

    handles = [Line2D([], [], color=PIPELINE_COLORS[k], linestyle=PIPELINE_LINESTYLES[k],
                      linewidth=1.3, marker=PIPELINE_MARKERS[k], markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.6, label=label)
               for k, label, _ in PIPES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"ic_pe_trajectories.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
