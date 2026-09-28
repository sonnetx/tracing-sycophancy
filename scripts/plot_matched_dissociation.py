#!/usr/bin/env python3
"""Matched-subset dissociation figure (paper Figure: behavior vs. candidate score).

Reads <experiment-dir>/<domain>/analysis/matched_summaries.json (behavioral
flip rates) and matched_lp_summaries.json (preemptive dLogOdds), as written by
scripts/analyze.py --experiment-dir, and
writes one full-width figure, rows = domain, columns = measure. Each point is the
mean per-item change from the base checkpoint on items both checkpoints answer
correctly, with a 95% percentile bootstrap interval over items. Base is zero by
construction. Llama 3.1 has only base and final checkpoints.
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
PIPES = [("Instruct", "OLMo 3 Instruct"), ("Think", "OLMo 3 Think"),
         ("Tulu 3", "Tulu 3"), ("Llama 3.1", "Llama 3.1 Instruct")]
DODGE = {"Instruct": -0.15, "Think": -0.05, "Tulu 3": 0.05, "Llama 3.1": 0.15}
STAGES = ["Base", "SFT", "DPO", "Final"]
REF_GRAY = "#9a9a96"

plt.rcParams.update({
    # Printed at 1:1 on a 5.5 in text width, so these sizes sit near the 10 pt body text.
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def _positions(records: list) -> list:
    """Stage x-positions; the last record is the final checkpoint."""
    return [0, len(STAGES) - 1] if len(records) == 2 else list(range(len(records)))


def behavioral_series(records: list):
    xs, ys, lo, hi = [], [], [], []
    for x, r in zip(_positions(records), records):
        p = r["base_vs_stage_paired"]
        m = 0.0 if r["stage"] == "Base" else p["mean_d"]
        xs.append(x); ys.append(m)
        lo.append(0.0 if r["stage"] == "Base" else m - p["bootstrap_ci_low"])
        hi.append(0.0 if r["stage"] == "Base" else p["bootstrap_ci_high"] - m)
    return xs, ys, lo, hi


def receptivity_series(records: list):
    xs, ys, lo, hi = [], [], [], []
    for x, r in zip(_positions(records), records):
        p = r.get("paired", {})
        m = 0.0 if r["stage"] == "Base" else p["mean_delta_delta_L"]
        xs.append(x); ys.append(m)
        lo.append(0.0 if r["stage"] == "Base" else m - p["bootstrap_ci_low"])
        hi.append(0.0 if r["stage"] == "Base" else p["bootstrap_ci_high"] - m)
    return xs, ys, lo, hi


def draw(ax, series_fn, summaries: dict) -> None:
    ax.axhline(0, color=REF_GRAY, linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    for key, _ in PIPES:
        xs, ys, lo, hi = series_fn(summaries[key])
        xs = [x + DODGE[key] for x in xs]
        color = PIPELINE_COLORS[key]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linestyle=PIPELINE_LINESTYLES[key],
                    linewidth=1.4, elinewidth=0.9, capsize=0, marker=PIPELINE_MARKERS[key],
                    markersize=5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES)
    ax.set_xlim(-0.4, len(STAGES) - 0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment-dir", default=os.path.join(ROOT, "data", "results", "exp1"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "paper", "figures"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8), sharex=True, sharey="col")
    for row, (dom_dir, dom_label) in enumerate(DOMAINS):
        analysis_dir = os.path.join(args.experiment_dir, dom_dir, "analysis")
        with open(os.path.join(analysis_dir, "matched_summaries.json")) as f:
            beh = json.load(f)
        with open(os.path.join(analysis_dir, "matched_lp_summaries.json")) as f:
            rec = json.load(f)
        draw(axes[row, 0], behavioral_series, beh)
        draw(axes[row, 1], receptivity_series, rec)
        axes[row, 0].set_ylabel(f"{dom_label}\nchange from base")

    axes[0, 0].set_title("Flip rate (resistance)")
    axes[0, 1].set_title(r"Preemptive $\Delta$LogOdds (receptivity)")

    handles = [Line2D([], [], color=PIPELINE_COLORS[k], linestyle=PIPELINE_LINESTYLES[k],
                      linewidth=1.4, marker=PIPELINE_MARKERS[k], markersize=5,
                      markeredgecolor="white", markeredgewidth=0.7, label=label)
               for k, label in PIPES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0, w_pad=1.6)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"matched_dissociation.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
