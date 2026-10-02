#!/usr/bin/env python3
"""Plot per-cell draw outcomes from the sampling experiment.

For each (pipeline, domain) at one temperature and context, every
(question, challenge) cell holds five sampled draws, each judged correct,
incorrect, or erroneous. The x-axis counts the incorrect draws in a cell out of
the draws actually made (no rescaling), and each bar is split by how many of
the remaining draws are erroneous, so erroneous draws stay visible instead of
being dropped from the denominator.

Usage:
    PYTHONPATH=. python3 scripts/plot_sampling_outcomes.py \\
        --sampling-dir data/results/exp1_sampling \\
        --output-dir paper/figures
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-safe: cluster login/compute nodes have no display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PIPELINES = [
    ("OLMo Instruct", "olmo3-7b-instruct"),
    ("Tulu 3",         "tulu3-llama31-8b"),
    ("Llama 3.1 Instruct", "llama31-8b-instruct"),
]
DOMAINS = [("computational", "Computational"), ("medical_advice", "Medical")]
NON_SIMPLE_TYPES = {"ethos", "justification", "citation"}
N_DRAWS = 5

# Erroneous-draw buckets, light to dark (one-hue ordinal ramp, validated for a
# white page). Ink and gridline colors keep text out of the data colors.
ERR_BUCKETS = [("No erroneous draws", 0, 0, "#86b6ef"), ("1-2 erroneous", 1, 2, "#3987e5"),
               ("3-5 erroneous", 3, 5, "#184f95")]
INK, INK_SECONDARY, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load_cell_outcomes(path: str, temperature: float, context: str) -> dict:
    """Return {(qid, challenge_id): Counter of judge labels} for non-simple draws at T."""
    cells = defaultdict(Counter)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if abs(r.get("temperature", -1) - temperature) > 1e-6:
                continue
            if r.get("challenge_context") != context:
                continue
            if r.get("challenge_type") not in NON_SIMPLE_TYPES:
                continue
            cells[(r["question_id"], r["challenge_id"])][r.get("factual_accuracy")] += 1
    return dict(cells)


def plot_one_domain(sampling_dir: str, domain: str, output_path: str,
                    temperature: float = 1.0, context: str = "preemptive") -> None:
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": AXIS,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    # Sized at the printed width (NeurIPS \linewidth = 5.5in) so 7pt stays 7pt.
    fig, axes = plt.subplots(1, len(PIPELINES), figsize=(5.5, 1.9), sharey=True)
    for ax, (label, model_key) in zip(axes, PIPELINES):
        ax.set_title(label, fontsize=7.5, color=INK, pad=3)
        p = os.path.join(sampling_dir, domain, model_key, "sampling_evaluated.jsonl")
        cells = load_cell_outcomes(p, temperature, context)
        if not cells:
            raise ValueError(f"No matching samples for {label} {domain}")
        short = sum(sum(c.values()) != N_DRAWS for c in cells.values())
        if short:
            print(f"  {label} {domain}: {short}/{len(cells)} cells have fewer than {N_DRAWS} draws")
        # grid[k][bucket] = number of cells with k incorrect draws in that erroneous bucket
        grid = defaultdict(Counter)
        for c in cells.values():
            err = c["erroneous"]
            bucket = next(name for name, lo, hi, _ in ERR_BUCKETS if lo <= err <= hi)
            grid[c["incorrect"]][bucket] += 1
        n = len(cells)
        bottom = [0.0] * (N_DRAWS + 1)
        for name, _, _, color in ERR_BUCKETS:
            heights = [100 * grid[k][name] / n for k in range(N_DRAWS + 1)]
            # White edges give the surface gap between stacked segments.
            ax.bar(range(N_DRAWS + 1), heights, bottom=bottom, width=0.72,
                   color=color, edgecolor="white", linewidth=0.8, zorder=2)
            bottom = [b + h for b, h in zip(bottom, heights)]
        mean_inc = sum(c["incorrect"] for c in cells.values()) / n
        mean_err = sum(c["erroneous"] for c in cells.values()) / n
        ax.text(0.97, 0.97, f"n = {n}\nincorrect {mean_inc:.2f}\nerroneous {mean_err:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                color=INK_SECONDARY, linespacing=1.25)
        ax.set_xticks(range(N_DRAWS + 1))
        ax.set_xlabel(f"Incorrect draws (of {N_DRAWS})", color=INK_SECONDARY, labelpad=2)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    axes[0].set_ylabel("Share of cells (%)", color=INK_SECONDARY, labelpad=2)
    axes[0].set_ylim(0, 100)
    # Legend gets its own row above the panel titles.
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.88))
    fig.legend(handles=[Patch(facecolor=c, label=name) for name, _, _, c in ERR_BUCKETS],
               loc="upper center", ncol=len(ERR_BUCKETS), bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=6.5, handlelength=1.0, handleheight=0.8,
               columnspacing=1.5, labelcolor=INK_SECONDARY)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", default="data/results/exp1_sampling")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context", default="preemptive")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # New file names, so the old rescaled figures stay in place until the caption changes.
    short_names = {"computational": "comp", "medical_advice": "med"}
    for dkey, dlabel in DOMAINS:
        short = short_names.get(dkey, dkey[:4])
        out = os.path.join(args.output_dir, f"{short}_sampling_outcomes.png")
        plot_one_domain(args.sampling_dir, dkey, out,
                         temperature=args.temperature, context=args.context)


if __name__ == "__main__":
    main()
