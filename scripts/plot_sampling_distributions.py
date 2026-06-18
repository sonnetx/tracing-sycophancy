#!/usr/bin/env python3
"""Plot per-item flip distributions from the sampling experiment.

For each (pipeline, domain, context, temperature), groups the 5 samples per
(question, challenge) into a per-item flip count (0-5) and plots the
distribution across items. Sampling-robust pipelines show U-shape at zero;
sampling-brittle pipelines spread across the 0-5 range.

Usage:
    PYTHONPATH=. python3 scripts/plot_sampling_distributions.py \\
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
import numpy as np

PIPELINES = [
    ("OLMo Instruct", "olmo3-7b-instruct"),
    ("Tulu 3",         "tulu3-llama31-8b"),
    ("Llama 3.1 Instruct", "llama31-8b-instruct"),
    ("OLMo Think",     "olmo3-7b-think"),
]
PIPELINE_COLORS = {
    "OLMo Instruct":        "#1f77b4",
    "Tulu 3":               "#2ca02c",
    "Llama 3.1 Instruct":   "#ff7f0e",
    "OLMo Think":           "#d62728",
}
DOMAINS = [("computational", "Computational"), ("medical_advice", "Medical")]
NON_SIMPLE_TYPES = {"ethos", "justification", "citation"}


def load_flip_counts(path: str, temperature: float, context: str) -> dict:
    """Return {(qid, challenge_id): n_flips} for PE non-simple samples at T."""
    per_item = defaultdict(lambda: [0, 0])  # [n_flip, n_coherent]
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if abs(r.get("temperature", -1) - temperature) > 1e-6:
                continue
            if r.get("challenge_context") != context:
                continue
            if r.get("challenge_type") not in NON_SIMPLE_TYPES:
                continue
            fa = r.get("factual_accuracy")
            if fa not in ("correct", "incorrect"):
                continue  # erroneous excluded (coherent-only denominator)
            key = (r["question_id"], r["challenge_id"])
            per_item[key][1] += 1
            if fa == "incorrect":
                per_item[key][0] += 1
    # Return fraction of flips per item (n_flip / n_coherent) as well as raw
    return {k: (v[0], v[1]) for k, v in per_item.items() if v[1] > 0}


def plot_one_domain(sampling_dir: str, domain: str, output_path: str,
                    temperature: float = 1.0, context: str = "preemptive") -> None:
    fig, axes = plt.subplots(1, len(PIPELINES), figsize=(4 * len(PIPELINES), 3.8),
                              sharey=True)
    for ax, (label, model_key) in zip(axes, PIPELINES):
        p = os.path.join(sampling_dir, domain, model_key, "sampling_evaluated.jsonl")
        counts = load_flip_counts(p, temperature, context)
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(label, fontsize=11, fontweight="bold")
            continue
        # Bin per-item flip counts from 0 to 5
        bin_counts = Counter()
        for (n_flip, n_coh) in counts.values():
            # Scale up to 5 if coherent count < 5 (rare)
            if n_coh == 5:
                bin_counts[n_flip] += 1
            else:
                # Normalize to 5-sample equivalent
                bin_counts[round(5.0 * n_flip / n_coh)] += 1
        bins = np.arange(7) - 0.5
        values = [bin_counts.get(i, 0) for i in range(6)]
        ax.bar(range(6), values, width=0.8, color=PIPELINE_COLORS.get(label, "gray"),
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Flips per item (out of 5)")
        ax.set_xticks(range(6))
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate mean flips per item
        total_flips = sum(i * c for i, c in bin_counts.items())
        total_items = sum(bin_counts.values())
        mean_flip = total_flips / total_items if total_items else 0
        ax.text(0.97, 0.95, f"n={total_items}\nmean={mean_flip:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                          edgecolor="gray"))
    axes[0].set_ylabel("Items")
    domain_label = dict(DOMAINS).get(domain, domain)
    fig.suptitle(f"Per-item flip distributions at T={temperature}, "
                 f"{context.replace('_', '-')}, non-simple, {domain_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
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
    # Filename stems must match the \includegraphics references in the paper
    # (paper/neurips_2025.tex, app:sampling): comp_* and med_*.
    short_names = {"computational": "comp", "medical_advice": "med"}
    for dkey, dlabel in DOMAINS:
        short = short_names.get(dkey, dkey[:4])
        out = os.path.join(args.output_dir, f"{short}_sampling_distributions.png")
        plot_one_domain(args.sampling_dir, dkey, out,
                         temperature=args.temperature, context=args.context)


if __name__ == "__main__":
    main()
