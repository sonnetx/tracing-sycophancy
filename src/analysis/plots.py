"""Visualization for sycophancy analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np


PIPELINE_COLORS = {
    "Think": "#1f77b4",
    "Instruct": "#d62728",
}
PIPELINE_MARKERS = {
    "Think": "s",
    "Instruct": "o",
}


def save_fig(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")


def plot_pipeline_trajectories(pipeline_data: dict, output_dir: str) -> None:
    """Plot metric trajectories across training stages for multiple pipelines.

    Args:
        pipeline_data: {pipeline_name: [(stage_label, summary), ...]}
    """
    metric_specs = [
        ("Initial Accuracy", lambda s: s.get("initial", {}).get("accuracy_rate", 0)),
        ("Challenge Accuracy", lambda s: s.get("challenges", {}).get("overall", {}).get("accuracy_rate", 0)),
        ("Agreement Rate", lambda s: s.get("challenges", {}).get("overall", {}).get("agreement_rate", 0)),
        ("Hedging Rate", lambda s: s.get("challenges", {}).get("overall", {}).get("hedging_rate", 0)),
        ("Refusal Rate", lambda s: s.get("challenges", {}).get("overall", {}).get("refusal_rate", 0)),
        ("Regressive Sycophancy", lambda s: s.get("sycophancy", {}).get("regressive_rate", 0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (metric_name, extract_fn) in enumerate(metric_specs):
        ax = axes.flatten()[idx]

        for pipe_name, stages in pipeline_data.items():
            labels = [label for label, _ in stages]
            values = [extract_fn(summary) for _, summary in stages]
            x = np.arange(len(labels))
            color = PIPELINE_COLORS.get(pipe_name, f"C{idx}")
            marker = PIPELINE_MARKERS.get(pipe_name, "o")
            ax.plot(x, values, f"{marker}-", linewidth=2, markersize=8,
                    color=color, label=pipe_name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_ylabel("Rate")
        ax.set_title(metric_name)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Sycophancy Metrics Across Training Pipeline (Base \u2192 SFT \u2192 DPO \u2192 Final)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, output_dir, "pipeline_trajectories.png")
    plt.close(fig)


def plot_pipeline_logprob_trajectories(pipeline_data: dict, output_dir: str) -> None:
    """Plot log-prob sycophancy metrics across training stages for multiple pipelines.

    Args:
        pipeline_data: {pipeline_name: [(stage_label, lp_summary), ...]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pipe_name, stages in pipeline_data.items():
        labels = [label for label, _ in stages]
        means = []
        pct_syc = []
        for _, summary in stages:
            ch = summary.get("challenges", {}).get("overall", {})
            means.append(ch.get("mean_delta_log_odds", 0))
            pct_syc.append(ch.get("pct_sycophantic", 0))

        x = np.arange(len(labels))
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")

        ax1.plot(x, means, f"{marker}-", linewidth=2, markersize=6,
                 color=color, label=pipe_name)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.plot(x, pct_syc, f"{marker}-", linewidth=2, markersize=6,
                 color=color, label=pipe_name)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Mean Delta Log-Odds")
    ax1.set_title("Sycophancy Signal (positive = sycophantic)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Fraction")
    ax2.set_title("% Questions with Sycophantic Shift")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Log-Prob Sycophancy Across Training Pipeline (Base \u2192 SFT \u2192 DPO \u2192 Final)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "pipeline_logprob_trajectories.png")
    plt.close(fig)


def plot_challenge_type_breakdown(summary: dict, output_dir: str, model_name: str = "") -> None:
    """Plot metrics broken down by challenge type."""
    challenges = summary.get("challenges", {})
    types = ["simple", "ethos", "justification", "citation"]
    metric_names = ["accuracy_rate", "agreement_rate", "hedging_rate", "refusal_rate"]
    metric_labels = ["Accuracy", "Agreement", "Hedging", "Refusal"]

    data = {label: [] for label in metric_labels}
    available_types = []
    for c_type in types:
        key = f"type_{c_type}"
        if key in challenges:
            available_types.append(c_type)
            for metric, label in zip(metric_names, metric_labels):
                data[label].append(challenges[key].get(metric, 0))

    if not available_types:
        return

    x = np.arange(len(available_types))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, values) in enumerate(data.items()):
        ax.bar(x + i * width, values, width, label=label)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([t.capitalize() for t in available_types])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Metrics by Challenge Type{' — ' + model_name if model_name else ''}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, f"challenge_type_breakdown_{model_name or 'model'}.png")
    plt.close(fig)


def plot_delta_log_odds_by_challenge_type(summary: dict, output_dir: str,
                                          model_name: str = "") -> None:
    """Bar chart of mean delta-log-odds by challenge type."""
    challenges = summary.get("challenges", {})
    types = ["simple", "ethos", "justification", "citation"]

    values = []
    available = []
    for c_type in types:
        key = f"type_{c_type}"
        if key in challenges:
            available.append(c_type.capitalize())
            values.append(challenges[key].get("mean_delta_log_odds", 0))

    if not available:
        return

    x = np.arange(len(available))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel("Mean Delta Log-Odds")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Sycophancy by Challenge Type{' — ' + model_name if model_name else ''}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, f"logprob_challenge_types_{model_name or 'model'}.png")
    plt.close(fig)
