"""Visualization for sycophancy analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")


def plot_checkpoint_trajectories(summaries: dict[str, dict], output_dir: str, title_prefix: str = "") -> None:
    """Plot metric trajectories across model checkpoints."""
    checkpoints = list(summaries.keys())
    if len(checkpoints) < 2:
        return

    metrics = {"Initial Accuracy": [], "Challenge Accuracy": [], "Agreement Rate": [],
               "Hedging Rate": [], "Refusal Rate": [], "Regressive Sycophancy": []}

    for cp in checkpoints:
        s = summaries[cp]
        metrics["Initial Accuracy"].append(s.get("initial", {}).get("accuracy_rate", 0))
        co = s.get("challenges", {}).get("overall", {})
        metrics["Challenge Accuracy"].append(co.get("accuracy_rate", 0))
        metrics["Agreement Rate"].append(co.get("agreement_rate", 0))
        metrics["Hedging Rate"].append(co.get("hedging_rate", 0))
        metrics["Refusal Rate"].append(co.get("refusal_rate", 0))
        metrics["Regressive Sycophancy"].append(s.get("sycophancy", {}).get("regressive_rate", 0))

    x = np.arange(len(checkpoints))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes.flatten()[idx]
        ax.plot(x, values, "o-", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title(name)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}Metrics Across Training Checkpoints", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, output_dir, "checkpoint_trajectories.png")
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


def plot_delta_log_odds_trajectory(summaries: dict[str, dict], output_dir: str,
                                   title_prefix: str = "") -> None:
    """Plot delta-log-odds across training checkpoints."""
    checkpoints = list(summaries.keys())
    if len(checkpoints) < 2:
        return

    means = []
    pct_syc = []
    for cp in checkpoints:
        ch = summaries[cp].get("challenges", {}).get("overall", {})
        means.append(ch.get("mean_delta_log_odds", 0))
        pct_syc.append(ch.get("pct_sycophantic", 0))

    x = np.arange(len(checkpoints))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, means, "o-", linewidth=2, markersize=6, color="#d62728")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Mean Delta Log-Odds")
    ax1.set_title("Sycophancy Signal (positive = sycophantic)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, pct_syc, "o-", linewidth=2, markersize=6, color="#2ca02c")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Fraction")
    ax2.set_title("% Questions with Sycophantic Shift")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}Log-Prob Sycophancy Across Checkpoints", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "logprob_checkpoint_trajectories.png")
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


def plot_base_vs_posttrained(base_summary: dict, pt_summary: dict, base_name: str,
                             pt_name: str, output_dir: str) -> None:
    """Plot side-by-side comparison of base vs post-trained model."""
    labels, base_vals, pt_vals = [], [], []

    for section, key, label in [("initial", "accuracy_rate", "Initial Accuracy"),
                                 ("initial", "hedging_rate", "Initial Hedging"),
                                 ("initial", "refusal_rate", "Initial Refusal")]:
        labels.append(label)
        base_vals.append(base_summary.get(section, {}).get(key, 0))
        pt_vals.append(pt_summary.get(section, {}).get(key, 0))

    for key, label in [("accuracy_rate", "Challenge Accuracy"), ("agreement_rate", "Agreement"),
                       ("hedging_rate", "Challenge Hedging"), ("refusal_rate", "Challenge Refusal")]:
        labels.append(label)
        base_vals.append(base_summary.get("challenges", {}).get("overall", {}).get(key, 0))
        pt_vals.append(pt_summary.get("challenges", {}).get("overall", {}).get(key, 0))

    for key, label in [("regressive_rate", "Regressive Sycophancy"), ("progressive_rate", "Progressive Sycophancy")]:
        labels.append(label)
        base_vals.append(base_summary.get("sycophancy", {}).get(key, 0))
        pt_vals.append(pt_summary.get("sycophancy", {}).get(key, 0))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, base_vals, width, label=base_name, color="#66b3ff")
    ax.bar(x + width / 2, pt_vals, width, label=pt_name, color="#ff9999")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Base vs Post-Trained Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, f"base_vs_posttrained_{base_name}_vs_{pt_name}.png")
    plt.close(fig)
