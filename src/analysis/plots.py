"""Visualization for sycophancy analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np


# Wong (2011) colorblind-safe palette
PIPELINE_COLORS = {
    "Think": "#0072B2",       # blue
    "Instruct": "#E69F00",    # orange
    "Tulu 3": "#009E73",      # teal
    "Llama 3.1": "#CC79A7",   # pink
}
PIPELINE_MARKERS = {
    "Think": "s",
    "Instruct": "o",
    "Tulu 3": "D",
    "Llama 3.1": "^",
}
PIPELINE_LINESTYLES = {
    "Think": "-",
    "Instruct": "-",
    "Tulu 3": "--",
    "Llama 3.1": ":",
}

# Pipelines to include in trajectory plots (exclude Llama, only 2 points)
TRAJECTORY_PIPELINES = {"Think", "Instruct", "Tulu 3"}


def save_fig(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")


def plot_pipeline_trajectories(pipeline_data: dict, output_dir: str) -> None:
    """Plot metric trajectories across training stages for multiple pipelines."""
    metric_specs = [
        ("Initial Accuracy", lambda s: s.get("initial", {}).get("accuracy_rate", 0)),
        ("Regressive Sycophancy", lambda s: s.get("sycophancy", {}).get("regressive_rate", 0)),
        ("Challenge Accuracy", lambda s: s.get("challenges", {}).get("overall", {}).get("accuracy_rate", 0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric_name, extract_fn) in enumerate(metric_specs):
        ax = axes[idx]

        for pipe_name, stages in pipeline_data.items():
            if pipe_name not in TRAJECTORY_PIPELINES:
                continue
            labels = [label for label, _ in stages]
            values = [extract_fn(summary) for _, summary in stages]
            x = np.arange(len(labels))
            color = PIPELINE_COLORS.get(pipe_name, "gray")
            marker = PIPELINE_MARKERS.get(pipe_name, "o")
            ls = PIPELINE_LINESTYLES.get(pipe_name, "-")
            ax.plot(x, values, marker=marker, linestyle=ls, linewidth=2,
                    markersize=8, color=color, label=pipe_name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_ylabel("Rate")
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 0.55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Behavioral Metrics Across Training Pipeline",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "pipeline_trajectories.png")
    plt.close(fig)


def plot_pipeline_logprob_trajectories(pipeline_data: dict, output_dir: str) -> None:
    """Plot log-prob sycophancy metrics across training stages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pipe_name, stages in pipeline_data.items():
        if pipe_name not in TRAJECTORY_PIPELINES:
            continue
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

        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")
        ax1.plot(x, means, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.plot(x, pct_syc, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Mean Delta Log-Odds")
    ax1.set_title("Sycophancy Signal (positive = sycophantic)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Fraction")
    ax2.set_title("% Questions with Sycophantic Shift", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.45, 0.75)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Log-Prob Sycophancy Across Training Pipeline",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "pipeline_logprob_trajectories.png")
    plt.close(fig)


def plot_behavioral_vs_representational(pipeline_data: dict, lp_pipeline_data: dict,
                                         output_dir: str) -> None:
    """Plot the behavioral-representational dissociation: regressive syco vs delta log-odds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pipe_name in TRAJECTORY_PIPELINES:
        if pipe_name not in pipeline_data or pipe_name not in lp_pipeline_data:
            continue

        gen_stages = pipeline_data[pipe_name]
        lp_stages = lp_pipeline_data[pipe_name]
        labels = [label for label, _ in gen_stages]
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")
        x = np.arange(len(labels))

        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")

        # Regressive sycophancy (behavioral)
        regr = [s.get("sycophancy", {}).get("regressive_rate", 0) for _, s in gen_stages]
        ax1.plot(x, regr, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        # Delta log-odds (representational)
        dlo = [s.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0)
               for _, s in lp_stages]
        ax2.plot(x, dlo, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")

    ax1.set_ylabel("Regressive Sycophancy Rate")
    ax1.set_title("Behavioral: Regressive Flip Rate", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean Delta Log-Odds")
    ax2.set_title("Representational: Delta Log-Odds", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Behavioral vs. Representational Sycophancy Across Training",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "behavioral_vs_representational.png")
    plt.close(fig)


def plot_control_comparison(pipeline_data: dict, control_summaries: dict,
                            output_dir: str,
                            lp_pipeline_data: dict | None = None) -> None:
    """Plot wrong-answer vs correct-answer control flip rates, with optional logprob panel."""
    has_lp = lp_pipeline_data is not None and len(lp_pipeline_data) > 0
    ncols = 2 if has_lp else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax_gen = axes[0]

    for pipe_name in TRAJECTORY_PIPELINES:
        if pipe_name not in pipeline_data:
            continue

        stages = pipeline_data[pipe_name]
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")
        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")

        stage_labels = []
        wrong_vals = []
        correct_vals = []

        for label, summary in stages:
            regr = summary.get("sycophancy", {}).get("regressive_rate", 0)
            wrong_vals.append(regr)
            stage_labels.append(label)
            ctrl = summary.get("controls", {})
            correct_vals.append(ctrl.get("correct", {}).get("flip_rate", 0))

        x = np.arange(len(stage_labels))

        ax_gen.plot(x, wrong_vals, marker=marker, linestyle=ls, linewidth=2,
                    markersize=8, color=color, label=f"{pipe_name} (wrong-answer)")
        ax_gen.plot(x, correct_vals, marker=marker, linestyle=":", linewidth=1.5,
                    markersize=6, color=color, alpha=0.5,
                    label=f"{pipe_name} (correct ctrl)")

        ax_gen.set_xticks(x)
        ax_gen.set_xticklabels(stage_labels, rotation=45, ha="right")

    ax_gen.set_ylabel("Flip Rate")
    ax_gen.set_title("Generative: Wrong-Answer vs. Control",
                     fontsize=12, fontweight="bold")
    ax_gen.grid(True, alpha=0.3)
    ax_gen.legend(fontsize=7)

    # Log-prob panel: wrong-answer vs correct-answer control delta log-odds
    if has_lp:
        ax_lp = axes[1]
        for pipe_name in TRAJECTORY_PIPELINES:
            if pipe_name not in lp_pipeline_data:
                continue

            stages = lp_pipeline_data[pipe_name]
            color = PIPELINE_COLORS.get(pipe_name, "gray")
            marker = PIPELINE_MARKERS.get(pipe_name, "o")
            ls = PIPELINE_LINESTYLES.get(pipe_name, "-")

            labels = [label for label, _ in stages]
            # Wrong-answer challenges: average across simple/ethos/justification/citation
            wrong_dlo = []
            ctrl_dlo = []
            for _, s in stages:
                ch = s.get("challenges", {})
                # Average wrong-answer types
                wrong_vals = [
                    ch.get(f"type_{t}", {}).get("mean_delta_log_odds", 0)
                    for t in ["simple", "ethos", "justification", "citation"]
                    if f"type_{t}" in ch
                ]
                wrong_dlo.append(np.mean(wrong_vals) if wrong_vals else 0)
                # Correct-answer control
                ctrl_dlo.append(
                    ch.get("type_correct", {}).get("mean_delta_log_odds", 0)
                )

            x = np.arange(len(labels))

            ax_lp.plot(x, wrong_dlo, marker=marker, linestyle=ls, linewidth=2,
                       markersize=8, color=color,
                       label=f"{pipe_name} (wrong-answer)")
            ax_lp.plot(x, ctrl_dlo, marker=marker, linestyle=":", linewidth=1.5,
                       markersize=6, color=color, alpha=0.5,
                       label=f"{pipe_name} (correct ctrl)")
            ax_lp.set_xticks(x)
            ax_lp.set_xticklabels(labels, rotation=45, ha="right")

        ax_lp.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_lp.set_ylabel("Mean Delta Log-Odds")
        ax_lp.set_title("Log-Prob: Wrong-Answer vs. Control",
                        fontsize=12, fontweight="bold")
        ax_lp.grid(True, alpha=0.3)
        ax_lp.legend(fontsize=7)

    fig.suptitle("Control Validation and Representational Sycophancy",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "control_comparison.png")
    plt.close(fig)


def plot_cross_pipeline_bar(summaries: dict, lp_summaries: dict,
                            output_dir: str) -> None:
    """Bar chart comparing all final-stage models on key metrics."""
    final_models = {
        "OLMo Think": "olmo3-7b-think",
        "OLMo Instruct": "olmo3-7b-instruct",
        "Llama 3.1 Inst.": "llama31-8b-instruct",
        "Tulu 3": "tulu3-llama31-8b",
    }

    model_names = []
    regr_rates = []
    dlo_values = []
    ctrl_rates = []

    for display_name, model_key in final_models.items():
        if model_key not in summaries:
            continue
        s = summaries[model_key]
        lp = lp_summaries.get(model_key, {})

        model_names.append(display_name)
        regr_rates.append(s.get("sycophancy", {}).get("regressive_rate", 0))
        dlo_values.append(lp.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0))
        ctrl = s.get("controls", {}).get("correct", {})
        ctrl_rates.append(ctrl.get("flip_rate", 0))

    if not model_names:
        return

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, regr_rates, width, label="Regressive Syco. Rate",
                     color="#E69F00", alpha=0.8)
    bars2 = ax1.bar(x + width/2, ctrl_rates, width, label="Correct-Answer Ctrl. Flip",
                     color="#56B4E9", alpha=0.8)
    line = ax2.plot(x, dlo_values, "s-", color="#0072B2", linewidth=2, markersize=10,
                     label="Mean Delta Log-Odds")

    ax1.set_ylabel("Flip Rate")
    ax2.set_ylabel("Mean Delta Log-Odds", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.grid(True, alpha=0.3, axis="y")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.suptitle("Final-Stage Model Comparison: Behavioral vs. Representational",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "cross_pipeline_bar.png")
    plt.close(fig)


def plot_challenge_type_breakdown(summary: dict, output_dir: str, model_name: str = "") -> None:
    """Plot metrics broken down by challenge type."""
    challenges = summary.get("challenges", {})
    types = ["simple", "ethos", "justification", "citation"]
    metric_names = ["accuracy_rate"]
    metric_labels = ["Accuracy"]

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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, data["Accuracy"], color="#0072B2", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in available_types])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Accuracy by Challenge Type: {model_name}" if model_name else
                 "Accuracy by Challenge Type", fontsize=12, fontweight="bold")
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
    colors = ["#E69F00" if v > 0 else "#009E73" for v in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel("Mean Delta Log-Odds")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Sycophancy by Challenge Type: {model_name}" if model_name else
                 "Sycophancy by Challenge Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, f"logprob_challenge_types_{model_name or 'model'}.png")
    plt.close(fig)


# -----------------------------------------------------------------------
# Cross-dataset and cross-model comparison plots
# -----------------------------------------------------------------------

def plot_domain_comparison(comp_lp_pipeline_data: dict, med_lp_pipeline_data: dict,
                           output_dir: str) -> None:
    """Side-by-side delta log-odds trajectories for computational vs medical."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for pipe_name in TRAJECTORY_PIPELINES:
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")
        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")

        if pipe_name in comp_lp_pipeline_data:
            stages = comp_lp_pipeline_data[pipe_name]
            labels = [label for label, _ in stages]
            vals = [s.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0)
                    for _, s in stages]
            x = np.arange(len(labels))
            ax1.plot(x, vals, marker=marker, linestyle=ls, linewidth=2,
                     markersize=8, color=color, label=pipe_name)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha="right")

        if pipe_name in med_lp_pipeline_data:
            stages = med_lp_pipeline_data[pipe_name]
            labels = [label for label, _ in stages]
            vals = [s.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0)
                    for _, s in stages]
            x = np.arange(len(labels))
            ax2.plot(x, vals, marker=marker, linestyle=ls, linewidth=2,
                     markersize=8, color=color, label=pipe_name)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha="right")

    for ax in (ax1, ax2):
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_ylabel("Mean Delta Log-Odds")
    ax1.set_title("Computational", fontsize=12, fontweight="bold")
    ax2.set_title("Medical", fontsize=12, fontweight="bold")

    fig.suptitle("Domain Comparison: Log-Prob Sycophancy Across Training",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "domain_comparison_logprob.png")
    plt.close(fig)


def plot_challenge_type_heatmap(summaries: dict, output_dir: str) -> None:
    """Heatmap of regressive flip rate by model x challenge type."""
    types = ["simple", "ethos", "justification", "citation"]
    # Order: base models first, then pipelines in training order
    model_order = [
        ("OLMo Base", "olmo3-7b-base"),
        ("OLMo Think SFT", "olmo3-7b-think-sft"),
        ("OLMo Think DPO", "olmo3-7b-think-dpo"),
        ("OLMo Think", "olmo3-7b-think"),
        ("OLMo Inst. SFT", "olmo3-7b-instruct-sft"),
        ("OLMo Inst. DPO", "olmo3-7b-instruct-dpo"),
        ("OLMo Instruct", "olmo3-7b-instruct"),
        ("Llama 3.1 Base", "llama31-8b-base"),
        ("Llama 3.1 Inst.", "llama31-8b-instruct"),
        ("Tulu 3 SFT", "tulu3-llama31-8b-sft"),
        ("Tulu 3 DPO", "tulu3-llama31-8b-dpo"),
        ("Tulu 3", "tulu3-llama31-8b"),
    ]

    model_labels = []
    data = []
    for display, key in model_order:
        if key not in summaries:
            continue
        s = summaries[key]
        syc = s.get("sycophancy", {})
        row = []
        for c_type in types:
            rate = syc.get(f"type_{c_type}", {}).get("regressive_rate", 0)
            row.append(rate)
        if any(r > 0 for r in row):
            data.append(row)
            model_labels.append(display)

    if not data:
        return

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(8, max(4, len(model_labels) * 0.45)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)

    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_yticklabels(model_labels)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(types)):
            val = data[i, j]
            text_color = "white" if val > 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    fig.colorbar(im, ax=ax, label="Regressive Flip Rate", shrink=0.8)
    ax.set_title("Regressive Sycophancy by Model and Challenge Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, "challenge_type_heatmap.png")
    plt.close(fig)


def plot_control_vs_sycophancy_scatter(summaries: dict, output_dir: str) -> None:
    """Scatter: correct-answer control flip rate vs regressive sycophancy rate."""
    # Color by pipeline family
    model_to_pipeline = {
        "olmo3-7b-base": "Think",  # shared base
        "olmo3-7b-think-sft": "Think",
        "olmo3-7b-think-dpo": "Think",
        "olmo3-7b-think": "Think",
        "olmo3-7b-instruct-sft": "Instruct",
        "olmo3-7b-instruct-dpo": "Instruct",
        "olmo3-7b-instruct": "Instruct",
        "llama31-8b-base": "Llama 3.1",
        "llama31-8b-instruct": "Llama 3.1",
        "tulu3-llama31-8b-sft": "Tulu 3",
        "tulu3-llama31-8b-dpo": "Tulu 3",
        "tulu3-llama31-8b": "Tulu 3",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_pipes = set()

    for model_key, s in summaries.items():
        pipe = model_to_pipeline.get(model_key)
        if pipe is None:
            continue
        ctrl = s.get("controls", {}).get("correct", {})
        ctrl_rate = ctrl.get("flip_rate", 0)
        regr_rate = s.get("sycophancy", {}).get("regressive_rate", 0)
        if regr_rate == 0:
            continue

        color = PIPELINE_COLORS.get(pipe, "gray")
        marker = PIPELINE_MARKERS.get(pipe, "o")
        label = pipe if pipe not in plotted_pipes else None
        plotted_pipes.add(pipe)

        ax.scatter(ctrl_rate, regr_rate, c=color, marker=marker, s=80,
                   label=label, edgecolors="black", linewidth=0.5, zorder=3)

        # Label the point
        short = model_key.replace("olmo3-7b-", "").replace("llama31-8b-", "").replace("tulu3-llama31-8b", "tulu3")
        ax.annotate(short, (ctrl_rate, regr_rate), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", alpha=0.7)

    # Diagonal line: if ctrl == regr, no sycophancy-specific effect
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="ctrl = regressive")

    ax.set_xlabel("Correct-Answer Control Flip Rate")
    ax.set_ylabel("Regressive Sycophancy Rate (wrong-answer)")
    ax.set_title("Control Stability vs. Sycophancy",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, output_dir, "control_vs_sycophancy_scatter.png")
    plt.close(fig)


def plot_progressive_vs_regressive(summaries: dict, output_dir: str) -> None:
    """Paired bar chart: progressive vs regressive sycophancy per model."""
    model_order = [
        ("OLMo Base", "olmo3-7b-base"),
        ("Think SFT", "olmo3-7b-think-sft"),
        ("Think DPO", "olmo3-7b-think-dpo"),
        ("Think", "olmo3-7b-think"),
        ("Inst. SFT", "olmo3-7b-instruct-sft"),
        ("Inst. DPO", "olmo3-7b-instruct-dpo"),
        ("Instruct", "olmo3-7b-instruct"),
        ("Llama Base", "llama31-8b-base"),
        ("Llama Inst.", "llama31-8b-instruct"),
        ("Tulu SFT", "tulu3-llama31-8b-sft"),
        ("Tulu DPO", "tulu3-llama31-8b-dpo"),
        ("Tulu 3", "tulu3-llama31-8b"),
    ]

    labels = []
    regr = []
    prog = []
    for display, key in model_order:
        if key not in summaries:
            continue
        syc = summaries[key].get("sycophancy", {})
        r = syc.get("regressive_rate", 0)
        p = syc.get("progressive_rate", 0)
        if r == 0 and p == 0:
            continue
        labels.append(display)
        regr.append(r)
        prog.append(p)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, regr, width, label="Regressive (correct \u2192 incorrect)",
           color="#E69F00", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, prog, width, label="Progressive (incorrect \u2192 correct)",
           color="#56B4E9", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Progressive vs. Regressive Sycophancy Across Models",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, output_dir, "progressive_vs_regressive.png")
    plt.close(fig)
