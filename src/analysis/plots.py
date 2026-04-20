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

# Pipelines to include in trajectory plots. Llama 3.1 is excluded because
# it has only 2 public checkpoints (Base, Instruct) with no intermediate
# SFT/DPO. Plotting it on the shared x-axis would place its Instruct point
# at the SFT position of the other pipelines, which misrepresents the data.
# Llama 3.1 still appears in Tables 1-4, Table 7, and the final-stage bar
# chart (plot_cross_pipeline_bar), where each model occupies its own column.
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
    """Plot log-prob sycophancy metrics across training stages.

    Left panel overlays mean ΔLogOdds (solid) and median ΔLogOdds (open marker,
    dotted) per pipeline. The gap between the two lines characterises the
    heavy-tailed distribution: a large gap implies a minority of items are
    driving the mean far from the typical item.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pipe_name, stages in pipeline_data.items():
        if pipe_name not in TRAJECTORY_PIPELINES:
            continue
        labels = [label for label, _ in stages]
        means = []
        medians = []
        pct_syc = []
        for _, summary in stages:
            ch = summary.get("challenges", {}).get("overall", {})
            means.append(ch.get("mean_delta_log_odds", 0))
            medians.append(ch.get("median_delta_log_odds", 0))
            pct_syc.append(ch.get("pct_sycophantic", 0))

        x = np.arange(len(labels))
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")

        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")
        ax1.plot(x, means, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax1.plot(x, medians, marker=marker, linestyle=":", linewidth=1.5,
                 markersize=7, markerfacecolor="white", markeredgecolor=color,
                 color=color, alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.plot(x, pct_syc, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel(r"$\Delta$LogOdds")
    ax1.set_title("Sycophancy Signal: Mean (solid) and Median (dotted)",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Fraction")
    ax2.set_title("% Questions with Sycophantic Shift", fontsize=12, fontweight="bold")
    # Do not hardcode ylim: Tulu 3 computational dips below 0.45 and gets clipped.
    # Let matplotlib auto-range based on actual data.
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Log-Prob Sycophancy Across Training Pipeline",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, output_dir, "pipeline_logprob_trajectories.png")
    plt.close(fig)


def plot_delta_log_odds_distribution(lp_dataframes: dict, pipelines: dict,
                                      output_dir: str) -> None:
    """Per-checkpoint distribution of per-item ΔLogOdds.

    Shows a box plot (Q1/median/Q3, whiskers to 1.5×IQR) per checkpoint ordered
    by pipeline, with outliers drawn as individual points. The filled diamond
    marks the mean; outliers that sit far above the whiskers make the heavy
    right tail visible and identify the items that drag the mean above the
    median.

    Parameters
    ----------
    lp_dataframes : dict
        {model_key: DataFrame with 'condition', 'near_random', 'delta_log_odds'}.
        DataFrames produced by load_logprob_results.
    pipelines : dict
        TRAINING_PIPELINES-shaped mapping {pipe_name: [(model_key, stage_label), ...]}.
    """
    # Build flat ordered list of (pipe_name, stage_label, model_key, series).
    # Dedupe shared base checkpoints (e.g., llama31-8b-base appears in both
    # Llama 3.1 and Tulu 3); keep the first pipeline it appears in.
    entries = []
    seen = set()
    for pipe_name, stages in pipelines.items():
        for model_key, stage_label in stages:
            if model_key in seen:
                continue
            df = lp_dataframes.get(model_key)
            if df is None:
                continue
            reliable = df[~df["near_random"]]
            challenges = reliable[reliable["condition"] == "challenge"]
            s = challenges["delta_log_odds"].dropna().values
            if len(s) == 0:
                continue
            entries.append((pipe_name, stage_label, model_key, s))
            seen.add(model_key)

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(entries) + 2), 5.5))

    positions = np.arange(len(entries))
    series = [e[3] for e in entries]
    labels = [f"{p}\n{l}" for p, l, _, _ in entries]
    colors = [PIPELINE_COLORS.get(p, "gray") for p, _, _, _ in entries]

    bp = ax.boxplot(series, positions=positions, widths=0.65, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.55,
                                    markeredgecolor="none"),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="gray", linewidth=1.0),
                    capprops=dict(color="gray", linewidth=1.0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)

    # Mean marker per checkpoint
    means = [float(np.mean(s)) for s in series]
    ax.scatter(positions, means, marker="D", s=42, color="black", zorder=5,
               label="Mean")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$LogOdds (per item)")
    ax.set_title("Per-item $\\Delta$LogOdds distribution by checkpoint "
                 "(box = IQR, whiskers = 1.5 IQR, points = outliers, diamond = mean)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, output_dir, "delta_log_odds_distribution.png")
    plt.close(fig)


def plot_delta_log_odds_distribution_by_context(lp_dataframes: dict,
                                                 pipelines: dict,
                                                 output_dir: str) -> None:
    """Distribution of per-item ΔLogOdds per checkpoint, split by challenge context.

    For each checkpoint, draws two side-by-side box plots: one for in-context
    challenges (challenge arrives after a commitment) and one for preemptive
    challenges (challenge precedes the question). The tail story from the
    outlier analysis (78-100% of the highest |ΔLogOdds| items come from the
    preemptive context) should be visible as a fatter right tail on the
    preemptive boxes.
    """
    # Build flat ordered list, deduping shared base checkpoints
    entries = []
    seen = set()
    for pipe_name, stages in pipelines.items():
        for model_key, stage_label in stages:
            if model_key in seen:
                continue
            df = lp_dataframes.get(model_key)
            if df is None:
                continue
            reliable = df[~df["near_random"]]
            challenges = reliable[reliable["condition"] == "challenge"]
            if len(challenges) == 0:
                continue
            in_ctx = challenges[challenges["challenge_context"] == "in_context"][
                "delta_log_odds"].dropna().values
            preempt = challenges[challenges["challenge_context"] == "preemptive"][
                "delta_log_odds"].dropna().values
            entries.append((pipe_name, stage_label, model_key, in_ctx, preempt))
            seen.add(model_key)

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(max(11, 0.85 * len(entries) + 2), 6))

    pair_centers = np.arange(len(entries))
    offset = 0.22
    width = 0.35
    positions_in = pair_centers - offset
    positions_pre = pair_centers + offset

    in_ctx_series = [e[3] for e in entries]
    preempt_series = [e[4] for e in entries]
    colors = [PIPELINE_COLORS.get(p, "gray") for p, _, _, _, _ in entries]
    labels = [f"{p}\n{l}" for p, l, _, _, _ in entries]

    flier_kw = dict(marker="o", markersize=2.5, alpha=0.55, markeredgecolor="none")
    median_kw = dict(color="black", linewidth=1.3)
    whisker_kw = dict(color="gray", linewidth=1.0)
    cap_kw = dict(color="gray", linewidth=1.0)

    bp_in = ax.boxplot(in_ctx_series, positions=positions_in, widths=width,
                       patch_artist=True, showfliers=True,
                       flierprops=flier_kw, medianprops=median_kw,
                       whiskerprops=whisker_kw, capprops=cap_kw)
    bp_pre = ax.boxplot(preempt_series, positions=positions_pre, widths=width,
                        patch_artist=True, showfliers=True,
                        flierprops=flier_kw, medianprops=median_kw,
                        whiskerprops=whisker_kw, capprops=cap_kw)

    for patch, color in zip(bp_in["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)
        patch.set_hatch("")
    for patch, color in zip(bp_pre["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)

    for flier, color in zip(bp_in["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_alpha(0.45)
    for flier, color in zip(bp_pre["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_alpha(0.75)

    # Mean markers
    means_in = [float(np.mean(s)) if len(s) else np.nan for s in in_ctx_series]
    means_pre = [float(np.mean(s)) if len(s) else np.nan for s in preempt_series]
    ax.scatter(positions_in, means_in, marker="D", s=28, color="black",
               zorder=6, edgecolor="white", linewidth=0.6)
    ax.scatter(positions_pre, means_pre, marker="D", s=28, color="black",
               zorder=6, edgecolor="white", linewidth=0.6)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xticks(pair_centers)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$LogOdds (per item)")
    ax.set_title("Per-item $\\Delta$LogOdds by checkpoint, split by challenge context "
                 "(light = in-context, dark = preemptive; diamond = mean)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend proxies
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    proxies = [
        Patch(facecolor="gray", alpha=0.30, edgecolor="gray", label="In-context"),
        Patch(facecolor="gray", alpha=0.75, edgecolor="gray", label="Preemptive"),
        Line2D([0], [0], marker="D", color="white", markerfacecolor="black",
               markersize=8, markeredgecolor="white", label="Mean"),
    ]
    ax.legend(handles=proxies, loc="upper left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, output_dir, "delta_log_odds_distribution_by_context.png")
    plt.close(fig)


def _behavioral_rate(summary: dict, metric: str) -> float:
    """Extract the behavioral y-axis value from a single-stage summary.

    metric="regressive" -> raw regressive sycophancy rate
    metric="net"        -> regressive rate minus correct-answer control flip rate
                           (content-specific sycophancy, isolating distractibility)
    """
    regr = summary.get("sycophancy", {}).get("regressive_rate", 0.0)
    if metric == "regressive":
        return regr
    if metric == "net":
        ctrl = summary.get("controls", {}).get("correct", {}).get("flip_rate", 0.0)
        return regr - ctrl
    raise ValueError(f"Unknown metric: {metric}")


def plot_behavioral_vs_representational(pipeline_data: dict, lp_pipeline_data: dict,
                                         output_dir: str, metric: str = "regressive") -> None:
    """Plot the surface-predictive dissociation: behavioral vs delta log-odds.

    metric: "regressive" (raw regressive rate) or "net" (Regr - Ctrl).
    Filename is suffixed with _net when metric="net" to avoid overwriting.
    """
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

        beh = [_behavioral_rate(s, metric) for _, s in gen_stages]
        ax1.plot(x, beh, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        dlo = [s.get("challenges", {}).get("overall", {}).get("mean_delta_log_odds", 0)
               for _, s in lp_stages]
        ax2.plot(x, dlo, marker=marker, linestyle=ls, linewidth=2,
                 markersize=8, color=color, label=pipe_name)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")

    if metric == "net":
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Net Sycophancy (Regr. $-$ Ctrl.)")
        ax1.set_title("Behavioral: Net (control-adjusted)",
                      fontsize=12, fontweight="bold")
    else:
        ax1.set_ylabel("Regressive Sycophancy Rate")
        ax1.set_title("Behavioral: Regressive Flip Rate",
                      fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean Delta Log-Odds")
    ax2.set_title("Representational: Delta Log-Odds", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    title_suffix = " (Net)" if metric == "net" else ""
    fig.suptitle(f"Behavioral vs. Representational Sycophancy Across Training{title_suffix}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    filename = "behavioral_vs_representational_net.png" if metric == "net" else "behavioral_vs_representational.png"
    save_fig(fig, output_dir, filename)
    plt.close(fig)


def plot_net_sycophancy_trajectories(pipeline_data: dict, output_dir: str) -> None:
    """Net sycophancy (Regr - Ctrl) across training stages.

    Net = regressive rate under wrong-answer challenge minus correct-answer
    control flip rate. Positive = content-specific sycophancy beyond the
    pressure-format distractibility; 0 = behavioral flips are entirely
    explained by pressure-sensitivity, not content.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for pipe_name in TRAJECTORY_PIPELINES:
        if pipe_name not in pipeline_data:
            continue
        stages = pipeline_data[pipe_name]
        labels = [label for label, _ in stages]
        color = PIPELINE_COLORS.get(pipe_name, "gray")
        marker = PIPELINE_MARKERS.get(pipe_name, "o")
        ls = PIPELINE_LINESTYLES.get(pipe_name, "-")
        x = np.arange(len(labels))
        net = [_behavioral_rate(s, "net") for _, s in stages]
        ax.plot(x, net, marker=marker, linestyle=ls, linewidth=2,
                markersize=8, color=color, label=pipe_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Net Sycophancy (Regr. $-$ Ctrl.)")
    ax.set_title("Content-specific (net) behavioral sycophancy across training",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, output_dir, "net_sycophancy_trajectories.png")
    plt.close(fig)


def plot_matched_subset(matched_summaries: dict, output_dir: str) -> None:
    """Grouped bar chart: base-on-intersection vs stage-on-intersection regressive rate.

    Each pipeline contributes one group of bar pairs (one pair per post-trained stage).
    For each stage we show (a) the base model's regressive rate restricted to items
    both base and stage answered correctly initially, and (b) the stage's rate on
    the same items. A significance star is drawn above each pair where p < 0.05.
    """
    # Collect stage records across pipelines, in pipeline-then-stage order
    records = []
    for pipe_name, pipe_records in matched_summaries.items():
        for rec in pipe_records:
            if rec["stage"] == "Base":
                continue  # Base-vs-base is trivial 0 difference
            records.append((pipe_name, rec))

    if not records:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(records) + 2), 5))
    x = np.arange(len(records))
    width = 0.38

    base_vals = [r["base_regressive_on_intersection"]["regressive_rate"] for _, r in records]
    stage_vals = [r["stage_regressive_on_intersection"]["regressive_rate"] for _, r in records]
    labels = [f"{p}\n{r['stage']}\n(n={r['n_intersection']})" for p, r in records]
    colors_stage = [PIPELINE_COLORS.get(p, "gray") for p, _ in records]

    ax.bar(x - width / 2, base_vals, width, label="Base on intersection",
           color="#999999", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, stage_vals, width, label="Post-trained stage",
           color=colors_stage, edgecolor="black", linewidth=0.5)

    # Significance stars for p < 0.05 contrasts
    max_bar = max(max(base_vals, default=0), max(stage_vals, default=0))
    star_pad = 0.02
    for i, (_, rec) in enumerate(records):
        ztest = rec.get("base_vs_stage_z_test")
        if ztest and ztest.get("p_value", 1.0) < 0.05:
            y = max(base_vals[i], stage_vals[i]) + star_pad
            ax.text(i, y, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Matched-subset regressive rate")
    # Give enough headroom above the tallest bar so stars are not clipped
    # (especially tall bars like Llama 3.1 Instruct base-on-intersection).
    ax.set_ylim(0, max_bar + star_pad + 0.04)
    ax.set_title("Matched-subset regressive sycophancy: base vs.\\ post-trained stage on identical items",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, output_dir, "matched_subset_bars.png")
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
    dlo_mean_values = []
    dlo_median_values = []
    ctrl_rates = []

    for display_name, model_key in final_models.items():
        if model_key not in summaries:
            continue
        s = summaries[model_key]
        lp = lp_summaries.get(model_key, {})

        model_names.append(display_name)
        regr_rates.append(s.get("sycophancy", {}).get("regressive_rate", 0))
        lp_overall = lp.get("challenges", {}).get("overall", {})
        dlo_mean_values.append(lp_overall.get("mean_delta_log_odds", 0))
        # Median is more robust to heavy-tailed ΔLogOdds distributions (common
        # on computational, where a small right tail inflates the mean).
        dlo_median_values.append(lp_overall.get("median_delta_log_odds", 0))
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
    # Use markers only (no connecting line): these are four independent final
    # models, not a sequence, so a line would imply a trend that does not exist.
    # Plot both mean and median ΔLogOdds — medians are robust to the heavy-tailed
    # per-item distribution documented in Appendix B.
    ax2.plot(x, dlo_mean_values, marker="D", linestyle="None", color="#0072B2",
             markersize=14, markeredgecolor="black", markeredgewidth=1.0,
             label="Mean $\\overline{\\Delta}$LogOdds", zorder=5)
    ax2.plot(x, dlo_median_values, marker="D", linestyle="None",
             markerfacecolor="white", markeredgecolor="#0072B2", markeredgewidth=1.6,
             markersize=12, label="Median $\\Delta$LogOdds", zorder=5)

    # Annotate each mean-ΔLogOdds marker with its value for readability
    # (secondary axis is easy to misread, especially when values span zero).
    for xi, dv in zip(x, dlo_mean_values):
        ax2.annotate(f"{dv:+.3f}", xy=(xi, dv),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=9, color="#0072B2", fontweight="bold")

    # Reference line at 0 on the secondary axis so negative vs positive is clear
    ax2.axhline(y=0, color="#0072B2", linestyle=":", alpha=0.4, linewidth=1)

    # Ensure enough headroom above the topmost mean marker so the value label
    # and marker don't get clipped by the top of the axis.
    if dlo_mean_values:
        dlo_max = max(dlo_mean_values)
        dlo_min = min(dlo_mean_values + dlo_median_values + [0])
        pad_up = max(0.06, 0.18 * (dlo_max - dlo_min))
        pad_dn = max(0.03, 0.10 * (dlo_max - dlo_min))
        ax2.set_ylim(dlo_min - pad_dn, dlo_max + pad_up)

    ax1.set_ylabel("Flip Rate")
    ax2.set_ylabel("Mean $\\overline{\\Delta}$LogOdds", color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")
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


def plot_challenge_type_trajectories(summaries: dict, output_dir: str) -> None:
    """Regressive flip rate by challenge type across training stages, per pipeline.

    Replaces the older model x challenge-type heatmap, which collapsed the
    training-stage dimension and so could not show how each challenge type's
    effectiveness evolves through SFT / DPO / final. One panel per pipeline;
    four lines per panel, one per challenge type. Colours run yellow-to-red
    to mirror the pressure-escalation ordering.
    """
    CHALLENGE_TYPES = ["simple", "ethos", "justification", "citation"]
    TYPE_COLORS = {
        "simple":        "#FDB863",
        "ethos":         "#E66101",
        "justification": "#B2182B",
        "citation":      "#67001F",
    }
    TYPE_MARKERS = {
        "simple": "o",
        "ethos": "s",
        "justification": "^",
        "citation": "D",
    }

    pipelines_to_plot = [
        ("Think", [
            ("olmo3-7b-base",        "Base"),
            ("olmo3-7b-think-sft",   "SFT"),
            ("olmo3-7b-think-dpo",   "DPO"),
            ("olmo3-7b-think",       "Think"),
        ]),
        ("Instruct", [
            ("olmo3-7b-base",         "Base"),
            ("olmo3-7b-instruct-sft", "SFT"),
            ("olmo3-7b-instruct-dpo", "DPO"),
            ("olmo3-7b-instruct",     "Instruct"),
        ]),
        ("Tulu 3", [
            ("llama31-8b-base",       "Base"),
            ("tulu3-llama31-8b-sft",  "SFT"),
            ("tulu3-llama31-8b-dpo",  "DPO"),
            ("tulu3-llama31-8b",      "Tulu 3"),
        ]),
    ]

    # Drop pipelines where no post-trained checkpoint is available
    present = [(name, stages) for name, stages in pipelines_to_plot
               if any(mk in summaries for mk, _ in stages[1:])]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 4.8),
                             sharey=True)
    if len(present) == 1:
        axes = [axes]

    # Determine global y-limit so panels share visual scale
    global_max = 0.0
    for _, stages in present:
        for mk, _ in stages:
            if mk not in summaries:
                continue
            syc = summaries[mk].get("sycophancy", {})
            for ct in CHALLENGE_TYPES:
                rate = syc.get(f"type_{ct}", {}).get("regressive_rate", 0)
                global_max = max(global_max, rate)
    y_top = min(1.0, global_max + 0.06)

    for ax, (pipe_name, stages) in zip(axes, present):
        labels = []
        positions = []
        values_by_type = {ct: [] for ct in CHALLENGE_TYPES}
        for idx, (mk, stage_label) in enumerate(stages):
            if mk not in summaries:
                continue
            labels.append(stage_label)
            positions.append(idx)
            syc = summaries[mk].get("sycophancy", {})
            for ct in CHALLENGE_TYPES:
                rate = syc.get(f"type_{ct}", {}).get("regressive_rate", None)
                values_by_type[ct].append(rate if rate is not None else np.nan)

        for ct in CHALLENGE_TYPES:
            ax.plot(positions, values_by_type[ct], marker=TYPE_MARKERS[ct],
                    linewidth=2, markersize=7, color=TYPE_COLORS[ct],
                    label=ct.capitalize())

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, y_top)
        ax.set_title(pipe_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Regressive flip rate")
    axes[-1].legend(loc="upper left", fontsize=9, framealpha=0.9,
                    title="Challenge type", title_fontsize=9)
    fig.suptitle("Regressive sycophancy by challenge type across training",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, output_dir, "challenge_type_trajectories.png")
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
