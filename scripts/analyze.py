#!/usr/bin/env python3
"""Step 5: Analyze evaluated results — compute statistics and generate plots.

Usage:
    python scripts/analyze.py \
        --results-dir data/results/exp1/ \
        --output-dir data/results/exp1/analysis/

The results-dir should contain subdirectories per model, each with an evaluated.jsonl file.
"""

import argparse
import json
import os

from src.analysis.plots import (
    plot_base_vs_posttrained,
    plot_challenge_type_breakdown,
    plot_checkpoint_trajectories,
    plot_delta_log_odds_by_challenge_type,
    plot_delta_log_odds_trajectory,
)
from src.analysis.stats import (
    binomial_ci,
    compute_logprob_summary,
    compute_metrics_summary,
    load_logprob_results,
    load_results_as_dataframe,
    two_proportion_z_test,
)


def find_evaluated_files(results_dir: str) -> dict[str, str]:
    """Find all evaluated.jsonl files in the results directory.

    Returns: dict mapping model_name → file_path
    """
    models = {}
    for entry in os.listdir(results_dir):
        evaluated_path = os.path.join(results_dir, entry, "evaluated.jsonl")
        if os.path.isfile(evaluated_path):
            models[entry] = evaluated_path
    return models


def find_logprob_files(results_dir: str) -> dict[str, str]:
    """Find all logprob_scores.jsonl files in the results directory."""
    models = {}
    for entry in os.listdir(results_dir):
        lp_path = os.path.join(results_dir, entry, "logprob_scores.jsonl")
        if os.path.isfile(lp_path):
            models[entry] = lp_path
    return models


def main():
    parser = argparse.ArgumentParser(description="Analyze sycophancy evaluation results")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing model result subdirectories")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for analysis results and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover models
    models = find_evaluated_files(args.results_dir)
    if not models:
        print(f"No evaluated.jsonl files found in {args.results_dir}")
        return

    print(f"Found {len(models)} models: {list(models.keys())}")

    # Compute summaries for each model
    summaries = {}
    for model_name, eval_path in models.items():
        print(f"\nAnalyzing: {model_name}")
        df = load_results_as_dataframe(eval_path)
        summary = compute_metrics_summary(df)
        summaries[model_name] = summary

        # Print key metrics
        print(f"  Initial accuracy: {summary.get('initial', {}).get('accuracy_rate', 0):.3f}")
        if "challenges" in summary and "overall" in summary["challenges"]:
            co = summary["challenges"]["overall"]
            print(f"  Challenge accuracy: {co.get('accuracy_rate', 0):.3f}")
            print(f"  Agreement rate: {co.get('agreement_rate', 0):.3f}")
            print(f"  Hedging rate: {co.get('hedging_rate', 0):.3f}")
            print(f"  Refusal rate: {co.get('refusal_rate', 0):.3f}")
        if "sycophancy" in summary:
            s = summary["sycophancy"]
            print(f"  Regressive sycophancy: {s.get('regressive_rate', 0):.3f} ({s.get('regressive_count', 0)}/{s.get('regressive_total', 0)})")
            print(f"  Progressive sycophancy: {s.get('progressive_rate', 0):.3f} ({s.get('progressive_count', 0)}/{s.get('progressive_total', 0)})")

        # Per-model plots
        plot_challenge_type_breakdown(summary, args.output_dir, model_name)

    # Save summaries as JSON
    summary_path = os.path.join(args.output_dir, "summaries.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nSaved summaries to {summary_path}")

    # Checkpoint trajectory plot (if multiple checkpoints)
    if len(summaries) >= 2:
        plot_checkpoint_trajectories(summaries, args.output_dir)

    # Pairwise comparisons (base vs post-trained)
    model_names = list(summaries.keys())
    if len(model_names) == 2:
        plot_base_vs_posttrained(
            summaries[model_names[0]],
            summaries[model_names[1]],
            model_names[0],
            model_names[1],
            args.output_dir,
        )

    # Statistical tests between first two models
    if len(model_names) >= 2:
        s1 = summaries[model_names[0]]
        s2 = summaries[model_names[1]]

        syc1 = s1.get("sycophancy", {})
        syc2 = s2.get("sycophancy", {})

        if syc1.get("regressive_total", 0) > 0 and syc2.get("regressive_total", 0) > 0:
            z_result = two_proportion_z_test(
                syc1["regressive_count"], syc1["regressive_total"],
                syc2["regressive_count"], syc2["regressive_total"],
            )
            print(f"\nZ-test: {model_names[0]} vs {model_names[1]} regressive sycophancy")
            print(f"  z={z_result['z_stat']:.3f}, p={z_result['p_value']:.4f}")

            ci1 = binomial_ci(syc1["regressive_count"], syc1["regressive_total"])
            ci2 = binomial_ci(syc2["regressive_count"], syc2["regressive_total"])
            print(f"  {model_names[0]}: {ci1['proportion']:.3f} (95% CI: {ci1['ci_low']:.3f}-{ci1['ci_high']:.3f})")
            print(f"  {model_names[1]}: {ci2['proportion']:.3f} (95% CI: {ci2['ci_low']:.3f}-{ci2['ci_high']:.3f})")

    # =================================================================
    # Log-probability analysis
    # =================================================================
    lp_models = find_logprob_files(args.results_dir)
    if lp_models:
        print(f"\n--- Log-prob analysis ({len(lp_models)} models) ---")
        lp_summaries = {}
        for model_name, lp_path in lp_models.items():
            print(f"\nLog-prob: {model_name}")
            lp_df = load_logprob_results(lp_path)
            lp_summary = compute_logprob_summary(lp_df)
            lp_summaries[model_name] = lp_summary

            bl = lp_summary.get("baseline", {})
            print(f"  Baseline log-odds (mean): {bl.get('mean_log_odds', 0):.3f}")
            print(f"  % favoring correct: {bl.get('pct_favoring_correct', 0):.3f}")
            if "challenges" in lp_summary and "overall" in lp_summary["challenges"]:
                co = lp_summary["challenges"]["overall"]
                print(f"  Mean delta log-odds: {co.get('mean_delta_log_odds', 0):.3f}")
                print(f"  % sycophantic: {co.get('pct_sycophantic', 0):.3f}")

            plot_delta_log_odds_by_challenge_type(lp_summary, args.output_dir, model_name)

        # Save log-prob summaries
        lp_summary_path = os.path.join(args.output_dir, "logprob_summaries.json")
        with open(lp_summary_path, "w") as f:
            json.dump(lp_summaries, f, indent=2, default=str)
        print(f"\nSaved log-prob summaries to {lp_summary_path}")

        if len(lp_summaries) >= 2:
            plot_delta_log_odds_trajectory(lp_summaries, args.output_dir)

    print(f"\nAnalysis complete. Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
