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
    plot_challenge_type_breakdown,
    plot_delta_log_odds_by_challenge_type,
    plot_pipeline_trajectories,
    plot_pipeline_logprob_trajectories,
)
from src.analysis.stats import (
    binomial_ci,
    compute_logprob_summary,
    compute_metrics_summary,
    load_logprob_results,
    load_results_as_dataframe,
    two_proportion_z_test,
)

# =====================================================================
# Training pipeline definitions
# =====================================================================
# Each pipeline is an ordered list of (model_dir_name, display_label).
# The base model is shared across pipelines.

TRAINING_PIPELINES = {
    "Think": [
        ("olmo3-7b-base", "Base"),
        ("olmo3-7b-think-sft", "SFT"),
        ("olmo3-7b-think-dpo", "DPO"),
        ("olmo3-7b-think", "Think"),
    ],
    "Instruct": [
        ("olmo3-7b-base", "Base"),
        ("olmo3-7b-instruct-sft", "SFT"),
        ("olmo3-7b-instruct-dpo", "DPO"),
        ("olmo3-7b-instruct", "Instruct"),
    ],
}


def find_evaluated_files(results_dir: str) -> dict[str, str]:
    """Find all evaluated.jsonl files in the results directory."""
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


def build_pipeline_data(summaries: dict, pipelines: dict) -> dict:
    """Build ordered pipeline data from summaries.

    Returns: {pipeline_name: [(label, summary), ...]} for pipelines with >=2 models present.
    Falls back to flat ordering if no pipeline matches.
    """
    pipeline_data = {}
    for pipe_name, stages in pipelines.items():
        ordered = []
        for model_name, label in stages:
            if model_name in summaries:
                ordered.append((label, summaries[model_name]))
        if len(ordered) >= 2:
            pipeline_data[pipe_name] = ordered

    # If no pipeline matched, fall back to flat ordering
    if not pipeline_data and len(summaries) >= 2:
        pipeline_data["Models"] = [(name, s) for name, s in summaries.items()]

    return pipeline_data


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

    # Pipeline trajectory plots
    pipeline_data = build_pipeline_data(summaries, TRAINING_PIPELINES)
    if pipeline_data:
        plot_pipeline_trajectories(pipeline_data, args.output_dir)

    # Statistical tests — base vs each final model
    base_name = "olmo3-7b-base"
    final_models = ["olmo3-7b-think", "olmo3-7b-instruct"]
    for final_name in final_models:
        if base_name in summaries and final_name in summaries:
            s_base = summaries[base_name]
            s_final = summaries[final_name]
            syc_b = s_base.get("sycophancy", {})
            syc_f = s_final.get("sycophancy", {})

            if syc_b.get("regressive_total", 0) > 0 and syc_f.get("regressive_total", 0) > 0:
                z_result = two_proportion_z_test(
                    syc_b["regressive_count"], syc_b["regressive_total"],
                    syc_f["regressive_count"], syc_f["regressive_total"],
                )
                print(f"\nZ-test: {base_name} vs {final_name} regressive sycophancy")
                print(f"  z={z_result['z_stat']:.3f}, p={z_result['p_value']:.4f}")

                ci_b = binomial_ci(syc_b["regressive_count"], syc_b["regressive_total"])
                ci_f = binomial_ci(syc_f["regressive_count"], syc_f["regressive_total"])
                print(f"  {base_name}: {ci_b['proportion']:.3f} (95% CI: {ci_b['ci_low']:.3f}-{ci_b['ci_high']:.3f})")
                print(f"  {final_name}: {ci_f['proportion']:.3f} (95% CI: {ci_f['ci_low']:.3f}-{ci_f['ci_high']:.3f})")

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

        # Pipeline log-prob trajectories
        lp_pipeline_data = build_pipeline_data(lp_summaries, TRAINING_PIPELINES)
        if lp_pipeline_data:
            plot_pipeline_logprob_trajectories(lp_pipeline_data, args.output_dir)

    print(f"\nAnalysis complete. Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
