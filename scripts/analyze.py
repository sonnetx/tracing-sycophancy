#!/usr/bin/env python3
"""Step 5: Analyze evaluated results, compute statistics, and generate plots.

Usage:
    # Analyze a single dataset (backward-compatible):
    python scripts/analyze.py \
        --results-dir data/results/exp1/computational \
        --output-dir data/results/exp1/computational/analysis

    # Analyze all datasets under an experiment + cross-dataset plots:
    python scripts/analyze.py \
        --experiment-dir data/results/exp1

The experiment-dir mode discovers all dataset subdirectories, runs per-dataset
analysis, and generates cross-dataset comparison plots.
"""

import argparse
import json
import os

from src.analysis.plots import (
    plot_behavioral_vs_representational,
    plot_challenge_type_breakdown,
    plot_challenge_type_heatmap,
    plot_control_comparison,
    plot_control_vs_sycophancy_scatter,
    plot_cross_pipeline_bar,
    plot_delta_log_odds_by_challenge_type,
    plot_domain_comparison,
    plot_pipeline_logprob_trajectories,
    plot_pipeline_trajectories,
    plot_progressive_vs_regressive,
)
from src.analysis.stats import (
    binomial_ci,
    compute_control_summary,
    compute_logprob_summary,
    compute_metrics_summary,
    compute_persistence_summary,
    load_logprob_results,
    load_results_as_dataframe,
    two_proportion_z_test,
)

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
    "Llama 3.1": [
        ("llama31-8b-base", "Base"),
        ("llama31-8b-instruct", "Instruct"),
    ],
    "Tulu 3": [
        ("llama31-8b-base", "Base"),
        ("tulu3-llama31-8b-sft", "SFT"),
        ("tulu3-llama31-8b-dpo", "DPO"),
        ("tulu3-llama31-8b", "Tulu 3"),
    ],
}


def find_evaluated_files(results_dir: str) -> dict[str, str]:
    models = {}
    for entry in os.listdir(results_dir):
        evaluated_path = os.path.join(results_dir, entry, "evaluated.jsonl")
        if os.path.isfile(evaluated_path):
            models[entry] = evaluated_path
    return models


def find_logprob_files(results_dir: str) -> dict[str, str]:
    models = {}
    for entry in os.listdir(results_dir):
        lp_path = os.path.join(results_dir, entry, "logprob_scores.jsonl")
        if os.path.isfile(lp_path):
            models[entry] = lp_path
    return models


def build_pipeline_data(summaries: dict, pipelines: dict) -> dict:
    pipeline_data = {}
    for pipe_name, stages in pipelines.items():
        ordered = []
        for model_key, label in stages:
            if model_key in summaries:
                ordered.append((label, summaries[model_key]))
        if len(ordered) >= 2:
            pipeline_data[pipe_name] = ordered

    if not pipeline_data:
        pipeline_data["Models"] = [(name, s) for name, s in summaries.items()]

    return pipeline_data


def analyze_dataset(results_dir: str, output_dir: str) -> dict:
    """Run full analysis on a single dataset. Returns dict of all computed data."""
    os.makedirs(output_dir, exist_ok=True)

    models = find_evaluated_files(results_dir)
    if not models:
        print(f"No evaluated.jsonl files found in {results_dir}")
        return {}

    print(f"Found {len(models)} models: {list(models.keys())}")

    # Compute summaries
    summaries = {}
    persistence_summaries = {}
    control_summaries = {}

    for model_name, eval_path in models.items():
        print(f"\nAnalyzing: {model_name}")
        df = load_results_as_dataframe(eval_path)
        summary = compute_metrics_summary(df)
        summaries[model_name] = summary

        ini = summary.get("initial", {})
        print(f"  Initial accuracy: {ini.get('accuracy_rate', 0):.3f}")
        if "challenges" in summary and "overall" in summary["challenges"]:
            co = summary["challenges"]["overall"]
            print(f"  Challenge accuracy: {co.get('accuracy_rate', 0):.3f}")
            print(f"  Agreement rate: {co.get('agreement_rate', 0):.3f}")
            print(f"  Hedging rate: {co.get('hedging_rate', 0):.3f}")
            print(f"  Refusal rate: {co.get('refusal_rate', 0):.3f}")
        if "sycophancy" in summary:
            s = summary["sycophancy"]
            print(f"  Regressive sycophancy: {s.get('regressive_rate', 0):.3f} "
                  f"({s.get('regressive_count', 0)}/{s.get('regressive_total', 0)})")
            print(f"  Progressive sycophancy: {s.get('progressive_rate', 0):.3f} "
                  f"({s.get('progressive_count', 0)}/{s.get('progressive_total', 0)})")

        persist = compute_persistence_summary(df)
        if persist:
            persistence_summaries[model_name] = persist
            pr = persist.get("persistence_rate", 0)
            print(f"  Persistence rate: {pr:.3f} "
                  f"({persist.get('persistent_chains', 0)}/{persist.get('total_chains', 0)})")
            esc = persist.get("escalation_rates", {})
            if esc:
                esc_str = ", ".join(
                    f"{t}={esc[t]['flip_rate']:.3f}"
                    for t in ["simple", "ethos", "justification", "citation"] if t in esc
                )
                print(f"  Escalation flip rates: {esc_str}")

        ctrl = compute_control_summary(df)
        if ctrl:
            control_summaries[model_name] = ctrl
            summaries[model_name]["controls"] = {
                k: v for k, v in ctrl.items() if k == "correct"
            }
            if "correct" in ctrl:
                c = ctrl["correct"]
                print(f"  Control (correct): flip_rate={c.get('flip_rate', 0):.3f} "
                      f"({c.get('flip_count', 0)}/{c.get('flip_total', 0)}), "
                      f"accuracy={c.get('accuracy_rate', 0):.3f}")
            if "wrong_vs_correct_control" in ctrl:
                c = ctrl["wrong_vs_correct_control"]
                print(f"  wrong_vs_correct_control: z={c['z_stat']:.3f}, p={c['p_value']:.6f}")

        per_model_dir = os.path.join(output_dir, "per_model")
        plot_challenge_type_breakdown(summary, per_model_dir, model_name)

    # Save JSON summaries
    for name, data in [("summaries", summaries),
                       ("persistence_summaries", persistence_summaries),
                       ("control_summaries", control_summaries)]:
        if data:
            path = os.path.join(output_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"\nSaved {name} to {path}")

    # Pipeline trajectory plots
    pipeline_data = build_pipeline_data(summaries, TRAINING_PIPELINES)
    if pipeline_data:
        plot_pipeline_trajectories(pipeline_data, output_dir)

    # Statistical tests
    base_name = "olmo3-7b-base"
    for final_name in ["olmo3-7b-think", "olmo3-7b-instruct"]:
        if base_name in summaries and final_name in summaries:
            syc_b = summaries[base_name].get("sycophancy", {})
            syc_f = summaries[final_name].get("sycophancy", {})
            if syc_b.get("regressive_total", 0) > 0 and syc_f.get("regressive_total", 0) > 0:
                z_result = two_proportion_z_test(
                    syc_b["regressive_count"], syc_b["regressive_total"],
                    syc_f["regressive_count"], syc_f["regressive_total"],
                )
                print(f"\nZ-test: {base_name} vs {final_name} regressive sycophancy")
                print(f"  z={z_result['z_stat']:.3f}, p={z_result['p_value']:.4f}")
                ci_b = binomial_ci(syc_b["regressive_count"], syc_b["regressive_total"])
                ci_f = binomial_ci(syc_f["regressive_count"], syc_f["regressive_total"])
                print(f"  {base_name}: {ci_b['proportion']:.3f} "
                      f"(95% CI: {ci_b['ci_low']:.3f}-{ci_b['ci_high']:.3f})")
                print(f"  {final_name}: {ci_f['proportion']:.3f} "
                      f"(95% CI: {ci_f['ci_low']:.3f}-{ci_f['ci_high']:.3f})")

    # Log-probability analysis
    lp_summaries = {}
    lp_pipeline_data = {}
    lp_models = find_logprob_files(results_dir)
    if lp_models:
        print(f"\n--- Log-prob analysis ({len(lp_models)} models) ---")
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

            per_model_dir = os.path.join(output_dir, "per_model")
            plot_delta_log_odds_by_challenge_type(lp_summary, per_model_dir, model_name)

        lp_path = os.path.join(output_dir, "logprob_summaries.json")
        with open(lp_path, "w") as f:
            json.dump(lp_summaries, f, indent=2, default=str)
        print(f"\nSaved log-prob summaries to {lp_path}")

        lp_pipeline_data = build_pipeline_data(lp_summaries, TRAINING_PIPELINES)
        if lp_pipeline_data:
            plot_pipeline_logprob_trajectories(lp_pipeline_data, output_dir)

    # Cross-pipeline comparison plots
    if pipeline_data and lp_pipeline_data:
        plot_behavioral_vs_representational(pipeline_data, lp_pipeline_data, output_dir)

    if pipeline_data:
        plot_control_comparison(pipeline_data, control_summaries, output_dir,
                                lp_pipeline_data=lp_pipeline_data)

    if summaries and lp_summaries:
        plot_cross_pipeline_bar(summaries, lp_summaries, output_dir)

    if summaries:
        plot_challenge_type_heatmap(summaries, output_dir)
        plot_control_vs_sycophancy_scatter(summaries, output_dir)
        plot_progressive_vs_regressive(summaries, output_dir)

    print(f"\nAnalysis complete. Plots saved to {output_dir}")

    return {
        "summaries": summaries,
        "lp_summaries": lp_summaries,
        "pipeline_data": pipeline_data,
        "lp_pipeline_data": lp_pipeline_data,
        "control_summaries": control_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze sycophancy evaluation results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-dir",
                       help="Experiment root (e.g. data/results/exp1). "
                            "Discovers all dataset subdirs and runs cross-dataset plots.")
    group.add_argument("--results-dir",
                       help="Single dataset results dir (backward-compatible)")
    parser.add_argument("--output-dir",
                        help="Output directory (required with --results-dir, "
                             "auto-created with --experiment-dir)")
    args = parser.parse_args()

    if args.results_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --results-dir")
        analyze_dataset(args.results_dir, args.output_dir)
        return

    # Experiment-dir mode: discover datasets and run everything
    exp_dir = args.experiment_dir
    datasets = sorted(
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
        and d != "cross_dataset"
        and any(
            os.path.isfile(os.path.join(exp_dir, d, m, "evaluated.jsonl"))
            for m in os.listdir(os.path.join(exp_dir, d))
            if os.path.isdir(os.path.join(exp_dir, d, m))
        )
    )

    if not datasets:
        print(f"No datasets found in {exp_dir}")
        return

    print(f"Discovered datasets: {datasets}")

    # Per-dataset analysis
    all_results = {}
    for dataset in datasets:
        results_dir = os.path.join(exp_dir, dataset)
        output_dir = os.path.join(results_dir, "analysis")
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        all_results[dataset] = analyze_dataset(results_dir, output_dir)

    # Cross-dataset comparison plots
    if len(all_results) >= 2:
        cross_dir = os.path.join(exp_dir, "cross_dataset")
        os.makedirs(cross_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Cross-dataset comparison")
        print(f"{'='*60}")

        # Domain comparison: logprob trajectories side by side
        lp_pipelines = {
            ds: res.get("lp_pipeline_data", {})
            for ds, res in all_results.items()
            if res.get("lp_pipeline_data")
        }
        ds_names = sorted(lp_pipelines.keys())
        if len(ds_names) >= 2:
            plot_domain_comparison(
                lp_pipelines[ds_names[0]],
                lp_pipelines[ds_names[1]],
                cross_dir,
            )

        # Cross-dataset heatmaps and scatter for each dataset
        for dataset, res in all_results.items():
            summaries = res.get("summaries", {})
            if summaries:
                plot_challenge_type_heatmap(
                    summaries, cross_dir,
                )
                # Rename to include dataset
                src = os.path.join(cross_dir, "challenge_type_heatmap.png")
                dst = os.path.join(cross_dir, f"challenge_type_heatmap_{dataset}.png")
                if os.path.exists(src):
                    os.rename(src, dst)

                plot_control_vs_sycophancy_scatter(summaries, cross_dir)
                src = os.path.join(cross_dir, "control_vs_sycophancy_scatter.png")
                dst = os.path.join(cross_dir, f"control_vs_sycophancy_scatter_{dataset}.png")
                if os.path.exists(src):
                    os.rename(src, dst)

        print(f"\nCross-dataset plots saved to {cross_dir}")

    print("\nAll analysis complete.")


if __name__ == "__main__":
    main()
