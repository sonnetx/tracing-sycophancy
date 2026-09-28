#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze citation-component ablation: decompose ΔLogOdds into DOI, abstract, and length contributions.

Expected experiment layout:
    data/results/exp_citation_ablation/{dataset}/{model_name}/logprob_scores.jsonl

The ablation challenges contain four types that form a decomposition ladder:
  - length_control : neutral filler text matched to citation length  (pure length)
  - justification  : ethos + wrong answer + 1-2 sentence rationale  (content w/o citation)
  - citation_no_doi: justification + fake abstract (no DOI)          (abstract contribution)
  - citation       : citation_no_doi + DOI token                     (full citation)

Incremental contributions (in ΔLogOdds units):
  length     = mean ΔLO(length_control)
  abstract   = mean ΔLO(citation_no_doi) − mean ΔLO(justification)
  DOI        = mean ΔLO(citation)        − mean ΔLO(citation_no_doi)

Usage:
    python scripts/analyze_citation_ablation.py \
        --experiment-dir data/results/exp_citation_ablation \
        [--output-dir data/results/exp_citation_ablation/analysis]
"""

import argparse
import json
import os

from src.analysis.stats import _delta_logodds_stats, load_logprob_results

ABLATION_TYPES = ["justification", "citation_no_doi", "citation", "length_control"]
DECOMP_ORDER = ["justification", "citation_no_doi", "citation"]


def analyze_dataset_ablation(results_dir: str) -> dict:
    """Load all logprob score files in a dataset dir and compute per-type ΔLO stats."""
    model_results = {}
    for entry in sorted(os.listdir(results_dir)):
        lp_path = os.path.join(results_dir, entry, "logprob_scores.jsonl")
        if not os.path.isfile(lp_path):
            continue
        df = load_logprob_results(lp_path)
        challenges = df[df["condition"] == "challenge"]
        if len(challenges) == 0:
            continue

        type_stats = {}
        for c_type in ABLATION_TYPES:
            subset = challenges[challenges["challenge_type"] == c_type]
            if len(subset) == 0:
                continue
            per_q = subset.groupby("question_id")["delta_log_odds"].mean()
            type_stats[c_type] = _delta_logodds_stats(per_q)

        if not type_stats:
            continue

        # Incremental decomposition (only when all three ladder types are present)
        decomp = {}
        means = {t: type_stats[t]["mean_delta_log_odds"]
                 for t in DECOMP_ORDER if t in type_stats}
        if "length_control" in type_stats:
            decomp["length_contribution"] = type_stats["length_control"]["mean_delta_log_odds"]
        if "citation_no_doi" in means and "justification" in means:
            decomp["abstract_contribution"] = means["citation_no_doi"] - means["justification"]
        if "citation" in means and "citation_no_doi" in means:
            decomp["doi_contribution"] = means["citation"] - means["citation_no_doi"]

        model_results[entry] = {"by_type": type_stats, "decomposition": decomp}

    return model_results


def print_dataset_report(dataset: str, results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"{'='*70}")
    header = f"{'Model':<30} {'just':>8} {'cit_nodoi':>10} {'citation':>9} {'len_ctrl':>9} "
    header += f"{'DOI':>8} {'abstract':>9} {'length':>8}"
    print(header)
    print("-" * len(header))
    for model_name, data in sorted(results.items()):
        bt = data["by_type"]
        dc = data["decomposition"]
        row = f"{model_name:<30}"
        for c_type in ABLATION_TYPES:
            val = bt[c_type]["mean_delta_log_odds"] if c_type in bt else float("nan")
            row += f" {val:>9.3f}" if not (val != val) else f" {'N/A':>9}"
        row += f"  {dc.get('doi_contribution', float('nan')):>7.3f}"
        row += f"  {dc.get('abstract_contribution', float('nan')):>8.3f}"
        row += f"  {dc.get('length_contribution', float('nan')):>7.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Analyze citation ablation ΔLogOdds decomposition")
    parser.add_argument("--experiment-dir", default="data/results/exp_citation_ablation",
                        help="Root dir with {dataset}/{model_name}/logprob_scores.jsonl layout")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save ablation_summary.json (defaults to experiment-dir/analysis)")
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    output_dir = args.output_dir or os.path.join(exp_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for entry in sorted(os.listdir(exp_dir)):
        dataset_dir = os.path.join(exp_dir, entry)
        if not os.path.isdir(dataset_dir) or entry == "analysis":
            continue
        results = analyze_dataset_ablation(dataset_dir)
        if not results:
            continue
        all_results[entry] = results
        print_dataset_report(entry, results)

    if not all_results:
        print(f"No logprob_scores.jsonl files found under {exp_dir}")
        return

    out_path = os.path.join(output_dir, "ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved ablation summary to {out_path}")


if __name__ == "__main__":
    main()
