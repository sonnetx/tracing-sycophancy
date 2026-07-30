#!/usr/bin/env python3
"""Paraphrase-dependence robustness for ΔLogOdds.

Loads per-variant logprob score files (from make_paraphrase_variants.py +
score_logprobs.py) and reports, per model:
  - Mean non-simple ΔLogOdds per variant (does the magnitude move?)
  - Per-question pairwise Spearman rank correlation between the c0_w0 baseline
    variant and each paraphrase variant (is the item-level structure stable?)
  - Mean per-question variance of ΔLogOdds across variants

And across models:
  - Spearman rank correlation of per-model mean ΔLogOdds between each variant
    and the c0_w0 baseline (are checkpoint rankings preserved?)

Usage:
    python scripts/analyze_paraphrase_robustness.py \
        --experiment-dir data/results/exp_paraphrase_robustness/medical_advice \
        --output data/results/exp_paraphrase_robustness/medical_advice/analysis/report.json
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.stats import load_logprob_results


def per_question_means(df: pd.DataFrame) -> pd.Series:
    ch = df[df["condition"] == "challenge"]
    ch = ch[ch["challenge_type"] != "simple"]
    return ch.groupby("question_id")["delta_log_odds"].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True,
                        help="Domain dir with per-model subdirs holding logprob_scores_<variant>.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pattern = re.compile(r"logprob_scores_(c\d+_w\d+)\.jsonl$")
    report = {"models": {}, "cross_model": {}}
    model_variant_means = {}

    for model in sorted(os.listdir(args.experiment_dir)):
        mdir = os.path.join(args.experiment_dir, model)
        if not os.path.isdir(mdir):
            continue
        files = {}
        for fname in sorted(os.listdir(mdir)):
            m = pattern.search(fname)
            if m:
                files[m.group(1)] = os.path.join(mdir, fname)
        if "c0_w0" not in files or len(files) < 2:
            continue

        per_q = {v: per_question_means(load_logprob_results(p))
                 for v, p in files.items()}
        base = per_q["c0_w0"]
        entry = {"variants": {}, "mean_per_question_variance": None}
        aligned = pd.DataFrame(per_q).dropna()
        entry["n_questions"] = int(len(aligned))
        entry["mean_per_question_variance"] = float(aligned.var(axis=1, ddof=1).mean())
        for v, s in per_q.items():
            joined = pd.concat([base, s], axis=1, keys=["base", "v"]).dropna()
            if v == "c0_w0":
                rho = 1.0
            else:
                rho, _ = spearmanr(joined["base"], joined["v"])
                rho = float(rho)
            entry["variants"][v] = {
                "mean_dlo": float(s.mean()),
                "spearman_vs_c0_w0": rho,
            }
        report["models"][model] = entry
        model_variant_means[model] = {v: d["mean_dlo"]
                                      for v, d in entry["variants"].items()}

    # Cross-model ranking stability per variant
    if len(model_variant_means) >= 3:
        models = sorted(model_variant_means)
        all_variants = sorted({v for d in model_variant_means.values() for v in d})
        base_vec = [model_variant_means[m].get("c0_w0") for m in models]
        for v in all_variants:
            vec = [model_variant_means[m].get(v) for m in models]
            if None in vec or None in base_vec:
                continue
            rho, _ = spearmanr(base_vec, vec)
            report["cross_model"][v] = {
                "per_model_mean_dlo": dict(zip(models, vec)),
                "spearman_vs_c0_w0": float(rho),
            }

    out = args.output or os.path.join(args.experiment_dir, "analysis", "paraphrase_robustness.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}\n")

    for model, entry in report["models"].items():
        print(f"== {model} (n={entry['n_questions']}, "
              f"mean per-question variance {entry['mean_per_question_variance']:.4f})")
        for v, d in sorted(entry["variants"].items()):
            print(f"   {v}: mean dLO {d['mean_dlo']:+.3f}  "
                  f"rho vs c0_w0 {d['spearman_vs_c0_w0']:.3f}")


if __name__ == "__main__":
    main()
