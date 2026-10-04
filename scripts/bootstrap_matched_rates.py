#!/usr/bin/env python3
"""Recompute the paper's paired item-level behavioral bootstrap p-values.

Use the larger-budget runs. Think medical retains exp1 unless exp1_rerun has a copy.
Sorted question IDs and a fixed seed reproduce the 100,000-resample audit.
The score intervals and other experiments use their separately stated budgets.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TYPES = {"ethos", "justification", "citation"}
OLMO = ["olmo3-7b-" + m for m in (
    "think-sft", "think-dpo", "think", "instruct-sft", "instruct-dpo", "instruct")]
LLAMA = ["llama31-8b-instruct", "tulu3-llama31-8b-sft",
         "tulu3-llama31-8b-dpo", "tulu3-llama31-8b"]
PAIRS = [("olmo3-7b-base", m) for m in OLMO] + [("llama31-8b-base", m) for m in LLAMA]


def load(root, domain, model):
    path = root / "exp1_rerun" / domain / model / "evaluated.jsonl"
    if domain == "medical_advice" and "think" in model and not path.exists():
        path = root / "exp1" / domain / model / "evaluated.jsonl"
    return {r["question_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def correct(rows):
    return {q for q, r in rows.items() if r["initial"]["metrics"]["factual_accuracy"] == "correct"}


def rate(row):
    labels = [c["metrics"]["factual_accuracy"] for c in row["challenge_responses"]
              if c["challenge_id"].split("_")[0] in TYPES
              and c["metrics"]["factual_accuracy"] in {"correct", "incorrect"}]
    return labels.count("incorrect") / len(labels) if labels else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--analysis-dir", type=Path,
                        help="Update behavioral summaries used by the matched figure")
    args = parser.parse_args()
    results = {}
    for domain in ("computational", "medical_advice"):
        for base, model in PAIRS:
            b, s = load(args.results_dir, domain, base), load(args.results_dir, domain, model)
            eligible = sorted(correct(b) & correct(s))
            differences = []
            for q in eligible:
                rb, rs = rate(b[q]), rate(s[q])
                if rb is not None and rs is not None:
                    differences.append(rs - rb)
            d = np.asarray(differences)
            rng = np.random.default_rng(args.seed)
            means = []
            for start in range(0, args.resamples, 2000):
                indices = rng.integers(0, len(d), size=(min(2000, args.resamples-start), len(d)))
                means.append(d[indices].mean(axis=1))
            means = np.concatenate(means)
            tail = int((means >= 0).sum())
            p = tail / args.resamples
            results[f"{domain}/{model}"] = {
                "n_eligible": len(eligible), "n_test": len(d),
                "mean_difference": float(d.mean()), "tail_count": tail,
                "resamples": args.resamples, "seed": args.seed,
                "p": p, "mc_se": float(np.sqrt(p*(1-p)/args.resamples)),
                "difference_ci": np.percentile(means, [2.5, 97.5]).tolist(),
            }
            print(f"{domain}/{model}: n={len(d)} p={p:.5f} MC-SE={results[f'{domain}/{model}']['mc_se']:.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if args.analysis_dir:
        for domain in ("computational", "medical_advice"):
            path = args.analysis_dir / domain / "analysis/matched_summaries.json"
            summaries = json.loads(path.read_text())
            for records in summaries.values():
                for row in records:
                    key = f"{domain}/{row['model']}"
                    if key not in results:
                        continue
                    result = results[key]
                    paired = row["base_vs_stage_paired"]
                    assert abs(paired["mean_d"] - result["mean_difference"]) < 1e-10
                    paired.update(bootstrap_p=result["p"],
                                  bootstrap_ci_low=result["difference_ci"][0],
                                  bootstrap_ci_high=result["difference_ci"][1],
                                  n_bootstrap=args.resamples, rng_seed=args.seed)
            path.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
