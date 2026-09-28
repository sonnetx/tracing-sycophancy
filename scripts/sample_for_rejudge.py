#!/usr/bin/env python3
"""Check 2: Sample 20% of evaluated responses for re-judging with an alternative judge.

Takes an existing evaluated.jsonl (GPT-4o judgments) and the corresponding
responses.jsonl, draws a stratified 20% sample by challenge_type, and writes:
  - A filtered evaluated JSONL preserving the original judge's labels
  - A filtered responses JSONL suitable for re-evaluation with evaluate.py

Usage:
    python scripts/sample_for_rejudge.py \
        --evaluated  data/results/exp1/computational/olmo3-7b-instruct/evaluated.jsonl \
        --responses  data/results/exp1/computational/olmo3-7b-instruct/responses.jsonl \
        --output-evaluated  data/results/exp_rejudge/computational/olmo3-7b-instruct/gpt4o_sample.jsonl \
        --output-responses  data/results/exp_rejudge/computational/olmo3-7b-instruct/sample_responses.jsonl \
        --fraction 0.20 \
        --seed 42
"""

import argparse
import os
import random
from collections import defaultdict

from src.utils import read_jsonl, write_jsonl


def stratified_sample_question_ids(evaluated_items: list[dict],
                                   fraction: float,
                                   seed: int) -> set:
    """Stratified sample of question_ids by the mix of challenge_types they contain."""
    # Build a coarse stratum per item: the set of challenge types present
    strata: dict[str, list[str]] = defaultdict(list)
    for item in evaluated_items:
        types = frozenset(
            cr.get("challenge_id", "").rsplit("_", 1)[0]
            for cr in item.get("challenge_responses", [])
        )
        strata[str(sorted(types))].append(item["question_id"])

    rng = random.Random(seed)
    sampled = set()
    for stratum_ids in strata.values():
        k = max(1, round(len(stratum_ids) * fraction))
        sampled.update(rng.sample(stratum_ids, min(k, len(stratum_ids))))

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample responses for alternative-judge re-evaluation")
    parser.add_argument("--evaluated", required=True,
                        help="Existing evaluated.jsonl (primary judge labels)")
    parser.add_argument("--responses", required=True,
                        help="Corresponding responses.jsonl (raw model outputs)")
    parser.add_argument("--output-evaluated", required=True,
                        help="Output path for sampled evaluated items")
    parser.add_argument("--output-responses", required=True,
                        help="Output path for sampled responses (input for evaluate.py)")
    parser.add_argument("--fraction", type=float, default=0.20,
                        help="Fraction to sample (default 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluated = read_jsonl(args.evaluated)
    responses = read_jsonl(args.responses)

    sampled_ids = stratified_sample_question_ids(evaluated, args.fraction, args.seed)
    print(f"Sampled {len(sampled_ids)} / {len(evaluated)} question_ids "
          f"({100 * len(sampled_ids) / max(1, len(evaluated)):.1f}%)")

    eval_sample = [item for item in evaluated if item["question_id"] in sampled_ids]
    resp_sample = [item for item in responses if item["question_id"] in sampled_ids]

    for path, data, label in [
        (args.output_evaluated, eval_sample, "evaluated sample"),
        (args.output_responses, resp_sample, "responses sample"),
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_jsonl(data, path)
        print(f"Wrote {len(data)} {label} items to {path}")


if __name__ == "__main__":
    main()
