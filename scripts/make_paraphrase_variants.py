#!/usr/bin/env python3
"""Materialize scoring variants from a paraphrase-subset JSONL.

Takes the output of generate_paraphrases.py and writes one JSONL per variant,
each with plain correct_answer / proposed_answer fields, so score_logprobs.py
runs on them unmodified. Challenges are left untouched (they assert the
original wrong answer), so the variants differ only in the surface form of
the scored candidates, one side at a time:

    c0_w0  original correct, original wrong (within-subset baseline)
    c1_w0, c2_w0, ...  paraphrased correct, original wrong
    c0_w1, c0_w2, ...  original correct, paraphrased wrong

Usage:
    python scripts/make_paraphrase_variants.py \
        --input data/processed/medical_advice_paraphrase_subset.jsonl \
        --output-dir data/processed/paraphrase_variants
"""

import argparse
import copy
import os

from src.utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    items = read_jsonl(args.input)
    n_c = min(len(r.get("correct_answer_paraphrases", [])) for r in items)
    n_w = min(len(r.get("proposed_answer_paraphrases", [])) for r in items)
    if n_c < 2 or n_w < 2:
        raise SystemExit("Input has no paraphrases; run generate_paraphrases.py first")

    variants = [(0, 0)]
    variants += [(ci, 0) for ci in range(1, n_c)]
    variants += [(0, wi) for wi in range(1, n_w)]

    os.makedirs(args.output_dir, exist_ok=True)
    for ci, wi in variants:
        out = []
        for r in items:
            v = copy.deepcopy(r)
            v["correct_answer"] = r["correct_answer_paraphrases"][ci]
            v["proposed_answer"] = r["proposed_answer_paraphrases"][wi]
            v.pop("correct_answer_paraphrases", None)
            v.pop("proposed_answer_paraphrases", None)
            out.append(v)
        path = os.path.join(args.output_dir, f"variant_c{ci}_w{wi}.jsonl")
        write_jsonl(out, path)
        print(f"Wrote {path} ({len(out)} items)")

    print("VARIANTS=" + " ".join(f"c{ci}_w{wi}" for ci, wi in variants))


if __name__ == "__main__":
    main()
