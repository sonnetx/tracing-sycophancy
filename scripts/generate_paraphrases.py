#!/usr/bin/env python3
"""Generate paraphrases of the correct and proposed answers for a subset of items.

Supports the paraphrase-robustness check for ΔLogOdds: the metric scores one
fixed (correct, incorrect) candidate pair per item, and long-form medical
answers admit many semantically equivalent surface forms. This script samples
a subset of items and asks GPT-4o for paraphrases of each candidate that
preserve all factual content. Scoring the paraphrased candidates (with the
challenge prompts unchanged) then measures how much the reported ΔLogOdds
depends on candidate surface form rather than content.

Cost note: default settings (200 items x 2 paraphrases x 2 candidates) make
~800 GPT-4o calls with ~350-token outputs, roughly $4-5. The script resumes
from an existing output file, so re-runs only pay for missing items.

Usage:
    python scripts/generate_paraphrases.py \
        --input data/processed/medical_advice.jsonl \
        --output data/processed/medical_advice_paraphrase_subset.jsonl \
        --backend-config config/models/gpt4o_judge.json \
        --n-items 200 --n-paraphrases 2 --seed 42
"""

import argparse
import random

from tqdm import tqdm

from src.utils import append_jsonl, load_backend, read_jsonl

PARAPHRASE_PROMPT = """Paraphrase the following answer to a medical question.
Preserve every factual claim, every named condition, treatment, and caveat, \
and keep approximately the same length. Change the wording and sentence \
structure as much as possible without altering the meaning. Return only the \
paraphrased answer, with no preamble.

Question: {question}

Answer to paraphrase:
{answer}"""


def paraphrase(backend, question: str, answer: str) -> str:
    response = backend.chat([
        {"role": "user",
         "content": PARAPHRASE_PROMPT.format(question=question, answer=answer)},
    ])
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Processed dataset JSONL (with challenges)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL: sampled items plus *_paraphrases fields")
    parser.add_argument("--backend-config", required=True,
                        help="API backend config (e.g. config/models/gpt4o_judge.json)")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-paraphrases", type=int, default=2,
                        help="Paraphrases per candidate, in addition to the original")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = read_jsonl(args.input)
    rng = random.Random(args.seed)
    subset = sorted(rng.sample(items, min(args.n_items, len(items))),
                    key=lambda r: r["id"])

    done_ids = set()
    try:
        done_ids = {r["id"] for r in read_jsonl(args.output)}
        print(f"Resuming: {len(done_ids)} items already in {args.output}")
    except FileNotFoundError:
        pass

    todo = [r for r in subset if r["id"] not in done_ids]
    n_calls = len(todo) * args.n_paraphrases * 2
    print(f"{len(todo)} items to paraphrase -> {n_calls} API calls")
    if not todo:
        return

    backend = load_backend(args.backend_config)

    for item in tqdm(todo, desc="paraphrasing"):
        correct = item["correct_answer"]
        proposed = item["proposed_answer"]
        item["correct_answer_paraphrases"] = [correct] + [
            paraphrase(backend, item["question"], correct)
            for _ in range(args.n_paraphrases)]
        item["proposed_answer_paraphrases"] = [proposed] + [
            paraphrase(backend, item["question"], proposed)
            for _ in range(args.n_paraphrases)]
        append_jsonl(item, args.output)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
