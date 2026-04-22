#!/usr/bin/env python3
"""Step 2: Generate challenge/rebuttal prompts for preprocessed questions.

Usage:
    python scripts/generate_challenges.py \
        --input data/processed/computational.jsonl \
        --output data/processed/computational.jsonl \
        --backend-config config/models/ollama_challenge_gen.json \
        --challenge-type factual \
        --ethos "an expert in math"
"""

import argparse
import sys

from tqdm import tqdm

from src.utils import load_backend, read_jsonl, write_jsonl


CHALLENGE_TYPES = {
    "factual": "src.challenges.factual.FactualChallengeGenerator",
}


def get_generator(challenge_type: str, **kwargs):
    if challenge_type not in CHALLENGE_TYPES:
        print(f"Unknown challenge type: {challenge_type}. Available: {list(CHALLENGE_TYPES.keys())}")
        sys.exit(1)

    module_path, class_name = CHALLENGE_TYPES[challenge_type].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generate challenge prompts for sycophancy evaluation")
    parser.add_argument("--input", required=True, help="Input JSONL file (preprocessed questions)")
    parser.add_argument("--output", required=True, help="Output JSONL file (questions with challenges)")
    parser.add_argument("--backend-config", default=None,
                        help="Path to backend config JSON for generating proposed answers/justifications. "
                             "If not provided, placeholders will be used.")
    parser.add_argument("--challenge-type", required=True, choices=list(CHALLENGE_TYPES.keys()))
    parser.add_argument("--ethos", default="an expert",
                        help="Authority claim for ethos challenges")
    args = parser.parse_args()

    # Load backend if config provided
    backend = None
    if args.backend_config:
        backend = load_backend(args.backend_config)

    generator_kwargs = {}
    if args.challenge_type == "factual":
        generator_kwargs = {
            "ethos_statement": args.ethos,
        }

    generator = get_generator(args.challenge_type, **generator_kwargs)

    items = read_jsonl(args.input)
    results = []
    for item in tqdm(items, desc="Generating challenges"):
        item = generator.generate(item, backend)
        results.append(item)

    write_jsonl(results, args.output)
    print(f"Wrote {len(results)} items with challenges to {args.output}")


if __name__ == "__main__":
    main()
