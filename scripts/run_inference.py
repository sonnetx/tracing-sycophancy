#!/usr/bin/env python3
"""Combined inference: runs both generative and log-prob tracks in a single
model load, avoiding the cost of loading a 7B model twice.

Usage:
    python scripts/run_inference.py \
        --input data/processed/computational.jsonl \
        --output-dir data/results/exp1/olmo3-7b-base/ \
        --backend-config /tmp/hf_config.json \
        --model-type base \
        --model-name olmo3-7b-base \
        --checkpoint base
"""

import argparse
import os

from tqdm import tqdm

from scripts.generate_responses import generate_for_item
from scripts.score_logprobs import score_item
from src.utils import append_jsonl, load_backend, read_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Run both generative and log-prob tracks with a single model load")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file (preprocessed questions with challenges)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (writes responses.jsonl and logprob_scores.jsonl)")
    parser.add_argument("--backend-config", required=True,
                        help="Path to model backend config JSON")
    parser.add_argument("--model-type", required=True, choices=["chat", "base"],
                        help="Model type: 'chat' for instruct models, 'base' for completion models")
    parser.add_argument("--model-name", required=True,
                        help="Human-readable model name for results")
    parser.add_argument("--checkpoint", default="default",
                        help="Checkpoint identifier")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output files, skipping already-processed items")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    responses_path = os.path.join(args.output_dir, "responses.jsonl")
    logprob_path = os.path.join(args.output_dir, "logprob_scores.jsonl")

    # Load model ONCE
    backend = load_backend(args.backend_config)

    # Base models generate garbage — cap tokens to avoid wasting time
    if args.model_type == "base":
        backend.max_new_tokens = 128

    items = read_jsonl(args.input)

    # Handle resume for both tracks
    gen_done = set()
    lp_done = set()
    if args.resume:
        try:
            gen_done = {r["question_id"] for r in read_jsonl(responses_path)}
            print(f"Generative: {len(gen_done)} already processed")
        except FileNotFoundError:
            pass
        try:
            lp_done = {r["question_id"] for r in read_jsonl(logprob_path)}
            print(f"Log-prob: {len(lp_done)} already processed")
        except FileNotFoundError:
            pass

    print(f"Processing {len(items)} items with {args.model_name}")

    for item in tqdm(items, desc=f"Inference ({args.model_name})"):
        qid = item["id"]

        # Generative track
        if qid not in gen_done:
            try:
                result = generate_for_item(
                    item, backend, args.model_type, args.model_name, args.checkpoint)
                append_jsonl(result, responses_path)
            except Exception as e:
                print(f"\nError generating {qid}: {e}")

        # Log-prob track
        if qid not in lp_done:
            try:
                lp_result = score_item(item, backend, args.model_name, args.checkpoint)
                if lp_result is not None:
                    append_jsonl(lp_result, logprob_path)
            except Exception as e:
                print(f"\nError scoring {qid}: {e}")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
