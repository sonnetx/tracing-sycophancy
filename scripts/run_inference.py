#!/usr/bin/env python3
"""Combined inference: runs both generative and log-prob tracks in a single
model load, using batched vLLM calls for dramatically faster throughput.

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
from src.utils import (
    append_jsonl,
    format_challenge_prompt,
    format_initial_prompt,
    format_logprob_baseline_prompt,
    format_logprob_challenge_prompt,
    load_backend,
    read_jsonl,
)

BATCH_SIZE = 50  # items per batch (50 items → up to ~450 generation calls batched)


def run_generative_batched(items, backend, model_type, model_name, checkpoint,
                           output_path, done_ids, batch_size=BATCH_SIZE):
    """Run generative track with batched vLLM calls."""
    remaining = [item for item in items if item["id"] not in done_ids]
    if not remaining:
        print("Generative: all items already processed")
        return
    print(f"Generative: {len(remaining)} items to process (batched, chunk={batch_size})")

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        # Phase 1: batch-generate initial responses
        initial_prompts = [format_initial_prompt(item["question"], model_type)
                           for item in batch]
        if model_type == "chat":
            initial_responses = backend.chat_batch(initial_prompts)
        else:
            initial_responses = backend.complete_batch(initial_prompts)

        # Phase 2: batch-generate all challenge responses
        challenge_prompts = []
        challenge_meta = []  # (item_idx, challenge)
        for idx, (item, init_resp) in enumerate(zip(batch, initial_responses)):
            for challenge in item.get("challenges", []):
                prompt = format_challenge_prompt(
                    question=item["question"],
                    initial_response=init_resp,
                    challenge=challenge["prompt"],
                    context=challenge["context"],
                    model_type=model_type,
                )
                challenge_prompts.append(prompt)
                challenge_meta.append((idx, challenge))

        if challenge_prompts:
            if model_type == "chat":
                challenge_responses = backend.chat_batch(challenge_prompts)
            else:
                challenge_responses = backend.complete_batch(challenge_prompts)
        else:
            challenge_responses = []

        # Phase 3: assemble results and save per-item
        item_challenge_resps = {i: [] for i in range(len(batch))}
        for (idx, challenge), resp in zip(challenge_meta, challenge_responses):
            ctx = challenge.get("context", "")
            if model_type == "chat":
                fmt = "chat"
            elif ctx == "in_context":
                fmt = "completion_concatenated"
            else:
                fmt = "completion"
            item_challenge_resps[idx].append({
                "challenge_id": challenge["id"],
                "response": resp,
                "prompt_format": fmt,
            })

        for idx, (item, init_resp) in enumerate(zip(batch, initial_responses)):
            result = {
                "question_id": item["id"],
                "model": model_name,
                "checkpoint": checkpoint,
                "model_type": model_type,
                "initial": {
                    "response": init_resp,
                    "prompt_format": "chat" if model_type == "chat" else "completion",
                },
                "challenge_responses": item_challenge_resps[idx],
            }
            append_jsonl(result, output_path)

        done = batch_start + len(batch)
        print(f"  Generative: {done}/{len(remaining)} items done")


def run_logprob_batched(items, backend, model_name, checkpoint,
                        output_path, done_ids, batch_size=BATCH_SIZE):
    """Run log-prob track with batched scoring calls."""
    remaining = [item for item in items if item["id"] not in done_ids]
    if not remaining:
        print("Log-prob: all items already scored")
        return
    print(f"Log-prob: {len(remaining)} items to score (batched, chunk={batch_size})")

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        # Collect all scoring triples for this batch
        # Each item has 1 baseline + N challenge pairs
        batch_triples = []   # (prefix, answer_a, answer_b)
        batch_layout = []    # (item_idx, "baseline"|challenge_dict)

        for idx, item in enumerate(batch):
            correct = item["correct_answer"]
            proposed = item.get("proposed_answer")
            if not proposed or "PLACEHOLDER" in proposed:
                continue

            # Baseline pair
            baseline_prefix = format_logprob_baseline_prompt(item["question"])
            batch_triples.append((baseline_prefix, f" {correct}", f" {proposed}"))
            batch_layout.append((idx, "baseline"))

            # Challenge pairs
            for challenge in item.get("challenges", []):
                ch_prefix = format_logprob_challenge_prompt(
                    question=item["question"],
                    initial_answer=correct,
                    challenge=challenge["prompt"],
                    context=challenge["context"],
                )
                batch_triples.append((ch_prefix, f" {correct}", f" {proposed}"))
                batch_layout.append((idx, challenge))

        if not batch_triples:
            continue

        # Batch score all pairs at once
        all_scores = backend.score_log_probs_pair_batch(batch_triples)

        # Reassemble per-item results
        from src.evaluation.logprob_scoring import compute_quality_flag
        vocab_size = backend.tokenizer.vocab_size

        item_results = {}  # idx → result dict
        score_idx = 0
        for (idx, tag), (correct_sc, incorrect_sc) in zip(batch_layout, all_scores):
            item = batch[idx]
            if tag == "baseline":
                log_odds = incorrect_sc["mean_log_prob"] - correct_sc["mean_log_prob"]
                quality = compute_quality_flag(correct_sc["mean_log_prob"], vocab_size)
                item_results[idx] = {
                    "question_id": item["id"],
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "baseline": {
                        "correct_answer": item["correct_answer"],
                        "proposed_answer": item.get("proposed_answer"),
                        "correct_log_prob": correct_sc["total_log_prob"],
                        "correct_mean_log_prob": correct_sc["mean_log_prob"],
                        "correct_num_tokens": correct_sc["num_tokens"],
                        "incorrect_log_prob": incorrect_sc["total_log_prob"],
                        "incorrect_mean_log_prob": incorrect_sc["mean_log_prob"],
                        "incorrect_num_tokens": incorrect_sc["num_tokens"],
                        "log_odds": log_odds,
                    },
                    "challenge_scores": [],
                    "quality": quality,
                    "_baseline_log_odds": log_odds,
                }
            else:
                # tag is the challenge dict
                if idx not in item_results:
                    continue
                baseline_log_odds = item_results[idx]["_baseline_log_odds"]
                challenged_log_odds = incorrect_sc["mean_log_prob"] - correct_sc["mean_log_prob"]
                delta = challenged_log_odds - baseline_log_odds
                item_results[idx]["challenge_scores"].append({
                    "challenge_id": tag["id"],
                    "challenge_type": tag.get("type"),
                    "challenge_context": tag.get("context"),
                    "correct_log_prob": correct_sc["total_log_prob"],
                    "correct_mean_log_prob": correct_sc["mean_log_prob"],
                    "correct_num_tokens": correct_sc["num_tokens"],
                    "incorrect_log_prob": incorrect_sc["total_log_prob"],
                    "incorrect_mean_log_prob": incorrect_sc["mean_log_prob"],
                    "incorrect_num_tokens": incorrect_sc["num_tokens"],
                    "log_odds": challenged_log_odds,
                    "delta_log_odds": delta,
                })

        # Save results (remove internal bookkeeping key)
        for idx in sorted(item_results):
            result = item_results[idx]
            result.pop("_baseline_log_odds", None)
            append_jsonl(result, output_path)

        done = batch_start + len(batch)
        print(f"  Log-prob: {done}/{len(remaining)} items done")


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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Items per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    batch_size = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    responses_path = os.path.join(args.output_dir, "responses.jsonl")
    logprob_path = os.path.join(args.output_dir, "logprob_scores.jsonl")

    # Load model ONCE
    backend = load_backend(args.backend_config)

    backend.max_new_tokens = 512

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

    print(f"Processing {len(items)} items with {args.model_name} (batch_size={batch_size})")

    # --- Generative track (batched) ---
    run_generative_batched(
        items, backend, args.model_type, args.model_name, args.checkpoint,
        responses_path, gen_done, batch_size=batch_size)

    # --- Log-prob track (batched) ---
    run_logprob_batched(
        items, backend, args.model_name, args.checkpoint,
        logprob_path, lp_done, batch_size=batch_size)

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
