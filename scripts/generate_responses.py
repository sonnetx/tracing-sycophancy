#!/usr/bin/env python3
"""Step 3: Generate model responses to questions and challenges.

This script runs inference for a single model on a preprocessed dataset.
It generates both initial responses and challenge responses.

Usage:
    # Chat model (instruct)
    python scripts/generate_responses.py \
        --input data/processed/computational.jsonl \
        --output data/results/exp1/llama_instruct/responses.jsonl \
        --backend-config config/models/llama_instruct.json \
        --model-type chat \
        --model-name llama-3.1-8b-instruct \
        --checkpoint instruct

    # Base model (completion)
    python scripts/generate_responses.py \
        --input data/processed/computational.jsonl \
        --output data/results/exp1/llama_base/responses.jsonl \
        --backend-config config/models/llama_base.json \
        --model-type base \
        --model-name llama-3.1-8b-base \
        --checkpoint base
"""

import argparse

from tqdm import tqdm

from src.utils import (
    append_jsonl,
    format_challenge_prompt,
    format_initial_prompt,
    load_backend,
    read_jsonl,
)


def generate_for_item(item: dict, backend, model_type: str, model_name: str, checkpoint: str) -> dict:
    """Generate initial and challenge responses for a single question."""
    question = item["question"]

    # --- Initial response ---
    prompt_or_messages = format_initial_prompt(question, model_type)
    if model_type == "chat":
        initial_response = backend.chat(prompt_or_messages)
    else:
        initial_response = backend.complete(prompt_or_messages)

    result = {
        "question_id": item["id"],
        "model": model_name,
        "checkpoint": checkpoint,
        "model_type": model_type,
        "initial": {
            "response": initial_response,
            "prompt_format": "chat" if model_type == "chat" else "completion",
        },
        "challenge_responses": [],
    }

    # --- Challenge responses ---
    challenges = item.get("challenges", [])
    for challenge in challenges:
        challenge_prompt = format_challenge_prompt(
            question=question,
            initial_response=initial_response,
            challenge=challenge["prompt"],
            context=challenge["context"],
            model_type=model_type,
        )

        if model_type == "chat":
            challenge_response = backend.chat(challenge_prompt)
            fmt = "chat"
        else:
            challenge_response = backend.complete(challenge_prompt)
            fmt = "completion_concatenated" if challenge["context"] == "in_context" else "completion"

        result["challenge_responses"].append({
            "challenge_id": challenge["id"],
            "response": challenge_response,
            "prompt_format": fmt,
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for sycophancy evaluation")
    parser.add_argument("--input", required=True, help="Input JSONL file (preprocessed questions with challenges)")
    parser.add_argument("--output", required=True, help="Output JSONL file for responses")
    parser.add_argument("--backend-config", required=True, help="Path to model backend config JSON")
    parser.add_argument("--model-type", required=True, choices=["chat", "base"],
                        help="Model type: 'chat' for instruct models, 'base' for completion models")
    parser.add_argument("--model-name", required=True, help="Human-readable model name for results")
    parser.add_argument("--checkpoint", default="default", help="Checkpoint identifier (e.g., 'base', 'instruct', 'step1000')")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-processed items")
    args = parser.parse_args()

    backend = load_backend(args.backend_config)
    items = read_jsonl(args.input)

    # Handle resume: load existing results and skip processed IDs
    processed_ids = set()
    if args.resume:
        try:
            existing = read_jsonl(args.output)
            processed_ids = {r["question_id"] for r in existing}
            print(f"Resuming: {len(processed_ids)} items already processed")
        except FileNotFoundError:
            pass

    remaining = [item for item in items if item["id"] not in processed_ids]
    print(f"Processing {len(remaining)} items ({len(processed_ids)} skipped)")

    for item in tqdm(remaining, desc=f"Generating responses ({args.model_name})"):
        try:
            result = generate_for_item(item, backend, args.model_type, args.model_name, args.checkpoint)
            append_jsonl(result, args.output)
        except Exception as e:
            print(f"\nError processing {item['id']}: {e}")
            continue


if __name__ == "__main__":
    main()
