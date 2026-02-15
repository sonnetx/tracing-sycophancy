#!/usr/bin/env python3
"""Step 4: Evaluate model responses — compute factual accuracy, agreement, hedging, and refusal.

Usage:
    python scripts/evaluate.py \
        --input data/results/exp1/llama_base/responses.jsonl \
        --questions data/processed/computational.jsonl \
        --output data/results/exp1/llama_base/evaluated.jsonl \
        --judge-config config/models/gpt4o_judge.json
"""

import argparse

from tqdm import tqdm

from src.evaluation.hedging import detect_hedging
from src.evaluation.judge import evaluate_challenge, evaluate_initial
from src.evaluation.refusal import detect_refusal
from src.utils import load_backend, read_jsonl, write_jsonl


def evaluate_item(response_item: dict, question_item: dict, judge) -> dict:
    """Evaluate all responses for a single question/model combination."""
    question = question_item["question"]
    correct_answer = question_item["correct_answer"]

    # Build challenge lookup
    challenge_lookup = {c["id"]: c for c in question_item.get("challenges", [])}

    # --- Evaluate initial response ---
    initial_response = response_item["initial"]["response"]
    initial_metrics = evaluate_initial(question, correct_answer, initial_response, judge)
    initial_metrics.update(detect_hedging(initial_response))
    initial_metrics.update(detect_refusal(initial_response))

    result = {
        "question_id": response_item["question_id"],
        "model": response_item["model"],
        "checkpoint": response_item["checkpoint"],
        "model_type": response_item.get("model_type", "unknown"),
        "initial": {
            "response": initial_response,
            "prompt_format": response_item["initial"].get("prompt_format"),
            "metrics": initial_metrics,
        },
        "challenge_responses": [],
    }

    # --- Evaluate challenge responses ---
    for cr in response_item.get("challenge_responses", []):
        challenge_id = cr["challenge_id"]
        challenge_info = challenge_lookup.get(challenge_id, {})
        challenge_prompt = challenge_info.get("prompt", "")
        cr_response = cr["response"]

        cr_metrics = evaluate_challenge(question, correct_answer, challenge_prompt, cr_response, judge)
        cr_metrics.update(detect_hedging(cr_response))
        cr_metrics.update(detect_refusal(cr_response))

        result["challenge_responses"].append({
            "challenge_id": challenge_id,
            "response": cr_response,
            "prompt_format": cr.get("prompt_format"),
            "metrics": cr_metrics,
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses for sycophancy metrics")
    parser.add_argument("--input", required=True, help="Input JSONL file (model responses)")
    parser.add_argument("--questions", required=True, help="Preprocessed questions JSONL (with challenges)")
    parser.add_argument("--output", required=True, help="Output JSONL file (evaluated responses)")
    parser.add_argument("--judge-config", required=True, help="Path to judge backend config JSON")
    args = parser.parse_args()

    judge = load_backend(args.judge_config)
    responses = read_jsonl(args.input)
    questions = read_jsonl(args.questions)

    # Build question lookup by ID
    question_lookup = {q["id"]: q for q in questions}

    results = []
    for response_item in tqdm(responses, desc="Evaluating responses"):
        qid = response_item["question_id"]
        question_item = question_lookup.get(qid)
        if question_item is None:
            print(f"Warning: no question found for {qid}, skipping")
            continue

        try:
            result = evaluate_item(response_item, question_item, judge)
            results.append(result)
        except Exception as e:
            print(f"\nError evaluating {qid}: {e}")
            continue

    write_jsonl(results, args.output)
    print(f"Wrote {len(results)} evaluated items to {args.output}")


if __name__ == "__main__":
    main()
