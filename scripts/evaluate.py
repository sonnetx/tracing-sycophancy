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

from src.evaluation.hedging import detect_hedging, detect_hedging_auto
from src.evaluation.judge import evaluate_challenge, evaluate_initial
from src.evaluation.refusal import detect_refusal
from src.utils import append_jsonl, load_backend, read_jsonl, write_jsonl


def evaluate_item(response_item: dict, question_item: dict, judge,
                   hedging_judge=None) -> dict:
    """Evaluate all responses for a single question/model combination."""
    question = question_item["question"]
    correct_answer = question_item["correct_answer"]

    # Build challenge lookup
    challenge_lookup = {c["id"]: c for c in question_item.get("challenges", [])}

    # --- Evaluate initial response ---
    initial_response = response_item["initial"]["response"]
    initial_refusal = detect_refusal(initial_response)
    initial_metrics = evaluate_initial(question, correct_answer, initial_response, judge)
    initial_metrics.update(detect_hedging_auto(initial_response, judge=hedging_judge))
    initial_metrics.update(initial_refusal)

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

    skip_challenges = (
        initial_refusal.get("refusal")
        or initial_metrics.get("factual_accuracy") == "erroneous"
    )

    # --- Evaluate challenge responses ---
    for cr in response_item.get("challenge_responses", []):
        challenge_id = cr["challenge_id"]
        challenge_info = challenge_lookup.get(challenge_id, {})
        challenge_prompt = challenge_info.get("prompt", "")
        cr_response = cr["response"]

        if skip_challenges:
            cr_metrics = {
                "factual_accuracy": "erroneous",
                "agreement": None,
                "skipped": True,
                "skip_reason": "initial_response_erroneous_or_refused",
            }
            cr_metrics.update(detect_hedging_auto(cr_response, challenge_prompt=challenge_prompt))
            cr_metrics.update(detect_refusal(cr_response))
        else:
            cr_metrics = evaluate_challenge(question, correct_answer, challenge_prompt, cr_response, judge)
            cr_metrics.update(detect_hedging_auto(cr_response, challenge_prompt=challenge_prompt,
                                                  judge=hedging_judge))
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
    parser.add_argument("--hedging-method", choices=["keyword", "llm"], default="keyword",
                        help="Hedging detection method: 'keyword' (default) or 'llm' (uses judge)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, overwriting any existing output file")
    args = parser.parse_args()

    judge = load_backend(args.judge_config)
    hedging_judge = judge if args.hedging_method == "llm" else None
    if hedging_judge:
        print(f"INFO: Using LLM judge for hedging detection")
    responses = read_jsonl(args.input)
    questions = read_jsonl(args.questions)

    # Build question lookup by ID
    question_lookup = {q["id"]: q for q in questions}

    # Resume by default: load already-evaluated IDs and append new results
    done_ids = set()
    if not args.no_resume:
        try:
            done_ids = {r["question_id"] for r in read_jsonl(args.output)}
            print(f"Resume: {len(done_ids)} items already evaluated, skipping")
        except FileNotFoundError:
            pass
    else:
        # Fresh run: truncate output file so append_jsonl starts clean
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        open(args.output, "w").close()

    count = 0
    skipped = 0
    for response_item in tqdm(responses, desc="Evaluating responses"):
        qid = response_item["question_id"]

        if qid in done_ids:
            skipped += 1
            continue

        question_item = question_lookup.get(qid)
        if question_item is None:
            print(f"Warning: no question found for {qid}, skipping")
            continue

        try:
            result = evaluate_item(response_item, question_item, judge,
                                   hedging_judge=hedging_judge)
            # Write incrementally so partial progress survives crashes
            append_jsonl(result, args.output)
            count += 1
        except Exception as e:
            print(f"\nError evaluating {qid}: {e}")
            continue

    print(f"Wrote {count} evaluated items to {args.output}"
          + (f" ({skipped} skipped via resume)" if skipped else ""))


if __name__ == "__main__":
    main()
