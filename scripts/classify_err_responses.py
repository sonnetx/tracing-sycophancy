#!/usr/bin/env python3
"""Sub-classify erroneous challenge responses as apology-capitulation, format-incoherence, etc.

Filters to items where the initial response was correct and the challenge response
was labelled 'erroneous' by the factual-accuracy judge (i.e., the Err. metric
numerator). Skips auto-propagated skips (initial was erroneous) and, by default,
base model checkpoints (whose Err. is format incoherence by design).

Usage:
    python scripts/classify_err_responses.py \\
        --evaluated data/results/exp1/llama_instruct/evaluated.jsonl \\
                    data/results/exp1/olmo_think_sft/evaluated.jsonl \\
        --questions data/processed/medical.jsonl \\
        --output data/results/err_subclassified_medical.jsonl \\
        --judge-config config/models/gpt4o_judge.json \\
        --domain medical
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.err_subclassifier import classify_err_response
from src.utils import append_jsonl, load_backend, read_jsonl

BASE_CHECKPOINTS = {"base", "pretrain", "pretraining"}


def _parse_challenge_id(challenge_id: str) -> tuple[str, str]:
    """Split 'simple_incontext' → ('simple', 'in_context')."""
    parts = challenge_id.rsplit("_", 1)
    if len(parts) == 2:
        challenge_type, raw_context = parts
        context = "in_context" if raw_context == "incontext" else raw_context
        return challenge_type, context
    return challenge_id, "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Sub-classify Err. challenge responses via LLM judge"
    )
    parser.add_argument("--evaluated", nargs="+", required=True,
                        help="One or more evaluated.jsonl file paths")
    parser.add_argument("--questions", required=True,
                        help="Preprocessed questions JSONL (with challenge texts)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for sub-classifications")
    parser.add_argument("--judge-config", required=True,
                        help="Path to judge backend config JSON")
    parser.add_argument("--domain", required=True, choices=["computational", "medical"],
                        help="Dataset domain label written to output records")
    parser.add_argument("--include-base", action="store_true",
                        help="Include base model checkpoints (excluded by default)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, overwriting any existing output file")
    args = parser.parse_args()

    subclassifier = load_backend(args.judge_config)

    questions = read_jsonl(args.questions)
    question_lookup = {q["id"]: q for q in questions}
    challenge_text_lookup: dict[str, dict[str, str]] = {
        q["id"]: {c["id"]: c.get("prompt", "") for c in q.get("challenges", [])}
        for q in questions
    }

    # Resume: track already-classified (question_id, model, checkpoint, challenge_id)
    output_path = Path(args.output)
    done_keys: set[tuple] = set()
    if not args.no_resume and output_path.exists():
        for rec in read_jsonl(output_path):
            done_keys.add((rec["question_id"], rec["model"], rec["checkpoint"], rec["challenge_id"]))
        print(f"Resume: {len(done_keys)} items already classified, skipping")
    elif args.no_resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    total = 0
    skipped_resume = 0
    skipped_filter = 0

    for eval_path in args.evaluated:
        items = read_jsonl(eval_path)
        label = Path(eval_path).parent.name
        for item in tqdm(items, desc=f"{label}"):
            question_id = item["question_id"]
            model = item["model"]
            checkpoint = item["checkpoint"]
            model_type = item.get("model_type", "unknown")

            if not args.include_base and checkpoint.lower() in BASE_CHECKPOINTS:
                skipped_filter += 1
                continue

            if item["initial"]["metrics"].get("factual_accuracy") != "correct":
                skipped_filter += 1
                continue

            question_item = question_lookup.get(question_id)
            if question_item is None:
                continue

            question_text = question_item["question"]
            initial_response = item["initial"]["response"]
            challenges_for_q = challenge_text_lookup.get(question_id, {})

            for cr in item.get("challenge_responses", []):
                challenge_id = cr["challenge_id"]
                cr_metrics = cr.get("metrics", {})

                if cr_metrics.get("factual_accuracy") != "erroneous":
                    continue
                if cr_metrics.get("skipped"):
                    continue

                key = (question_id, model, checkpoint, challenge_id)
                if key in done_keys:
                    skipped_resume += 1
                    continue

                challenge_text = challenges_for_q.get(challenge_id, "")
                challenge_response = cr["response"]
                challenge_type, challenge_context = _parse_challenge_id(challenge_id)

                try:
                    result = classify_err_response(
                        question=question_text,
                        initial_response=initial_response,
                        challenge_text=challenge_text,
                        challenge_response=challenge_response,
                        subclassifier=subclassifier,
                    )
                except Exception as e:
                    print(f"\nError classifying {question_id}/{challenge_id}: {e}")
                    continue

                record = {
                    "question_id": question_id,
                    "model": model,
                    "checkpoint": checkpoint,
                    "model_type": model_type,
                    "domain": args.domain,
                    "challenge_id": challenge_id,
                    "challenge_type": challenge_type,
                    "challenge_context": challenge_context,
                    "challenge_response": challenge_response,
                    "err_subtype": result["err_subtype"],
                }
                append_jsonl(record, output_path)
                done_keys.add(key)
                total += 1

    print(
        f"Classified {total} Err. responses"
        + (f" ({skipped_resume} skipped via resume)" if skipped_resume else "")
        + (f" ({skipped_filter} skipped by filter)" if skipped_filter else "")
        + f" → {args.output}"
    )


if __name__ == "__main__":
    main()
