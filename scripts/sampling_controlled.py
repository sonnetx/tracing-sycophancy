#!/usr/bin/env python3
"""Controlled sampling run that holds items, budget, and prompts fixed to the main greedy run.

Samples the preemptive wrong-answer challenges (ethos, justification, citation) and the
bare-question neutral baseline at one temperature, on the items the main greedy run answered
correctly, with the same generation cap. Each draw records its finish reason and generated-token
count, then is scored by the same GPT-4o judge as the main pipeline.

Outputs in --output-dir:
  sampling_generated.jsonl  one row per draw (response, finish_reason, n_tokens)
  sampling_evaluated.jsonl  the same rows plus the judge's factual_accuracy label
"""

import argparse
import json
import os

from vllm import SamplingParams

from src.evaluation.judge import evaluate_challenge
from src.utils import append_jsonl, format_challenge_prompt, load_backend, read_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="processed/{dataset}.jsonl")
    p.add_argument("--existing-eval", required=True,
                   help="Main greedy evaluated.jsonl that defines the initially correct items")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend-config", required=True)
    p.add_argument("--judge-config", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--types", nargs="+", default=["ethos", "justification", "citation", "neutral"])
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    gen_path = os.path.join(a.output_dir, "sampling_generated.jsonl")
    eval_path = os.path.join(a.output_dir, "sampling_evaluated.jsonl")

    items = {it["id"]: it for it in read_jsonl(a.input)}
    eligible = [r["question_id"] for r in read_jsonl(a.existing_eval)
                if (r["initial"].get("metrics") or {}).get("factual_accuracy") == "correct"]
    tasks = [(items[q], ch) for q in eligible if q in items for ch in items[q].get("challenges", [])
             if ch.get("context") == "preemptive" and ch.get("type") in a.types]

    # Resume at the prompt level: a prompt counts as done once all its draws are written.
    counts = {}
    if os.path.exists(gen_path):
        for r in read_jsonl(gen_path):
            k = (r["question_id"], r["challenge_id"])
            counts[k] = counts.get(k, 0) + 1
    pending = [(it, ch) for it, ch in tasks if counts.get((it["id"], ch["id"]), 0) < a.n_samples]
    print(f"eligible items={len(eligible)} prompts={len(tasks)} pending={len(pending)}", flush=True)

    if pending:
        backend = load_backend(a.backend_config)
        params = SamplingParams(max_tokens=a.max_new_tokens, temperature=a.temperature,
                                top_p=1.0, n=a.n_samples)
        for start in range(0, len(pending), a.batch_size):
            batch = pending[start:start + a.batch_size]
            prompts = [backend._apply_chat_template(format_challenge_prompt(
                question=it["question"], initial_response="", challenge=ch["prompt"],
                context="preemptive", model_type="chat")) for it, ch in batch]
            for (it, ch), req in zip(batch, backend.llm.generate(prompts, params)):
                if counts.get((it["id"], ch["id"]), 0):
                    continue   # partially written before an interruption; keep the earlier draws
                for k, o in enumerate(req.outputs):
                    append_jsonl({
                        "question_id": it["id"], "model": a.model_name,
                        "challenge_id": ch["id"], "challenge_type": ch.get("type"),
                        "challenge_context": "preemptive", "temperature": a.temperature,
                        "sample_idx": k, "max_new_tokens": a.max_new_tokens,
                        "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids),
                        "response": o.text.strip(),
                    }, gen_path)
            print(f"  generated {start + len(batch)}/{len(pending)} prompts", flush=True)

    judged = set()
    if os.path.exists(eval_path):
        judged = {(r["question_id"], r["challenge_id"], r["sample_idx"]) for r in read_jsonl(eval_path)}
    rows = [r for r in read_jsonl(gen_path)
            if (r["question_id"], r["challenge_id"], r["sample_idx"]) not in judged]
    print(f"judging {len(rows)} draws", flush=True)
    judge = load_backend(a.judge_config)
    by_id = {(q, c["id"]): c for q, it in items.items() for c in it.get("challenges", [])}
    for i, r in enumerate(rows):
        it = items[r["question_id"]]
        res = evaluate_challenge(question=it["question"], correct_answer=it["correct_answer"],
                                 challenge_prompt=by_id[(r["question_id"], r["challenge_id"])]["prompt"],
                                 ai_response=r["response"], judge=judge)
        append_jsonl({**r, "factual_accuracy": res["factual_accuracy"]}, eval_path)
        if i % 1000 == 0:
            print(f"  judged {i}/{len(rows)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
