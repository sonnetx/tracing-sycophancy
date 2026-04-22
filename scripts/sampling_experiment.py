#!/usr/bin/env python3
"""Sampling experiment for the decoding-time override claim.

Hypothesis: Post-training reduces preemptive-condition behavioural flip rate
at temperature 0 (greedy decoding) while the underlying preemptive
ΔLogOdds grows. If this is a decoding-time override of an amplified prior,
non-greedy sampling should surface more sycophantic flips because the
sampled distribution reflects the shifted probability mass.

Experimental design:
  - Restrict to preemptive challenges (where the override claim is strongest).
  - Use the existing temp=0 evaluated.jsonl to identify initially-correct items
    (so the denominator matches the main experiment).
  - For each (initially-correct item, preemptive challenge), sample N responses
    at each of several temperatures.
  - Score each sample with the same GPT-4o judge used in the main pipeline.
  - Report preemptive regressive rate per temperature per model.

Outputs:
  {output_dir}/sampling_generated.jsonl   — raw samples
  {output_dir}/sampling_evaluated.jsonl   — judge labels per sample
  {output_dir}/sampling_summary.json      — aggregate flip rates per temperature

Usage:
  python scripts/sampling_experiment.py \\
      --input data/processed/computational.jsonl \\
      --existing-eval data/results/exp1/computational/olmo3-7b-instruct/evaluated.jsonl \\
      --output-dir data/results/exp1_sampling/computational/olmo3-7b-instruct/ \\
      --backend-config /tmp/hf_config.json \\
      --model-type chat \\
      --model-name olmo3-7b-instruct \\
      --checkpoint instruct \\
      --temperatures 0.3 0.7 1.0 \\
      --n-samples 5
"""

import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from src.evaluation.judge import evaluate_challenge
from src.utils import (
    append_jsonl,
    format_challenge_prompt,
    load_backend,
    read_jsonl,
)


def load_initially_correct_with_responses(evaluated_path: str) -> tuple[set, dict]:
    """Return (set of initially-correct question_ids, {qid: initial_response}).

    Schema: row["initial"]["metrics"]["factual_accuracy"] (with a legacy
    fallback to row["initial"]["factual_accuracy"] for older files).
    The initial responses are needed for in-context sampling, where the
    challenge prompt includes the model's T=0 commitment as prior conversation.
    """
    correct = set()
    initials = {}
    for row in read_jsonl(evaluated_path):
        qid = row.get("question_id") or row.get("id")
        initial = row.get("initial") or {}
        metrics = initial.get("metrics") or {}
        fa = metrics.get("factual_accuracy") or initial.get("factual_accuracy")
        if fa == "correct":
            correct.add(qid)
            initials[qid] = initial.get("response", "")
    return correct, initials


def run_sampling(items, backend, model_type, model_name, checkpoint,
                  initially_correct, initial_responses,
                  temperatures, n_samples, contexts,
                  output_path, done_keys, batch_size=25):
    """Generate N samples per (initially-correct item, wrong-answer challenge,
    temperature) for each requested context in `contexts` (subset of
    {"preemptive", "in_context"}).

    For in-context tasks, the challenge prompt includes the model's T=0 initial
    response (from the main experiment's evaluated.jsonl) so the sampled flip
    rate is comparable to the original in-context measurement.

    done_keys is the set of (qid, cid, T, sample_idx) tuples already generated;
    the function skips those to support resume.
    """
    WRONG_TYPES = {"simple", "ethos", "justification", "citation"}
    tasks = []
    for item in items:
        if item["id"] not in initially_correct:
            continue
        for ch in item.get("challenges", []):
            if ch.get("context") not in contexts:
                continue
            if ch.get("type") not in WRONG_TYPES:
                continue   # skip correct/neutral controls
            tasks.append((item, ch))

    print(f"Initially-correct items filter: {len(initially_correct)} qids")
    print(f"Contexts to run: {sorted(contexts)}")
    print(f"(item, challenge) pairs: {len(tasks)}")
    if not tasks:
        print("[FATAL] No tasks to run.")
        print("  Check: evaluated.jsonl schema, challenge contexts, question_id matching")
        return

    for T in temperatures:
        pending = [(item, ch) for (item, ch) in tasks
                   if any((item["id"], ch["id"], T, s) not in done_keys
                          for s in range(n_samples))]
        if not pending:
            print(f"\n=== temperature={T} : all samples done, skipping ===")
            continue
        print(f"\n=== temperature={T} : {len(pending)} pairs to run ===")

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            prompts = []
            for item, ch in batch:
                # In-context: use model's T=0 initial response as prior commitment.
                # Preemptive: no prior commitment, use empty string.
                ctx = ch.get("context")
                prior = initial_responses.get(item["id"], "") if ctx == "in_context" else ""
                prompts.append(format_challenge_prompt(
                    question=item["question"],
                    initial_response=prior,
                    challenge=ch["prompt"],
                    context=ctx,
                    model_type=model_type,
                ))

            if model_type == "chat":
                samples_per_prompt = backend.chat_batch_sampling(
                    prompts, temperature=T, n=n_samples)
            else:
                samples_per_prompt = backend.complete_batch_sampling(
                    prompts, temperature=T, n=n_samples)

            for (item, ch), samples in zip(batch, samples_per_prompt):
                for sample_idx, response_text in enumerate(samples):
                    if (item["id"], ch["id"], T, sample_idx) in done_keys:
                        continue
                    append_jsonl({
                        "question_id": item["id"],
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "challenge_id": ch["id"],
                        "challenge_type": ch.get("type"),
                        "challenge_context": ch.get("context"),
                        "temperature": T,
                        "sample_idx": sample_idx,
                        "response": response_text,
                    }, output_path)

            print(f"  temp={T}: {batch_start + len(batch)}/{len(pending)} pairs")


def evaluate_all_samples(generated_path: str, items_by_id: dict,
                          judge, output_path: str, done_keys: set) -> None:
    """Score every sample with the existing GPT-4o judge."""
    rows = list(read_jsonl(generated_path))
    remaining = [r for r in rows
                 if (r["question_id"], r["challenge_id"], r["temperature"],
                     r["sample_idx"]) not in done_keys]
    print(f"\nJudge: evaluating {len(remaining)}/{len(rows)} samples")
    for row in tqdm(remaining):
        qid = row["question_id"]
        item = items_by_id.get(qid)
        if item is None:
            continue
        ch = next((c for c in item.get("challenges", [])
                   if c["id"] == row["challenge_id"]), None)
        if ch is None:
            continue
        result = evaluate_challenge(
            question=item["question"],
            correct_answer=item["correct_answer"],
            challenge_prompt=ch["prompt"],
            ai_response=row["response"],
            judge=judge,
        )
        row_out = dict(row)
        row_out["factual_accuracy"] = result["factual_accuracy"]
        append_jsonl(row_out, output_path)


def summarise(evaluated_path: str, summary_path: str) -> None:
    """Per-(context, temperature) regressive flip rate, averaged over samples
    and over (item, challenge) pairs. Regressive = factual_accuracy ==
    'incorrect' (denominator is initially-correct items, already filtered
    upstream)."""
    by = defaultdict(list)                   # (ctx, T) -> [flips]
    by_pair = defaultdict(lambda: defaultdict(list))   # (ctx, T) -> (qid, cid) -> [flips]
    for row in read_jsonl(evaluated_path):
        T = row["temperature"]
        ctx = row.get("challenge_context", "unknown")
        qid = row["question_id"]
        cid = row["challenge_id"]
        flipped = int(row.get("factual_accuracy") == "incorrect")
        by[(ctx, T)].append(flipped)
        by_pair[(ctx, T)][(qid, cid)].append(flipped)

    summary = {}
    for (ctx, T), flips in by.items():
        n = len(flips)
        flip_rate = sum(flips) / n if n else 0.0
        per_pair = [sum(v) / len(v) for v in by_pair[(ctx, T)].values()]
        avg_per_pair = sum(per_pair) / len(per_pair) if per_pair else 0.0
        summary.setdefault(ctx, {})[str(T)] = {
            "n_samples": n,
            "n_pairs": len(by_pair[(ctx, T)]),
            "flip_rate_all_samples": flip_rate,
            "flip_rate_avg_per_pair": avg_per_pair,
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for ctx in sorted(summary.keys()):
        print(f"  [{ctx}]")
        for T, s in sorted(summary[ctx].items(), key=lambda kv: float(kv[0])):
            print(f"    T={T}  pairs={s['n_pairs']:4d}  samples={s['n_samples']:5d}  "
                  f"flip_rate(all)={s['flip_rate_all_samples']:.3f}  "
                  f"flip_rate(per-pair avg)={s['flip_rate_avg_per_pair']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="processed/{dataset}.jsonl")
    parser.add_argument("--existing-eval", required=True,
                        help="Existing evaluated.jsonl from temp=0 experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--model-type", choices=["base", "chat"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.3, 0.7, 1.0])
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--contexts", nargs="+",
                        default=["preemptive"],
                        choices=["preemptive", "in_context"],
                        help="Which challenge contexts to sample. "
                             "Default preemptive only.")
    parser.add_argument("--judge-config", default=None,
                        help="Backend config for judge (defaults to OpenAI env-based)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max generation length per sample. Raise for CoT models "
                             "(Think pipeline needs >= 4096 to avoid truncation).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generated_path = os.path.join(args.output_dir, "sampling_generated.jsonl")
    evaluated_path = os.path.join(args.output_dir, "sampling_evaluated.jsonl")
    summary_path   = os.path.join(args.output_dir, "sampling_summary.json")

    if not os.path.isfile(args.existing_eval):
        raise SystemExit(f"[FATAL] existing-eval not found: {args.existing_eval}")
    if not os.path.isfile(args.input):
        raise SystemExit(f"[FATAL] input processed file not found: {args.input}")

    items = list(read_jsonl(args.input))
    items_by_id = {it["id"]: it for it in items}
    initially_correct, initial_responses = load_initially_correct_with_responses(
        args.existing_eval)
    n_eval_rows = sum(1 for _ in read_jsonl(args.existing_eval))
    print(f"Existing eval rows: {n_eval_rows}")
    print(f"Initially correct items (temp=0 baseline): {len(initially_correct)}")
    if "in_context" in args.contexts:
        n_with_initial = sum(1 for qid in initially_correct
                             if initial_responses.get(qid))
        print(f"  with T=0 initial response available: {n_with_initial}")

    # Resume: skip (qid, cid, T, sample_idx) tuples already generated
    done_gen_keys = set()
    if os.path.exists(generated_path):
        for row in read_jsonl(generated_path):
            done_gen_keys.add((row["question_id"], row["challenge_id"],
                               row["temperature"], row["sample_idx"]))
        print(f"Resume: {len(done_gen_keys)} samples already in {generated_path}")

    backend = load_backend(args.backend_config)
    backend.max_new_tokens = args.max_new_tokens
    run_sampling(items, backend, args.model_type, args.model_name,
                 args.checkpoint, initially_correct, initial_responses,
                 args.temperatures, args.n_samples, set(args.contexts),
                 generated_path, done_gen_keys)

    if not os.path.isfile(generated_path):
        raise SystemExit(
            f"[FATAL] No samples generated; {generated_path} does not exist.\n"
            f"  Existing eval rows loaded: {n_eval_rows}\n"
            f"  Initially correct items found: {len(initially_correct)}\n"
            f"  Check that {args.existing_eval} contains rows with "
            f"initial.factual_accuracy == 'correct' and matching question_ids."
        )

    # Load GPT-4o judge (separate backend; judge_config points at OpenAI config)
    judge_cfg = args.judge_config or args.backend_config
    judge = load_backend(judge_cfg)

    done_eval_keys = set()
    if os.path.exists(evaluated_path):
        for row in read_jsonl(evaluated_path):
            done_eval_keys.add((row["question_id"], row["challenge_id"],
                                row["temperature"], row["sample_idx"]))

    evaluate_all_samples(generated_path, items_by_id, judge, evaluated_path,
                          done_eval_keys)

    summarise(evaluated_path, summary_path)


if __name__ == "__main__":
    main()
