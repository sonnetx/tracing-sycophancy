#!/usr/bin/env python3
"""Extract top-K next-token log-probs at the first generation step under a
challenge, for a selected set of items. Visualizes the model's actual
probability distribution at the decision point — directly showing mode
sharpness per pipeline on the same items.

For each (question_id, challenge_id) the script:
  1. Reconstructs the challenge prompt (preemptive or in-context) with
     format_challenge_prompt.
  2. Runs vLLM with SamplingParams(max_tokens=1, logprobs=K, temperature=0).
  3. Extracts the K highest-probability next-token candidates with their
     log-probabilities.

Output: JSONL at --output-path with one row per (qid, cid) containing:
  {model, question_id, challenge_id, challenge_type, challenge_context,
   question, correct_answer, proposed_wrong_answer, prompt_repr,
   topk: [{token, logprob}, ...]}

Usage (on Sherlock, inside apptainer):
    python scripts/extract_next_token_logits.py \\
        --input data/processed/computational.jsonl \\
        --existing-eval data/results/exp1/computational/olmo3-7b-instruct/evaluated.jsonl \\
        --backend-config config/models/olmo3-7b-instruct.json \\
        --model-type chat --model-name olmo3-7b-instruct \\
        --question-ids comp_042 comp_105 comp_217 \\
        --challenge-ids ethos_preemptive citation_preemptive \\
        --topk 25 \\
        --output-path data/results/logits_probe/olmo3-7b-instruct/comp.jsonl
"""

import argparse
import json
import os
import sys

from src.utils import (
    format_challenge_prompt,
    read_jsonl,
    write_jsonl,
)


def load_backend_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_items(processed_path: str, question_ids: list[str]) -> dict:
    items = {}
    qids_set = set(question_ids)
    for item in read_jsonl(processed_path):
        qid = item.get("id") or item.get("question_id")
        if qid in qids_set:
            items[qid] = item
    missing = qids_set - set(items.keys())
    if missing:
        print(f"WARNING: question_ids not found in input: {sorted(missing)}")
    return items


def load_existing_eval(eval_path: str, qids: set) -> dict:
    """Return {qid: initial_response} for questions in qids."""
    out = {}
    if not eval_path or not os.path.exists(eval_path):
        return out
    for item in read_jsonl(eval_path):
        qid = item.get("question_id")
        if qid in qids:
            out[qid] = item.get("initial", {}).get("response", "")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="processed JSONL with question+challenges")
    parser.add_argument("--existing-eval",
                        help="existing evaluated.jsonl (to pull initial responses for IC)")
    parser.add_argument("--backend-config", required=True,
                        help="JSON file describing vLLM backend")
    parser.add_argument("--model-type", default="chat", choices=["chat", "base"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--question-ids", nargs="+", required=True)
    parser.add_argument("--challenge-ids", nargs="+", required=True,
                        help="e.g. ethos_preemptive citation_preemptive")
    parser.add_argument("--topk", type=int, default=20,
                        help="max 20 on vLLM v1 default config")
    parser.add_argument("--prompt-suffix", default="",
                        help=("raw string appended verbatim after the chat "
                              "template's generation prompt. Use to force the "
                              "model to a specific decision point. Example: "
                              r"'\nFinal answer: $' will make the next token "
                              "be the digit/symbol immediately following the '$'."))
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    # Lazy-load vLLM (heavy) only after arg parsing
    from vllm import LLM, SamplingParams

    cfg = load_backend_config(args.backend_config)
    items = load_items(args.input, args.question_ids)
    initials = load_existing_eval(args.existing_eval, set(items.keys()))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    kwargs = {
        "model": cfg["model"],
        "dtype": cfg.get("torch_dtype", "bfloat16"),
        "max_model_len": cfg.get("max_model_len", 8192),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.80),
    }
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    # temperature=1.0 so returned logprobs are unscaled (post-softmax of raw
    # logits). Temperature sweep is applied downstream in the plot by
    # recomputing softmax(logp / T).
    params = SamplingParams(max_tokens=1, logprobs=args.topk, temperature=1.0)

    rows_out = []
    for qid in args.question_ids:
        if qid not in items:
            continue
        item = items[qid]
        for cid in args.challenge_ids:
            ch = next((c for c in item.get("challenges", []) if c["id"] == cid), None)
            if ch is None:
                print(f"  skipping {qid}/{cid}: not found")
                continue
            ctx = ch.get("context", "preemptive")
            prompt_obj = format_challenge_prompt(
                question=item["question"],
                initial_response=initials.get(qid, ""),
                challenge=ch["prompt"],
                context=ctx,
                model_type=args.model_type,
            )
            # Render to token stream using vLLM directly
            if args.model_type == "chat":
                prompt_text = tokenizer.apply_chat_template(
                    prompt_obj, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = prompt_obj
            # Append the forcing suffix so the next-token distribution is at
            # the canonical decision point rather than at the preamble.
            if args.prompt_suffix:
                prompt_text = prompt_text + args.prompt_suffix
            outputs = llm.generate([prompt_text], params)
            topk = []
            if outputs and outputs[0].outputs:
                first_out = outputs[0].outputs[0]
                if first_out.logprobs and first_out.logprobs[0]:
                    # vLLM returns Dict[int, Logprob]; rank by logprob desc
                    lp_dict = first_out.logprobs[0]
                    ranked = sorted(lp_dict.items(), key=lambda kv: -kv[1].logprob)
                    for tok_id, lp in ranked[: args.topk]:
                        tok_str = lp.decoded_token if lp.decoded_token is not None \
                                  else tokenizer.decode([tok_id])
                        topk.append({"token": tok_str, "token_id": int(tok_id),
                                     "logprob": float(lp.logprob)})

            rows_out.append({
                "model": args.model_name,
                "question_id": qid,
                "challenge_id": cid,
                "challenge_type": ch.get("type"),
                "challenge_context": ctx,
                "question": item["question"],
                "correct_answer": item.get("correct_answer"),
                "proposed_wrong_answer": item.get("proposed_answer"),
                "prompt_text": prompt_text,
                "prompt_suffix": args.prompt_suffix,
                "topk": topk,
            })
            print(f"  {qid}/{cid}: top-3 = "
                  + ", ".join(f"{t['token']!r}({t['logprob']:.2f})" for t in topk[:3]))

    write_jsonl(rows_out, args.output_path)
    print(f"\nWrote {len(rows_out)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
