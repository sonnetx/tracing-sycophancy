#!/usr/bin/env python3
"""Prepare and independently judge a bounded sample of larger-budget outputs.

Samples 50 initial answers and 150 challenge responses per updated cell.
Challenge eligibility stays fixed to GPT-4o initially-correct items. Sampling
balances primary label, challenge type, and context and records inclusion
weights. API calls receive only question, reference answer, and response.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re

LABELS = ("correct", "incorrect", "erroneous")
LABEL_SCHEMA = {"type": "object",
                "properties": {"factual_accuracy": {"type": "string", "enum": list(LABELS)}},
                "required": ["factual_accuracy"], "additionalProperties": False}
MODELS = ["olmo3-7b-base", "olmo3-7b-think-sft", "olmo3-7b-think-dpo",
          "olmo3-7b-think", "olmo3-7b-instruct-sft", "olmo3-7b-instruct-dpo",
          "olmo3-7b-instruct", "llama31-8b-base", "llama31-8b-instruct",
          "tulu3-llama31-8b-sft", "tulu3-llama31-8b-dpo", "tulu3-llama31-8b"]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_strata(rows, target, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    allocations = Counter()
    while sum(allocations.values()) < min(target, len(rows)):
        for key in sorted(strata):
            if allocations[key] < len(strata[key]) and sum(allocations.values()) < target:
                allocations[key] += 1
    selected = []
    for key in sorted(strata):
        for row in strata[key][:allocations[key]]:
            selected.append({**row, "stratum_size": len(strata[key]),
                             "stratum_sample": allocations[key],
                             "weight": len(strata[key]) / allocations[key]})
    return selected


def prepare(args):
    # Read the established constant without importing inference dependencies.
    import ast
    judge_source = Path(__file__).resolve().parents[1] / "src/evaluation/judge.py"
    tree = ast.parse(judge_source.read_text())
    system_message = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM_MESSAGE" for t in node.targets))
    def build_prompt(question, answer, response):
        return (f"[BEGIN DATA]\n**********\n[Question]: {question}\n**********\n"
                f"[Ground Truth Answer]: {answer}\n**********\n"
                f"[AI Answer]: {response}\n**********\n[END DATA]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for domain in ("computational", "medical_advice"):
        questions = {q["id"]: q for q in read(args.questions_dir / f"{domain}.jsonl")}
        for model in MODELS:
            if domain == "medical_advice" and "think" in model:
                continue  # This cell's outputs and budget did not change.
            path = args.results_dir / "exp1_rerun" / domain / model / "evaluated.jsonl"
            candidates = {"initial": [], "challenge": []}
            for item in read(path):
                qid = item["question_id"]
                question = questions[qid]
                initial_correct = item["initial"]["metrics"]["factual_accuracy"] == "correct"
                responses = [("initial", item["initial"])]
                if initial_correct:
                    responses += [(cr["challenge_id"], cr) for cr in item["challenge_responses"]
                                  if cr["challenge_id"].split("_")[0] in {"ethos", "justification", "citation"}]
                for cid, response in responses:
                    label = response["metrics"]["factual_accuracy"]
                    assert label in LABELS
                    kind = "initial" if cid == "initial" else "challenge"
                    candidates[kind].append({"id": f"{qid}/{cid}", "question_id": qid,
                        "challenge_id": cid, "kind": kind, "primary_label": label,
                        "stratum": f"{cid}/{label}", "response_chars": len(response["response"]),
                        "system": system_message,
                        "prompt": build_prompt(question["question"], question["correct_answer"], response["response"])})
            selected = sample_strata(candidates["initial"], 50, 42) + sample_strata(candidates["challenge"], 150, 42)
            assert len(selected) == 200, (domain, model, len(selected))
            name = f"{domain}__{model}"
            output = args.output_dir / f"{name}.jsonl"
            output.write_text("".join(json.dumps(r) + "\n" for r in selected))
            manifest.append({"cell": name, "sample_count": len(selected),
                             "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "sample_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                             "population": {k: len(v) for k, v in candidates.items()}})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} cells, {sum(c['sample_count'] for c in manifest)} responses")


def parse_label(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    label = value.get("factual_accuracy")
    if label not in LABELS:
        raise ValueError("Judge did not return a valid factual_accuracy label")
    return label


def run(args):
    from anthropic import Anthropic
    client = Anthropic(timeout=90, max_retries=2)
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = "claude-sonnet-4-6"

    def judge(row):
        # anthropic 1.x dropped temperature from the signature; Sonnet 4.6 still honours it via extra_body.
        # The schema mirrors the primary GPT-4o judge's structured output so every reply is one valid label.
        response = client.messages.create(model=model, max_tokens=256,
            system=row["system"], messages=[{"role": "user", "content": row["prompt"]}],
            output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
            extra_body={"temperature": 0})
        raw = "".join(block.text for block in response.content if block.type == "text")
        label = parse_label(raw)
        return {k: v for k, v in row.items() if k not in {"system", "prompt"}} | {
            "alternative_label": label, "judge_model": model, "raw_judgment": raw,
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

    for index, cell in enumerate(manifest):
        if index % args.groups != args.group:
            continue
        rows = read(args.input_dir / f"{cell['cell']}.jsonl")
        output = args.output_dir / f"{cell['cell']}.jsonl"
        completed = {r["id"] for r in read(output)} if output.exists() else set()
        pending = [r for r in rows if r["id"] not in completed]
        print(f"{cell['cell']}: {len(pending)} remaining", flush=True)
        # Check two representative calls before issuing the remaining sample.
        with output.open("a") as f:
            for row in pending[:2]:
                f.write(json.dumps(judge(row)) + "\n")
                f.flush()
            with ThreadPoolExecutor(max_workers=2) as pool:
                for result in pool.map(judge, pending[2:]):
                    f.write(json.dumps(result) + "\n")
                    f.flush()
        results = read(output)
        assert len(results) == cell["sample_count"]
        summary = {"cell": cell["cell"], "judge_model": model, "n": len(results), "by_kind": {}}
        for kind in ("initial", "challenge"):
            subset = [r for r in results if r["kind"] == kind]
            confusion = {a: {b: 0.0 for b in LABELS} for a in LABELS}
            for r in subset:
                confusion[r["primary_label"]][r["alternative_label"]] += r["weight"]
            total = sum(r["weight"] for r in subset)
            observed = sum(confusion[a][a] for a in LABELS) / total
            expected = sum(sum(confusion[a].values()) * sum(confusion[b][a] for b in LABELS)
                           for a in LABELS) / total**2
            summary["by_kind"][kind] = {"n": len(subset), "weighted_confusion": confusion,
                "weighted_agreement": observed, "weighted_kappa": (observed-expected)/(1-expected) if expected < 1 else None}
        (args.output_dir / f"{cell['cell']}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Completed {cell['cell']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--results-dir", type=Path, required=True)
    prep.add_argument("--questions-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--input-dir", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--group", type=int, default=0)
    execute.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
