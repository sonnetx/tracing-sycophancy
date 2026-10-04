#!/usr/bin/env python3
"""Re-judge OLMo Think responses on the answer after the end-of-reasoning marker.

Reads evaluated.jsonl files and writes copies whose factual_accuracy comes from the
GPT-4o judge prompt applied to final_answer(response). Responses without the marker
keep their label because the judge input would not change, except challenges that
were skipped before and become eligible. The evaluate step's skip rule is reapplied
with the new initial label. Calls are cached so an interrupted run resumes for free.

Usage:
    python scripts/rejudge_think_final_answer.py --results-dir data/results \
        --questions-dir data/processed --output-dir data/results/think_final_answer \
        exp1_rerun16k/computational exp1/medical_advice [--dry-run]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.judge import (RESPONSE_FORMAT, SYSTEM_MESSAGE, THINK_END,
                                  _build_prompt, _parse_response, final_answer)

MODELS = ["olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think"]
SKIP = {"factual_accuracy": "erroneous", "agreement": None, "skipped": True,
        "skip_reason": "initial_response_erroneous_or_refused"}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Judge:
    def __init__(self, cache_path, workers):
        from openai import OpenAI
        self.client = OpenAI()
        self.cache_path = cache_path
        self.cache = {r["key"]: r["label"] for r in read(cache_path)} if cache_path.exists() else {}
        self.lock = threading.Lock()
        self.workers = workers
        self.tokens = [0, 0]

    def call(self, key, question, answer, text):
        if key in self.cache:
            return
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": _build_prompt(question, answer, text)}]
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(model="gpt-4o", temperature=0,
                    messages=messages, response_format=RESPONSE_FORMAT)
                break
            except Exception as e:
                if attempt == 7:
                    print(f"failed {key}: {e}", flush=True)
                    return
                time.sleep(min(60, 2 ** attempt))
        label = _parse_response(r.choices[0].message.content)["factual_accuracy"]
        with self.lock:
            self.cache[key] = label
            self.tokens[0] += r.usage.prompt_tokens
            self.tokens[1] += r.usage.completion_tokens
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "label": label}) + "\n")
            n = len(self.cache)
            if n % 1000 == 0:
                print(f"{n} cached, {self.tokens[0]:,} in / {self.tokens[1]:,} out tokens this run", flush=True)

    def run(self, jobs):
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(lambda j: self.call(*j), jobs))

    def run_batch(self, jobs, state_path, chunk=10000, max_inflight=2):
        """Same requests through the Batch API at half price. In-flight batch IDs are saved,
        so a restarted run polls them instead of submitting the same requests again."""
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        failures = 0
        while True:
            pending = {j[0]: j for j in jobs if j[0] not in self.cache}
            if not pending:
                return
            inflight = {k for keys in state.values() for k in keys}
            new = [pending[k] for k in pending if k not in inflight]
            while new and len(state) < max_inflight:
                part, new = new[:chunk], new[chunk:]
                path = state_path.with_name(f"batch_input_{len(self.cache)}_{len(state)}.jsonl")
                with open(path, "w") as f:
                    for key, question, answer, text in part:
                        f.write(json.dumps({"custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                            "body": {"model": "gpt-4o", "temperature": 0, "response_format": RESPONSE_FORMAT,
                                     "messages": [{"role": "system", "content": SYSTEM_MESSAGE},
                                                  {"role": "user", "content": _build_prompt(question, answer, text)}]}}) + "\n")
                with open(path, "rb") as f:
                    upload = self.client.files.create(file=f, purpose="batch")
                path.unlink()
                batch = self.client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                                   completion_window="24h")
                state[batch.id] = [j[0] for j in part]
                state_path.write_text(json.dumps(state))
                print(f"submitted {batch.id} with {len(part)} requests", flush=True)
            time.sleep(60)
            for bid in list(state):
                b = self.client.batches.retrieve(bid)
                if b.status not in ("completed", "failed", "expired", "cancelled"):
                    print(f"{bid} {b.status} {b.request_counts.completed}/{b.request_counts.total}", flush=True)
                    continue
                got = self.ingest(self.client.files.content(b.output_file_id).text) if b.output_file_id else 0
                if b.status == "completed" and got < 0.9 * len(state[bid]):
                    # Stop rather than pay again for requests whose results did not parse.
                    sys.exit(f"{bid} completed but only {got}/{len(state[bid])} results parsed")
                if b.status != "completed":
                    failures += 1
                    print(f"{bid} {b.status}: {b.errors}", flush=True)
                    if failures > 5:
                        sys.exit("too many failed batches")
                del state[bid]
                state_path.write_text(json.dumps(state))
                print(f"{bid} {b.status}, {len(self.cache)} cached, "
                      f"{self.tokens[0]:,} in / {self.tokens[1]:,} out tokens this run", flush=True)

    def ingest(self, text):
        got = 0
        with open(self.cache_path, "a") as f:
            for line in text.splitlines():
                r = json.loads(line)
                body = (r.get("response") or {}).get("body") or {}
                if r.get("error") or (r.get("response") or {}).get("status_code") != 200:
                    continue
                label = _parse_response(body["choices"][0]["message"]["content"])["factual_accuracy"]
                self.cache[r["custom_id"]] = label
                self.tokens[0] += body["usage"]["prompt_tokens"]
                self.tokens[1] += body["usage"]["completion_tokens"]
                f.write(json.dumps({"key": r["custom_id"], "label": label}) + "\n")
                got += 1
        return got


def initial_label(r, base, cache):
    """New initial label, or the old one while a pending call has not filled the cache."""
    if THINK_END in r["initial"]["response"]:
        return cache.get(f"{base}/initial", r["initial"]["metrics"]["factual_accuracy"])
    return r["initial"]["metrics"]["factual_accuracy"]


def plan(rows, source, cache, stage):
    """Yield (key, response) for every response whose label needs a new call."""
    for r in rows:
        base = f"{source}/{r['question_id']}"
        if stage == "initial":
            if THINK_END in r["initial"]["response"]:
                yield f"{base}/initial", r["initial"]["response"]
            continue
        if r["initial"]["metrics"].get("refusal") or initial_label(r, base, cache) == "erroneous":
            continue
        for c in r["challenge_responses"]:
            if THINK_END in c["response"] or c["metrics"].get("skipped"):
                yield f"{base}/{c['challenge_id']}", c["response"]


def rebuild(rows, source, cache):
    out, changed = [], 0
    for r in rows:
        base = f"{source}/{r['question_id']}"
        init = dict(r["initial"], metrics=dict(r["initial"]["metrics"]))
        if THINK_END in init["response"]:
            init["metrics"]["factual_accuracy"] = cache[f"{base}/initial"]
            init["metrics"]["judge_input"] = "final_answer"
        skip = init["metrics"].get("refusal") or init["metrics"]["factual_accuracy"] == "erroneous"
        challenges = []
        for c in r["challenge_responses"]:
            m = {k: v for k, v in c["metrics"].items() if k not in ("skipped", "skip_reason")}
            if skip:
                m.update(SKIP)
            elif THINK_END in c["response"] or c["metrics"].get("skipped"):
                m["factual_accuracy"] = cache[f"{base}/{c['challenge_id']}"]
                m["judge_input"] = "final_answer" if THINK_END in c["response"] else "full_response"
            changed += m["factual_accuracy"] != c["metrics"]["factual_accuracy"]
            challenges.append(dict(c, metrics=m))
        changed += init["metrics"]["factual_accuracy"] != r["initial"]["metrics"]["factual_accuracy"]
        out.append(dict(r, initial=init, challenge_responses=challenges))
    return out, changed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", help="experiment/domain pairs under --results-dir")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--questions-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch", action="store_true", help="use the OpenAI Batch API (half price, slower)")
    p.add_argument("--max-inflight", type=int, default=2, help="batches queued at once (enqueued-token limits)")
    p.add_argument("--dry-run", action="store_true", help="count calls and input characters only")
    p.add_argument("--limit", type=int, help="judge only the first N calls per stage and write nothing (smoke test)")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for source in args.sources:
        domain = source.split("/")[-1]
        questions = {q["id"]: q for q in read(args.questions_dir / f"{domain}.jsonl")}
        for model in MODELS:
            rows = read(args.results_dir / source / model / "evaluated.jsonl")
            cells.append((source, model, rows, questions))

    judge = None if args.dry_run else Judge(args.output_dir / "judge_cache.jsonl", args.workers)
    cache = {} if args.dry_run else judge.cache
    for stage in ("initial", "challenge"):
        jobs = []
        for source, model, rows, questions in cells:
            cell_jobs = []
            for key, text in plan(rows, f"{source}/{model}", cache, stage):
                q = questions[key.split("/")[-2]]
                cell_jobs.append((key, q["question"], q["correct_answer"], final_answer(text)))
            print(f"{stage} {source}/{model}: {len(cell_jobs)} calls, "
                  f"{sum(len(j[3]) for j in cell_jobs) // max(len(cell_jobs), 1)} mean answer chars", flush=True)
            jobs += cell_jobs
        if args.limit:
            jobs = jobs[:args.limit]
        print(f"{stage} total: {len(jobs)} calls, {sum(len(j[3]) for j in jobs):,} answer chars", flush=True)
        if args.dry_run:
            continue
        if args.batch:
            judge.run_batch(jobs, args.output_dir / f"batches_{stage}.json", max_inflight=args.max_inflight)
        else:
            judge.run(jobs)
        missing = [j[0] for j in jobs if j[0] not in judge.cache]
        if missing:
            sys.exit(f"{len(missing)} calls failed; rerun to retry them")
    if args.dry_run or args.limit:
        return
    for source, model, rows, _ in cells:
        out, changed = rebuild(rows, f"{source}/{model}", judge.cache)
        dest = args.output_dir / source / model
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "evaluated.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        for extra in (args.results_dir / source / model).iterdir():
            if extra.name != "evaluated.jsonl" and not (dest / extra.name).exists():
                (dest / extra.name).symlink_to(extra.resolve())
        print(f"{source}/{model}: {changed} labels changed", flush=True)
    print(f"tokens this run: {judge.tokens[0]:,} in, {judge.tokens[1]:,} out", flush=True)


if __name__ == "__main__":
    main()
