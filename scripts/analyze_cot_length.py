#!/usr/bin/env python3
"""Mean response length (chars) for OLMo Think: unchallenged initial vs
challenge conditions, grouped by context and wrong-answer vs control."""
import io
import json
import statistics
import sys

WRONG = {"simple", "ethos", "justification", "citation"}

for path in sys.argv[1:]:
    groups = {}
    initial = []
    n = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            resp = (r.get("initial") or {}).get("response") or ""
            initial.append(len(resp))
            for ch in r.get("challenge_responses", []):
                cid = ch.get("challenge_id", "")
                text = ch.get("response") or ""
                base, _, ctx = cid.rpartition("_")
                kind = "wrong" if base in WRONG else base
                groups.setdefault((kind, ctx), []).append(len(text))
    print("==", path, f"n_items={n}")
    print(f"  initial (unchallenged): mean={statistics.mean(initial):7.0f} median={statistics.median(initial):7.0f} n={len(initial)}")
    for key in sorted(groups):
        v = groups[key]
        print(f"  {key[0]:>8}/{key[1]:<10}: mean={statistics.mean(v):7.0f} median={statistics.median(v):7.0f} n={len(v)}")
