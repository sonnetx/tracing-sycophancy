"""Control conditions must never enter wrong-answer ΔLogOdds aggregates.

Log-prob files store correct-answer and neutral controls alongside the
wrong-answer challenges, all under condition == "challenge". Excluding only
"simple" once let the controls into every "non-simple" mean.
"""

import importlib.util
import json
import os
import tempfile
import unittest

from src.analysis.stats import (
    compute_logprob_summary,
    compute_matched_delta_logodds,
    compute_paired_delta_logodds,
    load_logprob_results,
    select_wrong_answer_challenges,
)

# Wrong-answer challenges carry +1.0, simple +0.5, controls a large negative
# value, so any leak moves the mean away from 1.0.
DLO = {"simple": 0.5, "ethos": 1.0, "justification": 1.0, "citation": 1.0,
       "correct": -5.0, "neutral": -5.0}


def _item(qid: str, scale: float = 1.0) -> dict:
    scores = []
    for t, v in DLO.items():
        for ctx, tag in (("in_context", "incontext"), ("preemptive", "preemptive")):
            scores.append({
                "challenge_id": f"{t}_{tag}", "challenge_type": t, "challenge_context": ctx,
                "correct_log_prob": 0.0, "incorrect_log_prob": 0.0,
                "correct_mean_log_prob": 0.0, "incorrect_mean_log_prob": 0.0,
                "log_odds": v * scale, "delta_log_odds": v * scale,
            })
    return {
        "question_id": qid, "model": "m", "checkpoint": "c",
        "baseline": {"correct_log_prob": 0.0, "incorrect_log_prob": 0.0,
                     "correct_mean_log_prob": 0.0, "incorrect_mean_log_prob": 0.0,
                     "log_odds": 0.0},
        "challenge_scores": scores,
    }


def _write(items: list[dict], directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path


def _load_script(name: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WrongAnswerFilterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        qids = ["q0", "q1", "q2"]
        self.base = load_logprob_results(_write([_item(q) for q in qids], self.tmp.name, "base.jsonl"))
        self.stage = load_logprob_results(
            _write([_item(q, scale=2.0) for q in qids], self.tmp.name, "stage.jsonl"))
        self.qids = set(qids)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selector_drops_controls(self):
        ch = self.base[self.base["condition"] == "challenge"]
        self.assertEqual(set(select_wrong_answer_challenges(ch)["challenge_type"]),
                         {"ethos", "justification", "citation"})
        self.assertEqual(set(select_wrong_answer_challenges(ch, non_simple_only=False)["challenge_type"]),
                         {"simple", "ethos", "justification", "citation"})

    def test_matched_mean_excludes_controls(self):
        for ctx in ("preemptive", "in_context"):
            r = compute_matched_delta_logodds(self.base, self.qids, context=ctx)
            self.assertAlmostEqual(r["mean_delta_log_odds"], 1.0)
            self.assertEqual(r["n_obs"], 9)

    def test_paired_excludes_controls(self):
        r = compute_paired_delta_logodds(self.base, self.stage, self.qids,
                                         context="preemptive", n_bootstrap=50)
        self.assertAlmostEqual(r["mean_delta_delta_L"], 1.0)

    def test_summary_excludes_controls(self):
        s = compute_logprob_summary(self.base)["challenges"]
        self.assertAlmostEqual(s["overall"]["mean_delta_log_odds"], 1.0)
        self.assertAlmostEqual(s["preemptive"]["mean_delta_log_odds"], 1.0)
        self.assertAlmostEqual(s["in_context"]["mean_delta_log_odds"], 1.0)
        self.assertAlmostEqual(s["overall_all_types"]["mean_delta_log_odds"], 0.875)

    def test_robustness_scripts_exclude_controls(self):
        para = _load_script("analyze_paraphrase_robustness")
        self.assertAlmostEqual(para.per_question_means(self.base).mean(), 1.0)
        self.assertAlmostEqual(para.per_question_means(self.base, "preemptive").mean(), 1.0)
        cand = _load_script("analyze_candidate_robustness")
        self.assertAlmostEqual(cand.per_checkpoint_mean_dlo(self.base, context="in_context"), 1.0)


if __name__ == "__main__":
    unittest.main()
