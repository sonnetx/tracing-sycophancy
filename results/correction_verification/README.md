# Correction verification outputs

These files let readers check the corrected log-probability numbers without rerunning inference. Earlier versions of the analysis averaged "non-simple" challenges by excluding only the simple type, which let the correct-answer and neutral control conditions into those averages. Commit `ed28099` selects ethos, justification, and citation explicitly, and `tests/test_wrong_answer_filter.py` guards the rule.

- `old/` holds analysis outputs from the pre-fix code. They reproduce the numbers in earlier versions of the paper.
- `new/` holds the same analyses from the fixed code. `exp1_<domain>/matched_lp_summaries.json` has the matched-subset ΔLogOdds growth with bootstrap intervals and Wilcoxon tests. `candidate_*.json`, `length.json`, and `paraphrase_*.json` are the robustness checks, split by context where noted.
- `paraphrase_scores.tgz` holds per-item log-probability scores for the 200-item medical paraphrase subset at all 12 checkpoints (variants `c0_w0` original, `c1_w0`/`c2_w0` paraphrased correct answer, `c0_w1`/`c0_w2` paraphrased wrong answer).
- `sampling_labels.tgz` holds judge labels for every sampled response (question, challenge, temperature, sample index, label), without the response text.
- `initial_correct.json` lists, per checkpoint, the questions answered correctly without challenge. It defines the matched subsets.
- `receptivity_vs_flips_T1.0.csv` is the output of `scripts/analyze_receptivity_vs_sampled_flips.py` at T=1.0.

The medical items and reference answers come from MedQuAD. Wrong answers, challenges, and paraphrases were generated with GPT-4o.
