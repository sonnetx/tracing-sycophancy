#!/usr/bin/env python3
"""Recompute budget sensitivity with the paper's existing statistical functions.

Inputs are original and rerun experiment directories. Think medical reuses the
original 8192-token runs unless the rerun directory has its own copy. Candidate
scores come from the original experiment. Fixed-item sensitivity keeps original eligibility even when a new initial
answer is incorrect. No generations or judge calls are made.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze import TRAINING_PIPELINES
from src.analysis.stats import (load_results_as_dataframe, load_logprob_results,
    compute_metrics_summary, compute_control_summary, compute_matched_regressive,
    compute_paired_regressive, compute_paired_delta_logodds)
from src.analysis.plots import plot_challenge_type_trajectories, plot_control_comparison


def correct(df):
    return set(df.loc[(df.response_type == 'initial') & (df.factual_accuracy == 'correct'), 'question_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-dir', type=Path, required=True)
    p.add_argument('--rerun-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    models = list(dict.fromkeys(m for stages in TRAINING_PIPELINES.values() for m, _ in stages))
    audit = {}
    for dom in ('computational', 'medical_advice'):
        out = args.output_dir / dom / 'analysis'
        out.mkdir(parents=True, exist_ok=True)
        frames, scores, summaries, controls = {}, {}, {}, {}
        for m in models:
            old = args.original_dir / dom / m
            new = args.rerun_dir / dom / m
            if dom == 'medical_advice' and 'think' in m and not new.exists():
                new = old
            frames[m] = {v: load_results_as_dataframe(str(d / 'evaluated.jsonl')) for v, d in [('old', old), ('new', new)]}
            scores[m] = load_logprob_results(str(old / 'logprob_scores.jsonl'))
            df = frames[m]['new']
            wrong = df[(df.response_type == 'initial') | df.challenge_type.isin(['simple', 'ethos', 'justification', 'citation'])]
            summaries[m] = compute_metrics_summary(wrong)
            controls[m] = compute_control_summary(frames[m]['new'])
            summaries[m]['controls'] = controls[m]
            df = frames[m]['new']
            ch = df[(df.response_type == 'challenge') & df.question_id.isin(correct(df))]
            summaries[m]['type_context'] = {}
            summaries[m]['context_metrics'] = {}
            for ctx in ('in_context', 'preemptive'):
                for typ in ('simple', 'ethos', 'justification', 'citation'):
                    c = ch[(ch.challenge_context == ctx) & (ch.challenge_type == typ)]
                    coh = c[c.factual_accuracy.isin(['correct', 'incorrect'])]
                    summaries[m]['type_context'][f'{typ}_{ctx}'] = {
                        'count': int((coh.factual_accuracy == 'incorrect').sum()),
                        'total': len(coh),
                        'rate': float((coh.factual_accuracy == 'incorrect').mean())}
                c = ch[(ch.challenge_context == ctx) & ch.challenge_type.isin(['simple', 'ethos', 'justification', 'citation'])]
                coh = c[c.factual_accuracy.isin(['correct', 'incorrect'])]
                rate = float((coh.factual_accuracy == 'incorrect').mean())
                ctrl = controls[m]['correct']['by_context'][ctx]['flip_rate']
                summaries[m]['context_metrics'][ctx] = {'regr': rate, 'ctrl': ctrl,
                    'net': rate - ctrl, 'err': float((c.factual_accuracy == 'erroneous').mean())}
        beh, lp = {}, {}
        for pipe, stages in TRAINING_PIPELINES.items():
            bm = stages[0][0]
            beh[pipe], lp[pipe] = [], []
            for m, stage in stages:
                original_items = correct(frames[bm]['old']) & correct(frames[m]['old'])
                new_items = correct(frames[bm]['new']) & correct(frames[m]['new'])
                comparisons = {}
                for version, items in [('old', original_items), ('new', new_items), ('fixed_items', original_items)]:
                    v = 'old' if version == 'old' else 'new'
                    b, s = frames[bm][v], frames[m][v]
                    if version == 'fixed_items':
                        b, s = b.copy(), s.copy()
                        for df in (b, s):
                            df.loc[(df.response_type == 'initial') & df.question_id.isin(items), 'factual_accuracy'] = 'correct'
                    comparisons[version] = {
                        'n': len(items),
                        'base': compute_matched_regressive(b, items),
                        'stage': compute_matched_regressive(s, items),
                        'paired': compute_paired_regressive(b, s, items),
                        'receptivity': compute_paired_delta_logodds(scores[bm], scores[m], items, context='preemptive'),
                    }
                if m != bm:
                    audit[f'{dom}/{m}'] = comparisons
                r = comparisons['new']
                beh[pipe].append({'stage': stage, 'model': m, 'n_intersection': r['n'],
                    'base_regressive_on_intersection': r['base'], 'stage_regressive_on_intersection': r['stage'],
                    'base_vs_stage_paired': r['paired']})
                lp[pipe].append({'stage': stage, 'model': m, 'n_intersection': r['n'], 'paired': r['receptivity']})
        for name, data in [('summaries', summaries), ('control_summaries', controls), ('matched_summaries', beh), ('matched_lp_summaries', lp)]:
            (out / f'{name}.json').write_text(json.dumps(data, indent=2, default=str) + '\n')
        plot_challenge_type_trajectories(summaries, str(out))
        pipeline_data = {pipe: [(label, summaries[m]) for m, label in stages] for pipe, stages in TRAINING_PIPELINES.items()}
        plot_control_comparison(pipeline_data, controls, str(out))
    (args.output_dir / 'matched_budget_comparison.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Saved analysis to', args.output_dir)


if __name__ == '__main__':
    main()
