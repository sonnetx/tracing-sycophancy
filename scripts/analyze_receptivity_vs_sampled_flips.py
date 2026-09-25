#!/usr/bin/env python3
"""Within-checkpoint association between ΔLogOdds (receptivity) and sampled flips.

Unit: (question, challenge) cell at one checkpoint/domain/context, five samples.
Outcome: incorrect / coherent samples (coherent = correct + incorrect).
Model: binomial GLM  flips ~ z(ΔLogOdds) + challenge_type + z(baseline log-odds),
fit separately per checkpoint x domain x context, non-simple challenges only.
CI: cluster bootstrap over questions. Erroneous-rate model reported alongside.
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
SAMP = sys.argv[1]           # dir holding exp1_sampling*/<domain>/<model>.jsonl (judge labels only)
EXP1 = sys.argv[2]           # data/results/exp1
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
NBOOT = 1000
rng = np.random.default_rng(0)

rows = []
for exp in ("exp1_sampling", "exp1_sampling_ic"):
    for p in glob.glob(f"{SAMP}/{exp}/*/*.jsonl"):
        rows += [json.loads(l) for l in open(p)]
s = pd.DataFrame(rows).drop_duplicates(
    ["question_id", "model", "challenge_id", "temperature", "sample_idx"])
s = s[(s.temperature == TEMP) & (s.challenge_type.isin(["ethos", "justification", "citation"])) & (~s.model.str.endswith("-base"))]
s["domain"] = np.where(s.question_id.str.startswith("med_"), "medical_advice", "computational")

agg = (s.assign(inc=s.factual_accuracy.eq("incorrect"), cor=s.factual_accuracy.eq("correct"),
                err=s.factual_accuracy.eq("erroneous"))
         .groupby(["domain", "model", "question_id", "challenge_id", "challenge_type", "challenge_context"])
         [["inc", "cor", "err"]].sum().reset_index())
agg["n"] = agg.inc + agg.cor + agg.err
agg["coh"] = agg.inc + agg.cor

lp = []
for (dom, model) in agg[["domain", "model"]].drop_duplicates().itertuples(index=False):
    for l in open(f"{EXP1}/{dom}/{model}/logprob_scores.jsonl"):
        r = json.loads(l)
        for c in r["challenge_scores"]:
            lp.append({"domain": dom, "model": model, "question_id": r["question_id"],
                       "challenge_id": c["challenge_id"], "dlo": c["delta_log_odds"],
                       "base_lo": r["baseline"]["log_odds"]})
df = agg.merge(pd.DataFrame(lp), on=["domain", "model", "question_id", "challenge_id"], how="inner")

def fit(d, outcome, denom):
    d = d[d[denom] > 0].copy()
    d["y"] = d[outcome] / d[denom]
    d["z"] = (d.dlo - d.dlo.mean()) / d.dlo.std()
    d["zb"] = (d.base_lo - d.base_lo.mean()) / d.base_lo.std()
    m = smf.glm("y ~ z + C(challenge_type) + zb", d, family=sm.families.Binomial(),
                var_weights=d[denom]).fit()
    return m.params["z"]

out = []
for (dom, model, ctx), d in df.groupby(["domain", "model", "challenge_context"]):
    qs = d.question_id.unique()
    try:
        b_flip, b_err = fit(d, "inc", "coh"), fit(d, "err", "n")
    except Exception as e:
        print("skip", dom, model, ctx, e); continue
    boots_f, boots_e = [], []
    groups = {q: g for q, g in d.groupby("question_id")}
    for _ in range(NBOOT):
        pick = rng.choice(qs, len(qs), replace=True)
        bd = pd.concat([groups[q] for q in pick], ignore_index=True)
        try:
            boots_f.append(fit(bd, "inc", "coh")); boots_e.append(fit(bd, "err", "n"))
        except Exception:
            pass
    lo_f, hi_f = np.percentile(boots_f, [2.5, 97.5]); lo_e, hi_e = np.percentile(boots_e, [2.5, 97.5])
    # descriptive: coherent flip rate by within-type ΔLO tercile
    d = d.copy()
    d["terc"] = d.groupby("challenge_type").dlo.transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
    t = d.groupby("terc")[["inc", "coh"]].sum()
    terc = (t.inc / t.coh).round(3).tolist()
    out.append(dict(domain=dom, model=model, ctx=ctx, n_q=len(qs), n_cells=len(d),
                    flip_rate=round(d.inc.sum() / d.coh.sum(), 3), err_rate=round(d.err.sum() / d.n.sum(), 3),
                    OR_flip=round(np.exp(b_flip), 3), OR_flip_lo=round(np.exp(lo_f), 3), OR_flip_hi=round(np.exp(hi_f), 3),
                    OR_err=round(np.exp(b_err), 3), OR_err_lo=round(np.exp(lo_e), 3), OR_err_hi=round(np.exp(hi_e), 3),
                    flip_by_dlo_tercile=terc, n_boot=len(boots_f)))
    print(out[-1], flush=True)

res = pd.DataFrame(out)
res.to_csv(f"receptivity_vs_flips_T{TEMP}.csv", index=False)
pd.set_option("display.width", 250)
print(res.drop(columns=["n_boot"]).to_string(index=False))
