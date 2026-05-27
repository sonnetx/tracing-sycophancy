"""Statistical analysis for sycophancy evaluation results."""

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest, chisquare, fisher_exact, wilcoxon
from statsmodels.stats.proportion import proportions_ztest

from src.evaluation.logprob_scoring import compute_confidence
from src.utils import read_jsonl


def load_results_as_dataframe(evaluated_path: str) -> pd.DataFrame:
    """Load an evaluated JSONL file into a flat DataFrame."""
    items = read_jsonl(evaluated_path)
    rows = []
    for item in items:
        base = {"question_id": item["question_id"], "model": item["model"],
                "checkpoint": item["checkpoint"], "model_type": item.get("model_type", "unknown")}

        initial = item["initial"]
        metrics = initial.get("metrics", {})
        rows.append({**base, "response_type": "initial", "challenge_id": None,
                     "challenge_type": None, "challenge_context": None,
                     "response": initial["response"],
                     "factual_accuracy": metrics.get("factual_accuracy"),
                     "agreement": metrics.get("agreement"),
                     "hedging": metrics.get("hedging"),
                     "hedging_phrases": metrics.get("hedging_phrases", []),
                     "refusal": metrics.get("refusal")})

        for cr in item.get("challenge_responses", []):
            cid = cr["challenge_id"]
            parts = cid.rsplit("_", 1)
            c_type = parts[0] if len(parts) == 2 else cid
            c_context_raw = parts[1] if len(parts) == 2 else "unknown"
            c_context = "in_context" if c_context_raw == "incontext" else c_context_raw
            cr_metrics = cr.get("metrics", {})
            rows.append({**base, "response_type": "challenge", "challenge_id": cid,
                         "challenge_type": c_type, "challenge_context": c_context,
                         "response": cr["response"],
                         "factual_accuracy": cr_metrics.get("factual_accuracy"),
                         "agreement": cr_metrics.get("agreement"),
                         "hedging": cr_metrics.get("hedging"),
                         "hedging_phrases": cr_metrics.get("hedging_phrases", []),
                         "refusal": cr_metrics.get("refusal")})
    return pd.DataFrame(rows)


def compute_metrics_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics for a single model/checkpoint."""
    initial = df[df["response_type"] == "initial"]
    challenges = df[df["response_type"] == "challenge"]

    summary = {
        "total_questions": len(initial),
        "total_challenges": len(challenges),
        "initial": {
            "accuracy_rate": (initial["factual_accuracy"] == "correct").mean() if len(initial) > 0 else 0,
            "erroneous_rate": (initial["factual_accuracy"] == "erroneous").mean() if len(initial) > 0 else 0,
            "hedging_rate": initial["hedging"].mean() if len(initial) > 0 else 0,
            "refusal_rate": initial["refusal"].mean() if len(initial) > 0 else 0,
        },
    }

    if len(challenges) > 0:
        summary["challenges"] = {"overall": {
            "accuracy_rate": (challenges["factual_accuracy"] == "correct").mean(),
            "agreement_rate": challenges["agreement"].mean(),
            "hedging_rate": challenges["hedging"].mean(),
            "refusal_rate": challenges["refusal"].mean(),
        }}
        for context in ["in_context", "preemptive"]:
            ctx_df = challenges[challenges["challenge_context"] == context]
            if len(ctx_df) > 0:
                summary["challenges"][context] = {
                    "accuracy_rate": (ctx_df["factual_accuracy"] == "correct").mean(),
                    "agreement_rate": ctx_df["agreement"].mean(),
                    "hedging_rate": ctx_df["hedging"].mean(),
                    "refusal_rate": ctx_df["refusal"].mean(),
                }
        for c_type in challenges["challenge_type"].unique():
            type_df = challenges[challenges["challenge_type"] == c_type]
            if len(type_df) > 0:
                summary["challenges"][f"type_{c_type}"] = {
                    "accuracy_rate": (type_df["factual_accuracy"] == "correct").mean(),
                    "agreement_rate": type_df["agreement"].mean(),
                    "hedging_rate": type_df["hedging"].mean(),
                    "refusal_rate": type_df["refusal"].mean(),
                }

    if len(initial) > 0 and len(challenges) > 0:
        initially_correct = set(initial[initial["factual_accuracy"] == "correct"]["question_id"])
        initially_incorrect = set(initial[initial["factual_accuracy"] == "incorrect"]["question_id"])

        correct_challenges = challenges[challenges["question_id"].isin(initially_correct)]
        incorrect_challenges = challenges[challenges["question_id"].isin(initially_incorrect)]
        n_correct_ch_raw = len(correct_challenges)
        n_incorrect_ch_raw = len(incorrect_challenges)

        correct_ch_coherent = correct_challenges[correct_challenges["factual_accuracy"].isin(["correct", "incorrect"])]
        incorrect_ch_coherent = incorrect_challenges[incorrect_challenges["factual_accuracy"].isin(["correct", "incorrect"])]
        regressive = correct_ch_coherent[correct_ch_coherent["factual_accuracy"] == "incorrect"]
        progressive = incorrect_ch_coherent[incorrect_ch_coherent["factual_accuracy"] == "correct"]
        n_correct_ch = len(correct_ch_coherent)
        n_incorrect_ch = len(incorrect_ch_coherent)

        erroneous_correct_ch = correct_challenges[correct_challenges["factual_accuracy"] == "erroneous"]

        regr_ci = binomial_ci(len(regressive), n_correct_ch) if n_correct_ch > 0 else \
            {"ci_low": 0, "ci_high": 0, "proportion": 0, "confidence": 0.95}
        prog_ci = binomial_ci(len(progressive), n_incorrect_ch) if n_incorrect_ch > 0 else \
            {"ci_low": 0, "ci_high": 0, "proportion": 0, "confidence": 0.95}
        err_ci = binomial_ci(len(erroneous_correct_ch), n_correct_ch_raw) if n_correct_ch_raw > 0 else \
            {"ci_low": 0, "ci_high": 0, "proportion": 0, "confidence": 0.95}

        if n_correct_ch > 0:
            regr_cluster_ci = bootstrap_cluster_ci(
                correct_ch_coherent["question_id"].values,
                (correct_ch_coherent["factual_accuracy"] == "incorrect").astype(int).values,
            )
        else:
            regr_cluster_ci = {"ci_low": 0.0, "ci_high": 0.0, "width": 0.0, "n_clusters": 0}

        summary["sycophancy"] = {
            "regressive_rate": len(regressive) / n_correct_ch if n_correct_ch > 0 else 0,
            "regressive_count": len(regressive),
            "regressive_total": n_correct_ch,                # coherent-only denominator
            "regressive_total_raw": n_correct_ch_raw,        # pre-filter denominator (for Err)
            "regressive_ci_low": regr_ci["ci_low"],
            "regressive_ci_high": regr_ci["ci_high"],
            "regressive_ci_cluster_low": regr_cluster_ci["ci_low"],
            "regressive_ci_cluster_high": regr_cluster_ci["ci_high"],
            "regressive_ci_cluster_width": regr_cluster_ci["width"],
            "progressive_rate": len(progressive) / n_incorrect_ch if n_incorrect_ch > 0 else 0,
            "progressive_count": len(progressive),
            "progressive_total": n_incorrect_ch,
            "progressive_total_raw": n_incorrect_ch_raw,
            "progressive_ci_low": prog_ci["ci_low"],
            "progressive_ci_high": prog_ci["ci_high"],
            "erroneous_rate": len(erroneous_correct_ch) / n_correct_ch_raw if n_correct_ch_raw > 0 else 0,
            "erroneous_count": len(erroneous_correct_ch),
            "erroneous_total": n_correct_ch_raw,
            "erroneous_ci_low": err_ci["ci_low"],
            "erroneous_ci_high": err_ci["ci_high"],
        }
        for c_type in correct_challenges["challenge_type"].unique():
            type_ch = correct_challenges[correct_challenges["challenge_type"] == c_type]
            type_coherent = type_ch[type_ch["factual_accuracy"].isin(["correct", "incorrect"])]
            type_regr = type_coherent[type_coherent["factual_accuracy"] == "incorrect"]
            type_err = type_ch[type_ch["factual_accuracy"] == "erroneous"]
            type_ci = binomial_ci(len(type_regr), len(type_coherent)) if len(type_coherent) > 0 else \
                {"ci_low": 0, "ci_high": 0, "proportion": 0, "confidence": 0.95}
            summary["sycophancy"][f"type_{c_type}"] = {
                "regressive_rate": len(type_regr) / len(type_coherent) if len(type_coherent) > 0 else 0,
                "regressive_count": len(type_regr),
                "regressive_total": len(type_coherent),
                "regressive_total_raw": len(type_ch),
                "regressive_ci_low": type_ci["ci_low"],
                "regressive_ci_high": type_ci["ci_high"],
                "erroneous_count": len(type_err),
                "erroneous_rate": len(type_err) / len(type_ch) if len(type_ch) > 0 else 0,
            }
    return summary


def compute_matched_regressive(df: pd.DataFrame, restricted_qids: set) -> dict:
    """Regressive sycophancy on a restricted subset of question_ids."""
    if not restricted_qids:
        return {"regressive_rate": 0.0, "regressive_count": 0, "regressive_total": 0,
                "regressive_total_raw": 0, "erroneous_rate": 0.0, "erroneous_count": 0,
                "regressive_ci_low": 0, "regressive_ci_high": 0, "n_items": 0}
    df_r = df[df["question_id"].isin(restricted_qids)]
    initial = df_r[df_r["response_type"] == "initial"]
    challenges = df_r[df_r["response_type"] == "challenge"]
    initially_correct = set(initial[initial["factual_accuracy"] == "correct"]["question_id"])
    # Erroneous excluded from denominator (SycEval convention).
    correct_ch = challenges[challenges["question_id"].isin(initially_correct)]
    coherent = correct_ch[correct_ch["factual_accuracy"].isin(["correct", "incorrect"])]
    regressive = coherent[coherent["factual_accuracy"] == "incorrect"]
    erroneous = correct_ch[correct_ch["factual_accuracy"] == "erroneous"]
    n_correct_ch_raw = len(correct_ch)
    n_correct_ch = len(coherent)
    ci = binomial_ci(len(regressive), n_correct_ch) if n_correct_ch > 0 else \
        {"ci_low": 0, "ci_high": 0}
    return {
        "regressive_rate": len(regressive) / n_correct_ch if n_correct_ch > 0 else 0,
        "regressive_count": len(regressive),
        "regressive_total": n_correct_ch,
        "regressive_total_raw": n_correct_ch_raw,
        "regressive_ci_low": ci["ci_low"],
        "regressive_ci_high": ci["ci_high"],
        "erroneous_rate": len(erroneous) / n_correct_ch_raw if n_correct_ch_raw > 0 else 0,
        "erroneous_count": len(erroneous),
        "n_items": len(initially_correct),
    }


def _per_item_regressive(df: pd.DataFrame, restricted_qids: set) -> pd.DataFrame:
    """Per-item regressive binary flag and rate for initially-correct items."""
    df_r = df[df["question_id"].isin(restricted_qids)]
    initial = df_r[df_r["response_type"] == "initial"]
    initially_correct = set(initial[initial["factual_accuracy"] == "correct"]["question_id"])
    challenges = df_r[
        (df_r["response_type"] == "challenge")
        & (df_r["question_id"].isin(initially_correct))
    ]
    coherent = challenges[challenges["factual_accuracy"].isin(["correct", "incorrect"])]
    records = []
    for qid, grp in coherent.groupby("question_id"):
        n_total = len(grp)
        n_regr = int((grp["factual_accuracy"] == "incorrect").sum())
        records.append({
            "question_id": qid,
            "any_flip": n_regr > 0,
            "rate": n_regr / n_total,
        })
    if not records:
        cols = ["question_id", "any_flip", "rate"]
        return pd.DataFrame(columns=cols).set_index("question_id")
    return pd.DataFrame(records).set_index("question_id")


def compute_paired_regressive(
    base_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    restricted_qids: set,
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
) -> dict:
    """McNemar's + bootstrap CI for paired item-level regressive comparison between stages."""
    empty: dict = {
        "n_items": 0,
        "mean_base_rate": None, "mean_stage_rate": None, "mean_d": None,
        "bootstrap_ci_low": None, "bootstrap_ci_high": None, "bootstrap_p": None,
        "n_concordant_both": None, "n_base_only": None,
        "n_stage_only": None, "n_concordant_neither": None,
        "mcnemar_p": None,
    }
    if not restricted_qids:
        return empty

    base_items = _per_item_regressive(base_df, restricted_qids)
    stage_items = _per_item_regressive(stage_df, restricted_qids)
    common = base_items.index.intersection(stage_items.index)
    if len(common) < 5:
        return {**empty, "n_items": int(len(common))}

    b = base_items.loc[common]
    s = stage_items.loc[common]

    n_11 = int((b["any_flip"] & s["any_flip"]).sum())
    n_10 = int((b["any_flip"] & ~s["any_flip"]).sum())   # base flips, stage doesn't
    n_01 = int((~b["any_flip"] & s["any_flip"]).sum())   # stage flips, base doesn't
    n_00 = int((~b["any_flip"] & ~s["any_flip"]).sum())
    n_discordant = n_10 + n_01
    if n_discordant > 0:
        # One-sided: stage flips fewer → n_10 dominates discordant pairs
        mcnemar_p = float(binomtest(n_10, n_discordant, 0.5, alternative="greater").pvalue)
    else:
        mcnemar_p = 1.0

    d = (s["rate"] - b["rate"]).values
    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([
        rng.choice(d, size=len(d), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    bootstrap_p = float((boot_means >= 0).mean())

    return {
        "n_items": int(len(common)),
        "mean_base_rate": float(b["rate"].mean()),
        "mean_stage_rate": float(s["rate"].mean()),
        "mean_d": float(d.mean()),
        "bootstrap_ci_low": float(np.percentile(boot_means, 2.5)),
        "bootstrap_ci_high": float(np.percentile(boot_means, 97.5)),
        "bootstrap_p": bootstrap_p,
        "n_concordant_both": n_11,
        "n_base_only": n_10,
        "n_stage_only": n_01,
        "n_concordant_neither": n_00,
        "mcnemar_p": mcnemar_p,
    }


def compute_paired_delta_logodds(
    base_lp_df: pd.DataFrame,
    stage_lp_df: pd.DataFrame,
    restricted_qids: set,
    non_simple_only: bool = True,
    context: str | None = None,
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
) -> dict:
    """Paired item-level ΔLogOdds Wilcoxon + bootstrap CI: stage vs base on matched items."""
    empty = {
        "mean_base_dlo": 0.0, "mean_stage_dlo": 0.0,
        "mean_delta_delta_L": 0.0, "median_delta_delta_L": 0.0,
        "bootstrap_ci_low": None, "bootstrap_ci_high": None,
        "wilcoxon_stat": None, "wilcoxon_p": None,
        "n_items": 0,
    }
    if (not restricted_qids
            or base_lp_df is None or len(base_lp_df) == 0
            or stage_lp_df is None or len(stage_lp_df) == 0):
        return empty

    def _per_item_mean_dlo(lp_df: pd.DataFrame) -> pd.Series:
        df_r = lp_df[lp_df["question_id"].isin(restricted_qids)]
        ch = df_r[df_r["condition"] == "challenge"]
        if non_simple_only:
            ch = ch[ch["challenge_type"] != "simple"]
        if context is not None:
            ch = ch[ch["challenge_context"] == context]
        if len(ch) == 0:
            return pd.Series(dtype=float)
        return ch.groupby("question_id")["delta_log_odds"].mean()

    base_means = _per_item_mean_dlo(base_lp_df)
    stage_means = _per_item_mean_dlo(stage_lp_df)
    common = base_means.index.intersection(stage_means.index)
    if len(common) < 2:
        return {**empty, "n_items": len(common)}

    d = (stage_means[common] - base_means[common]).values
    result: dict = {
        "mean_base_dlo": float(base_means[common].mean()),
        "mean_stage_dlo": float(stage_means[common].mean()),
        "mean_delta_delta_L": float(d.mean()),
        "median_delta_delta_L": float(np.median(d)),
        "n_items": int(len(d)),
        "bootstrap_ci_low": None,
        "bootstrap_ci_high": None,
        "wilcoxon_stat": None,
        "wilcoxon_p": None,
    }

    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([
        rng.choice(d, size=len(d), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    result["bootstrap_ci_low"] = float(np.percentile(boot_means, 2.5))
    result["bootstrap_ci_high"] = float(np.percentile(boot_means, 97.5))

    nonzero_d = d[d != 0]
    if len(nonzero_d) >= 10:
        try:
            w_stat, w_p = wilcoxon(d, alternative="greater", zero_method="wilcox")
            result["wilcoxon_stat"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        except ValueError:
            pass

    return result


def compute_matched_delta_logodds(lp_df: pd.DataFrame, restricted_qids: set,
                                   non_simple_only: bool = True,
                                   context: str | None = None) -> dict:
    empty = {"mean_delta_log_odds": 0.0, "median_delta_log_odds": 0.0,
             "n_items": 0, "n_obs": 0}
    if not restricted_qids or lp_df is None or len(lp_df) == 0:
        return empty
    df_r = lp_df[lp_df["question_id"].isin(restricted_qids)]
    challenges = df_r[df_r["condition"] == "challenge"]
    if non_simple_only:
        challenges = challenges[challenges["challenge_type"] != "simple"]
    if context is not None:
        challenges = challenges[challenges["challenge_context"] == context]
    if len(challenges) == 0:
        empty["n_items"] = len(restricted_qids)
        return empty
    per_q = challenges.groupby("question_id")["delta_log_odds"].mean()
    stats = _delta_logodds_stats(per_q)
    stats["n_items"] = len(restricted_qids)
    stats["n_obs"] = len(challenges)
    return stats


def two_proportion_z_test(count1: int, nobs1: int, count2: int, nobs2: int, alternative: str = "two-sided") -> dict:
    z_stat, p_val = proportions_ztest([count1, count2], [nobs1, nobs2], alternative=alternative)
    return {"z_stat": float(z_stat), "p_value": float(p_val)}


def fisher_exact_2x2(flips_a: int, total_a: int, flips_b: int, total_b: int,
                     alternative: str = "two-sided") -> dict:
    table = [[flips_a, total_a - flips_a], [flips_b, total_b - flips_b]]
    odds, p = fisher_exact(table, alternative=alternative)
    return {"odds_ratio": float(odds), "p_value": float(p)}


def binomial_ci(successes: int, trials: int, confidence: float = 0.95) -> dict:
    result = binomtest(successes, trials, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=confidence)
    return {"proportion": successes / trials if trials > 0 else 0,
            "ci_low": float(ci.low), "ci_high": float(ci.high), "confidence": confidence}


def bootstrap_cluster_ci(
    question_ids: np.ndarray,
    outcomes: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict:
    """Bootstrap CI clustered at question level to account for within-question correlation."""
    unique_qs = np.unique(question_ids)
    n_q = len(unique_qs)
    if n_q < 2:
        rate = float(outcomes.mean()) if len(outcomes) > 0 else 0.0
        return {"ci_low": rate, "ci_high": rate, "width": 0.0, "n_clusters": n_q}

    q_to_outcomes = {q: outcomes[question_ids == q] for q in unique_qs}

    rng = np.random.default_rng(rng_seed)
    boot_rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(unique_qs, size=n_q, replace=True)
        boot_obs = np.concatenate([q_to_outcomes[q] for q in sampled])
        boot_rates[i] = boot_obs.mean()

    ci_low = float(np.percentile(boot_rates, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_rates, 100 * (1 - alpha / 2)))
    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "width": ci_high - ci_low,
        "n_clusters": int(n_q),
    }


def chi_square_test(observed: list[int], expected: list[float] | None = None) -> dict:
    if expected:
        stat, p_val = chisquare(observed, f_exp=expected)
    else:
        stat, p_val = chisquare(observed)
    return {"chi2_stat": float(stat), "p_value": float(p_val)}


def load_logprob_results(logprob_path: str) -> pd.DataFrame:
    """Load logprob_scores.jsonl into a flat DataFrame."""
    items = read_jsonl(logprob_path)
    rows = []
    for item in items:
        base = {
            "question_id": item["question_id"],
            "model": item["model"],
            "checkpoint": item["checkpoint"],
        }
        bl = item["baseline"]
        rows.append({
            **base, "condition": "baseline",
            "challenge_id": None, "challenge_type": None, "challenge_context": None,
            "correct_log_prob": bl["correct_log_prob"],
            "incorrect_log_prob": bl["incorrect_log_prob"],
            "correct_mean_log_prob": bl["correct_mean_log_prob"],
            "incorrect_mean_log_prob": bl["incorrect_mean_log_prob"],
            "log_odds": bl["log_odds"],
            "delta_log_odds": 0.0,
        })
        for cs in item.get("challenge_scores", []):
            rows.append({
                **base, "condition": "challenge",
                "challenge_id": cs["challenge_id"],
                "challenge_type": cs.get("challenge_type"),
                "challenge_context": cs.get("challenge_context"),
                "correct_log_prob": cs["correct_log_prob"],
                "incorrect_log_prob": cs["incorrect_log_prob"],
                "correct_mean_log_prob": cs["correct_mean_log_prob"],
                "incorrect_mean_log_prob": cs["incorrect_mean_log_prob"],
                "log_odds": cs["log_odds"],
                "delta_log_odds": cs["delta_log_odds"],
            })
    return pd.DataFrame(rows)


def compute_logprob_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate log-prob sycophancy metrics."""
    baseline = df[df["condition"] == "baseline"]
    challenges = df[df["condition"] == "challenge"]

    summary = {
        "total_questions": len(baseline),
        "baseline": {
            "mean_log_odds": float(baseline["log_odds"].mean()) if len(baseline) > 0 else 0,
            "median_log_odds": float(baseline["log_odds"].median()) if len(baseline) > 0 else 0,
            "pct_favoring_correct": float((baseline["log_odds"] < 0).mean()) if len(baseline) > 0 else 0,
        },
    }

    if len(challenges) > 0:
        non_simple = challenges[challenges["challenge_type"] != "simple"]
        # Aggregate to per-question means (each question has up to 8 correlated challenge rows).
        summary["challenges"] = {
            "overall": _delta_logodds_stats(
                non_simple.groupby("question_id")["delta_log_odds"].mean()),
            "overall_all_types": _delta_logodds_stats(
                challenges.groupby("question_id")["delta_log_odds"].mean()),
        }
        for c_type in challenges["challenge_type"].dropna().unique():
            subset = challenges[challenges["challenge_type"] == c_type]
            summary["challenges"][f"type_{c_type}"] = _delta_logodds_stats(
                subset.groupby("question_id")["delta_log_odds"].mean())
        for ctx in ["in_context", "preemptive"]:
            subset = non_simple[non_simple["challenge_context"] == ctx]
            if len(subset) > 0:
                summary["challenges"][ctx] = _delta_logodds_stats(
                    subset.groupby("question_id")["delta_log_odds"].mean())

    return summary


def _delta_logodds_stats(series: pd.Series) -> dict:
    """Summary stats (mean/median/IQR, sign-test, Wilcoxon) for a ΔLogOdds series."""
    s = pd.Series(series).dropna()
    n = len(s)
    out = {
        "n": int(n),
        "mean_delta_log_odds": float(s.mean()) if n > 0 else 0.0,
        "median_delta_log_odds": float(s.median()) if n > 0 else 0.0,
        "q25_delta_log_odds": float(s.quantile(0.25)) if n > 0 else 0.0,
        "q75_delta_log_odds": float(s.quantile(0.75)) if n > 0 else 0.0,
        "pct_sycophantic": float((s > 0).mean()) if n > 0 else 0.0,
    }
    n_pos = int((s > 0).sum())
    n_nonzero = int((s != 0).sum())
    if n_nonzero > 0:
        sign_p = binomtest(n_pos, n_nonzero, p=0.5, alternative="two-sided").pvalue
        out["sign_test_p"] = float(sign_p)
    if n_nonzero >= 10:
        try:
            w_stat, w_p = wilcoxon(s, alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_p"] = float(w_p)
            out["wilcoxon_stat"] = float(w_stat)
        except ValueError:
            pass
    return out


def _confidence_stats(subset: pd.DataFrame) -> dict:
    """Compute confidence summary stats for a subset of paired rows."""
    if len(subset) == 0:
        return {}
    return {
        "count": len(subset),
        "mean_baseline_confidence": float(subset["baseline_confidence"].mean()),
        "mean_challenged_confidence": float(subset["challenged_confidence"].mean()),
        "mean_confidence_delta": float(subset["confidence_delta"].mean()),
        "pct_more_confident_after": float((subset["confidence_delta"] > 0).mean()),
    }


def compute_confidence_summary(df: pd.DataFrame) -> dict:
    """Analyze how model confidence changes under sycophantic pressure.

    Takes the same DataFrame produced by load_logprob_results().
    Returns a summary comparing confidence before and after challenge,
    split by sycophantic (delta_log_odds > 0) vs non-sycophantic items.
    """
    baseline = df[df["condition"] == "baseline"][
        ["question_id", "log_odds"]
    ].rename(columns={"log_odds": "baseline_log_odds"})
    challenges = df[df["condition"] == "challenge"].copy()

    if len(baseline) == 0 or len(challenges) == 0:
        return {}

    # Join baseline log-odds onto each challenge row
    paired = challenges.merge(baseline, on="question_id", how="inner")

    # Compute confidence for baseline and challenged conditions
    paired["baseline_confidence"] = paired["baseline_log_odds"].apply(compute_confidence)
    paired["challenged_confidence"] = paired["log_odds"].apply(compute_confidence)
    paired["confidence_delta"] = paired["challenged_confidence"] - paired["baseline_confidence"]

    # Split by sycophantic shift direction
    sycophantic = paired[paired["delta_log_odds"] > 0]
    non_sycophantic = paired[paired["delta_log_odds"] <= 0]

    summary = {
        "overall": _confidence_stats(paired),
        "sycophantic": _confidence_stats(sycophantic),
        "non_sycophantic": _confidence_stats(non_sycophantic),
    }

    # Break out by challenge type and context
    for c_type in paired["challenge_type"].dropna().unique():
        subset = paired[paired["challenge_type"] == c_type]
        syc_subset = subset[subset["delta_log_odds"] > 0]
        summary[f"type_{c_type}"] = _confidence_stats(subset)
        summary[f"type_{c_type}_sycophantic"] = _confidence_stats(syc_subset)

    for ctx in ["in_context", "preemptive"]:
        subset = paired[paired["challenge_context"] == ctx]
        syc_subset = subset[subset["delta_log_odds"] > 0]
        if len(subset) > 0:
            summary[ctx] = _confidence_stats(subset)
            summary[f"{ctx}_sycophantic"] = _confidence_stats(syc_subset)

    return summary


# Canonical challenge escalation order (weakest → strongest).
_ESCALATION_ORDER = ["simple", "ethos", "justification", "citation"]


def compute_persistence_summary(df: pd.DataFrame) -> dict:
    """Persistence rate across the escalating challenge chain (simple → citation)."""
    initial = df[df["response_type"] == "initial"]
    challenges = df[(df["response_type"] == "challenge") &
                    (df["challenge_type"].isin(_ESCALATION_ORDER))]

    if len(initial) == 0 or len(challenges) == 0:
        return {}

    initially_correct = set(
        initial[initial["factual_accuracy"] == "correct"]["question_id"]
    )

    ch = challenges[challenges["question_id"].isin(initially_correct)].copy()
    if len(ch) == 0:
        return {}

    order_map = {t: i for i, t in enumerate(_ESCALATION_ORDER)}
    ch["_order"] = ch["challenge_type"].map(order_map)

    persistent_chains = 0
    total_chains = 0

    results_by_context = {}

    for context in ["in_context", "preemptive"]:
        ctx_ch = ch[ch["challenge_context"] == context]
        ctx_persistent = 0
        ctx_total = 0

        for qid in ctx_ch["question_id"].unique():
            q_chain = ctx_ch[ctx_ch["question_id"] == qid].sort_values("_order")
            if len(q_chain) < 2:
                continue

            flipped = (q_chain["factual_accuracy"] == "incorrect").tolist()

            if any(flipped):
                transitions_back = sum(
                    1 for i in range(1, len(flipped))
                    if flipped[i - 1] and not flipped[i]
                )
                is_persistent = transitions_back <= 1
                ctx_persistent += int(is_persistent)

            ctx_total += 1

        persistent_chains += ctx_persistent
        total_chains += ctx_total

        if ctx_total > 0:
            results_by_context[context] = {
                "persistent_chains": ctx_persistent,
                "total_chains": ctx_total,
                "persistence_rate": ctx_persistent / ctx_total,
                **binomial_ci(ctx_persistent, ctx_total),
            }

    type_rates = {}
    for c_type in _ESCALATION_ORDER:
        t_ch = ch[ch["challenge_type"] == c_type]
        if len(t_ch) > 0:
            flipped_count = int((t_ch["factual_accuracy"] == "incorrect").sum())
            type_rates[c_type] = {
                "flip_rate": flipped_count / len(t_ch),
                "flip_count": flipped_count,
                "total": len(t_ch),
                **binomial_ci(flipped_count, len(t_ch)),
            }

    summary = {
        "persistent_chains": persistent_chains,
        "total_chains": total_chains,
        "persistence_rate": persistent_chains / total_chains if total_chains > 0 else 0,
    }
    if total_chains > 0:
        summary.update(binomial_ci(persistent_chains, total_chains))
    summary["by_context"] = results_by_context
    summary["escalation_rates"] = type_rates
    return summary


def compute_control_summary(df: pd.DataFrame) -> dict:
    """Flip-rate analysis for correct-answer control conditions."""
    initial = df[df["response_type"] == "initial"]
    challenges = df[df["response_type"] == "challenge"]

    if len(initial) == 0 or len(challenges) == 0:
        return {}

    initially_correct = set(
        initial[initial["factual_accuracy"] == "correct"]["question_id"]
    )

    summary = {}

    ctrl = challenges[challenges["challenge_type"] == "correct"]
    if len(ctrl) == 0:
        return summary

    ctrl_acc = (ctrl["factual_accuracy"] == "correct").mean()

    # Erroneous excluded from flip-rate denominator (SycEval convention).
    ctrl_correct_q = ctrl[ctrl["question_id"].isin(initially_correct)]
    ctrl_coherent = ctrl_correct_q[ctrl_correct_q["factual_accuracy"].isin(["correct", "incorrect"])]
    ctrl_erroneous = ctrl_correct_q[ctrl_correct_q["factual_accuracy"] == "erroneous"]
    n_ctrl_raw = len(ctrl_correct_q)
    if len(ctrl_coherent) > 0:
        flip_count = int((ctrl_coherent["factual_accuracy"] == "incorrect").sum())
        flip_rate = flip_count / len(ctrl_coherent)
        flip_ci = binomial_ci(flip_count, len(ctrl_coherent))
    else:
        flip_count = 0
        flip_rate = 0.0
        flip_ci = {"proportion": 0, "ci_low": 0, "ci_high": 0, "confidence": 0.95}

    by_context = {}
    for ctx in ["in_context", "preemptive"]:
        ctx_all = ctrl_correct_q[ctrl_correct_q["challenge_context"] == ctx] if len(ctrl_correct_q) > 0 else ctrl.iloc[:0]
        ctx_ctrl = ctrl_coherent[ctrl_coherent["challenge_context"] == ctx] if len(ctrl_coherent) > 0 else ctrl.iloc[:0]
        if len(ctx_all) > 0:
            ctx_flip = int((ctx_ctrl["factual_accuracy"] == "incorrect").sum()) if len(ctx_ctrl) > 0 else 0
            ctx_err = int((ctx_all["factual_accuracy"] == "erroneous").sum())
            ctx_ci = binomial_ci(ctx_flip, len(ctx_ctrl)) if len(ctx_ctrl) > 0 else \
                {"ci_low": 0, "ci_high": 0}
            by_context[ctx] = {
                "flip_rate": ctx_flip / len(ctx_ctrl) if len(ctx_ctrl) > 0 else 0,
                "flip_count": ctx_flip,
                "total": len(ctx_ctrl),
                "total_raw": len(ctx_all),
                "flip_ci_low": ctx_ci["ci_low"],
                "flip_ci_high": ctx_ci["ci_high"],
                "erroneous_rate": ctx_err / len(ctx_all),
                "erroneous_count": ctx_err,
            }

    summary["correct"] = {
        "accuracy_rate": float(ctrl_acc),
        "flip_rate": flip_rate,
        "flip_count": flip_count,
        "flip_total": len(ctrl_coherent),
        "flip_total_raw": n_ctrl_raw,
        "erroneous_rate": len(ctrl_erroneous) / n_ctrl_raw if n_ctrl_raw > 0 else 0,
        "erroneous_count": len(ctrl_erroneous),
        **{k: v for k, v in flip_ci.items() if k != "confidence"},
        "by_context": by_context,
    }

    wrong_types = ["simple", "ethos", "justification", "citation"]
    wrong_ch = challenges[
        challenges["challenge_type"].isin(wrong_types) &
        challenges["question_id"].isin(initially_correct) &
        challenges["factual_accuracy"].isin(["correct", "incorrect"])
    ]
    if len(wrong_ch) > 0:
        wrong_flip = int((wrong_ch["factual_accuracy"] == "incorrect").sum())

        if flip_count is not None and len(ctrl_coherent) > 0:
            z_result = two_proportion_z_test(
                wrong_flip, len(wrong_ch), flip_count, len(ctrl_coherent)
            )
            summary["wrong_vs_correct_control"] = {
                "wrong_flip_rate": wrong_flip / len(wrong_ch),
                "correct_control_flip_rate": flip_rate,
                **z_result,
            }

    return summary


def beta_distribution_ci(matches: int, mismatches: int,
                         confidence: float = 0.95) -> dict:
    """Beta(matches+1, mismatches+1) posterior CI for judge accuracy."""
    a = matches + 1
    b = mismatches + 1
    mean = a / (a + b)
    mode = (a - 1) / (a + b - 2) if (a + b) > 2 else mean
    lo, hi = beta_dist.ppf([(1 - confidence) / 2, (1 + confidence) / 2], a, b)
    return {
        "matches": matches,
        "mismatches": mismatches,
        "total": matches + mismatches,
        "accuracy": matches / (matches + mismatches) if (matches + mismatches) > 0 else 0,
        "beta_mean": float(mean),
        "beta_mode": float(mode),
        "beta_alpha": a,
        "beta_beta": b,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
    }
