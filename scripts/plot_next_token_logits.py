#!/usr/bin/env python3
"""Plot next-token probability distributions as rank-probability curves.

For each (question_id, challenge_id), produces one figure with per-pipeline
subpanels. Each subpanel shows the sorted top-K probability decay as a line,
with one line per temperature (applied by recomputing softmax(logp / T) on
the extracted top-K). Sharp modes look like a near-vertical drop; soft
modes decay gradually.

Usage:
    python scripts/plot_next_token_logits.py \\
        --input-dir data/results/logits_probe \\
        --output-dir paper/figures/logits_probe
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PIPELINE_ORDER = [
    "olmo3-7b-instruct", "tulu3-llama31-8b",
    "llama31-8b-instruct", "olmo3-7b-think",
]
PIPELINE_LABEL = {
    "olmo3-7b-instruct":     "OLMo Instruct",
    "tulu3-llama31-8b":      "Tulu 3",
    "llama31-8b-instruct":   "Llama 3.1 Instruct",
    "olmo3-7b-think":        "OLMo Think",
}
TEMPERATURES = [0.3, 0.7, 1.0, 1.5]
TEMP_COLORS = {
    0.3: "#2c7fb8",
    0.7: "#41b6c4",
    1.0: "#ffaa33",
    1.5: "#d7191c",
}


def load_all(input_dir: str) -> dict:
    """Return {(qid, cid): {model: row}}."""
    out = {}
    for fpath in glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                key = (r["question_id"], r["challenge_id"])
                out.setdefault(key, {})[r["model"]] = r
    return out


def softmax_at_temp(logps: list[float], T: float) -> np.ndarray:
    """softmax(logp / T) over the top-K (no renormalization of tail mass)."""
    arr = np.array(logps, dtype=float) / T
    arr -= arr.max()
    probs = np.exp(arr)
    probs /= probs.sum()
    return probs


def _unwrap(s: str) -> str:
    s = s.strip()
    for pre, suf in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(pre) and s.endswith(suf) and len(s) > len(pre) + len(suf):
            return s[len(pre):-len(suf)].strip()
    return s


def _strip_tok(t: str) -> str:
    return t.strip().lstrip("$").lstrip()


def _find_rank(tokens: list, target_first_char: str, avoid_char: str = "") -> int | None:
    """Return 1-indexed rank of first token whose stripped form starts with
    target_first_char (and differs from avoid_char, to disambiguate correct/wrong
    when the first non-wrapper char is the same)."""
    if not target_first_char or target_first_char == avoid_char:
        return None
    for i, tok in enumerate(tokens):
        s = _strip_tok(tok)
        if s and s[:1] == target_first_char:
            return i + 1
    return None


def plot_one(key: tuple, rows_by_model: dict, out_path: str,
             topk_shown: int = 20, log_y: bool = True) -> None:
    models_present = [m for m in PIPELINE_ORDER if m in rows_by_model]
    if not models_present:
        return
    fig, axes = plt.subplots(1, len(models_present),
                              figsize=(4.2 * len(models_present), 4.6),
                              sharey=True)
    if len(models_present) == 1:
        axes = [axes]
    qid, cid = key
    first = rows_by_model[models_present[0]]
    correct = _unwrap(str(first.get("correct_answer", "")))
    wrong = _unwrap(str(first.get("proposed_wrong_answer", "")))
    pc = correct[:1] if correct else ""
    pw = wrong[:1] if wrong else ""

    for ax, m in zip(axes, models_present):
        r = rows_by_model[m]
        topk_rows = r.get("topk", [])[:topk_shown]
        tokens = [t["token"] for t in topk_rows]
        logps = [t["logprob"] for t in topk_rows]
        if not logps:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        n = len(logps)
        ranks = np.arange(1, n + 1)

        # Find correct/wrong positions in the top-K (1-indexed; None if absent)
        rank_c = _find_rank(tokens, pc, avoid_char=pw)
        rank_w = _find_rank(tokens, pw, avoid_char=pc)

        for T in TEMPERATURES:
            probs = softmax_at_temp(logps, T)
            # At a single item, the rank order is identical across T (softmax
            # is monotone in logits/T), so we can plot probs in original order.
            ax.plot(ranks, probs, marker="o", markersize=4,
                    linewidth=2, color=TEMP_COLORS[T], label=f"T={T}",
                    alpha=0.85, zorder=2)
            # Overlay big green/red markers on this curve at the
            # correct/wrong positions so the eye immediately spots them.
            if rank_c is not None:
                ax.scatter([rank_c], [probs[rank_c - 1]], s=110,
                           facecolor=TEMP_COLORS[T], edgecolor="#2ca02c",
                           linewidth=2.5, zorder=5)
            if rank_w is not None:
                ax.scatter([rank_w], [probs[rank_w - 1]], s=110,
                           facecolor=TEMP_COLORS[T], edgecolor="#d62728",
                           linewidth=2.5, marker="s", zorder=5)

        # Add a corner text box with P(correct) and P(wrong) at T=1.0
        probs_T1 = softmax_at_temp(logps, 1.0)
        p_c = probs_T1[rank_c - 1] if rank_c is not None else 0.0
        p_w = probs_T1[rank_w - 1] if rank_w is not None else 0.0
        if rank_c is not None or rank_w is not None:
            text_c = f"P(correct)={p_c:.3f}" + (f" @rank {rank_c}" if rank_c else "")
            text_w = f"P(wrong)={p_w:.3f}" + (f" @rank {rank_w}" if rank_w else " (not in top-20)")
            ax.text(0.97, 0.97, f"At T=1.0:\n{text_c}\n{text_w}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.92))

        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.2)
        else:
            ax.set_ylim(0, 1.05)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_xlabel("Rank (top-20 only; tail truncated)")
        ax.set_title(PIPELINE_LABEL.get(m, m), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("P(token | challenged prompt)")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    suptitle = (f"Rank-probability decay at decision point (qid={qid}, {cid})    |    "
                f"correct={correct!r} (○ green), wrong={wrong!r} (□ red)")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_paired_bars(key: tuple, rows_by_model: dict, out_path: str) -> None:
    """Option B: paired bars of P(correct) vs P(wrong) per pipeline at T=1.0.
    Clean, unambiguous view for a single item."""
    models_present = [m for m in PIPELINE_ORDER if m in rows_by_model]
    if not models_present:
        return
    qid, cid = key
    first = rows_by_model[models_present[0]]
    correct = _unwrap(str(first.get("correct_answer", "")))
    wrong = _unwrap(str(first.get("proposed_wrong_answer", "")))
    pc = correct[:1] if correct else ""
    pw = wrong[:1] if wrong else ""

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(models_present)), 4.2))
    x = np.arange(len(models_present))
    bar_w = 0.38
    p_corrects, p_wrongs = [], []
    for m in models_present:
        topk_rows = rows_by_model[m].get("topk", [])
        tokens = [t["token"] for t in topk_rows]
        logps = [t["logprob"] for t in topk_rows]
        if not logps:
            p_corrects.append(0.0); p_wrongs.append(0.0); continue
        probs = softmax_at_temp(logps, 1.0)
        rc = _find_rank(tokens, pc, avoid_char=pw)
        rw = _find_rank(tokens, pw, avoid_char=pc)
        p_corrects.append(probs[rc - 1] if rc else 0.0)
        p_wrongs.append(probs[rw - 1] if rw else 0.0)

    bars_c = ax.bar(x - bar_w / 2, p_corrects, bar_w, color="#2ca02c",
                    edgecolor="black", linewidth=0.6, label=f"P(correct = {correct!r})")
    bars_w = ax.bar(x + bar_w / 2, p_wrongs, bar_w, color="#d62728",
                    edgecolor="black", linewidth=0.6, label=f"P(wrong = {wrong!r})")
    for b, v in list(zip(bars_c, p_corrects)) + list(zip(bars_w, p_wrongs)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABEL.get(m, m) for m in models_present],
                       fontsize=10)
    ax.set_ylabel("P(token | challenged prompt) at T=1.0")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Correct vs wrong next-token probability (qid={qid}, {cid})",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/results/logits_probe")
    parser.add_argument("--output-dir", default="paper/figures/logits_probe")
    parser.add_argument("--topk-shown", type=int, default=20)
    parser.add_argument("--linear-y", action="store_true",
                        help="Use linear Y axis (default is log for decay visibility)")
    args = parser.parse_args()

    by_key = load_all(args.input_dir)
    if not by_key:
        print(f"No JSONL found under {args.input_dir}")
        return
    for key, rows in sorted(by_key.items()):
        qid, cid = key
        out_curve = os.path.join(args.output_dir, f"{qid}_{cid}.png")
        plot_one(key, rows, out_curve,
                 topk_shown=args.topk_shown, log_y=not args.linear_y)
        out_bars = os.path.join(args.output_dir, f"{qid}_{cid}_paired.png")
        plot_paired_bars(key, rows, out_bars)


if __name__ == "__main__":
    main()
