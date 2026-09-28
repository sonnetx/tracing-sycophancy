#!/usr/bin/env python3
"""Instruction-following control figures (belief vs command vs truth-orthogonal).

Reads data/if_controls_pe_summary.json (one record per pipeline x stage x domain,
each with belief/command/truth_orthogonal flip% and dLogOdds) and writes two
full-width PNGs into paper/figures/:

  if_controls_dissociation.png  -- finals, 4 pipelines, 2x2 (domain x track):
      behavior (flip%) diverges (command >> belief) where preference (dLogOdds)
      is equal (belief ~ command).  The contrast IS the compliance result.
  if_controls_trajectory.png    -- SFT->DPO->Final, command (solid) vs belief
      (dashed) per pipeline, comp|med: the command>belief gap never closes.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "if_controls_pe_summary.json")
OUT  = os.path.join(ROOT, "paper", "figures")

BELIEF, COMMAND = "#4C72B0", "#DD8452"          # blue, orange
PIPE_COL = {"Think": "#C44E52", "Instruct": "#4C72B0", "Tulu": "#55A868"}
DOMS = [("computational", "Computational"), ("medical_advice", "Medical")]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

recs = json.load(open(DATA, encoding="utf-8"))
idx = {(r["pipeline"], r["stage"], r["domain"]): r for r in recs}


def _labels(ax, xs, vals, fmt, dy, fs=7):
    for x, v in zip(xs, vals):
        if v is None:
            continue
        ax.text(x, v + dy, fmt % v, ha="center", va="bottom", fontsize=fs)


# ---------------------------------------------------------------- dissociation
def dissociation():
    pipes = [("Instruct", "OLMo\nInstruct"), ("Llama", "Llama\nInstruct"),
             ("Tulu", "Tulu 3"), ("Think", "OLMo Think\n(reasoning)")]
    x = np.arange(len(pipes)); w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for ri, (dom, domlab) in enumerate(DOMS):
        bf = [idx[(p, "Final", dom)]["belief"]["flip"] for p, _ in pipes]
        cf = [idx[(p, "Final", dom)]["command"]["flip"] for p, _ in pipes]
        bd = [idx[(p, "Final", dom)]["belief"]["dlo"] for p, _ in pipes]
        cd = [idx[(p, "Final", dom)]["command"]["dlo"] for p, _ in pipes]
        axf, axd = axes[ri][0], axes[ri][1]
        axf.bar(x - w/2, bf, w, color=BELIEF, label="belief")
        axf.bar(x + w/2, cf, w, color=COMMAND, label="command")
        axd.bar(x - w/2, bd, w, color=BELIEF)
        axd.bar(x + w/2, cd, w, color=COMMAND)
        _labels(axf, x - w/2, bf, "%.0f", 1.5); _labels(axf, x + w/2, cf, "%.0f", 1.5)
        _labels(axd, x - w/2, bd, "%.2f", .02); _labels(axd, x + w/2, cd, "%.2f", .02)
        axf.set_ylim(0, 100); axd.set_ylim(0, 1.45)
        axf.set_ylabel(domlab, fontsize=13, fontweight="bold")
        for ax in (axf, axd):
            ax.set_xticks(x)
            ax.set_xticklabels([l for _, l in pipes] if ri == 1 else [""] * len(pipes),
                               fontsize=9)
            ax.axvline(2.5, color="0.7", ls=":", lw=1)   # set reasoning pipeline apart
    axes[0][0].set_title("Behavior: regressive flip rate (%)", fontsize=12)
    axes[0][1].set_title(r"Preference: $\Delta$LogOdds", fontsize=12)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "if_controls_dissociation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ----------------------------------------------------------------- trajectory
def trajectory():
    stages = ["SFT", "DPO", "Final"]
    tpipes = [("Think", "OLMo Think"), ("Instruct", "OLMo Instruct"), ("Tulu", "Tulu 3")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, (dom, domlab) in zip(axes, DOMS):
        for p, _ in tpipes:
            col = PIPE_COL[p]
            cf = [idx[(p, s, dom)]["command"]["flip"] for s in stages]
            bf = [idx[(p, s, dom)]["belief"]["flip"] for s in stages]
            ax.plot(stages, cf, "-o", color=col, lw=2, ms=6)
            ax.plot(stages, bf, "--o", color=col, lw=2, ms=6, mfc="white")
        ax.set_title(domlab, fontsize=12); ax.set_ylim(0, 100)
        ax.set_xlabel("training stage")
    axes[0].set_ylabel("regressive flip rate (%)")
    handles = [Line2D([0], [0], color=PIPE_COL[p], lw=3, label=l) for p, l in tpipes]
    handles += [Line2D([0], [0], color="0.4", lw=2, ls="-", label="command"),
                Line2D([0], [0], color="0.4", lw=2, ls="--", label="belief")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    p = os.path.join(OUT, "if_controls_trajectory.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dissociation()
    trajectory()
