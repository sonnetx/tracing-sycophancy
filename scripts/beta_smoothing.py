#!/usr/bin/env python3
import argparse
import json
import os


MODEL_ORDER = [
    ("olmo3-7b-base", "OLMo 3 Base", None, None),
    ("olmo3-7b-think-sft", "Think SFT", "Think", "SFT"),
    ("olmo3-7b-think-dpo", "Think DPO", "Think", "DPO"),
    ("olmo3-7b-think", "Think", "Think", "Think"),
    ("olmo3-7b-instruct-sft", "Instruct SFT", "Instruct", "SFT"),
    ("olmo3-7b-instruct-dpo", "Instruct DPO", "Instruct", "DPO"),
    ("olmo3-7b-instruct", "Instruct", "Instruct", "Instruct"),
    ("llama31-8b-base", "Llama 3.1 Base", None, None),
    ("llama31-8b-instruct", "Llama 3.1 Instruct", "Llama 3.1", "Instruct"),
    ("tulu3-llama31-8b-sft", "Tulu 3 SFT", "Tulu 3", "SFT"),
    ("tulu3-llama31-8b-dpo", "Tulu 3 DPO", "Tulu 3", "DPO"),
    ("tulu3-llama31-8b", "Tulu 3", "Tulu 3", "Tulu 3"),
]


def posterior(count: int, total: int) -> float:
    """Beta(1,1) posterior mean: (k+1)/(n+2)."""
    return (count + 1) / (total + 2)


def shift_pp(raw: float, post: float) -> float:
    """Shift in percentage points (raw - posterior)."""
    return (raw - post) * 100


def load(dataset: str, name: str, root: str) -> dict:
    p = os.path.join(root, dataset, "analysis", f"{name}.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)


def build_rows(root: str) -> list:
    """Build rows: (model_display, domain, metric, count, total, raw, post)."""
    rows = []
    for dataset, dom_label in [("computational", "Comp."), ("medical_advice", "Med.")]:
        summ = load(dataset, "summaries", root)
        ctrl = load(dataset, "control_summaries", root)
        matched = load(dataset, "matched_summaries", root)

        for mk, display, _, _ in MODEL_ORDER:
            s = summ.get(mk)
            if s:
                syc = s.get("sycophancy", {})
                if syc.get("regressive_total", 0) > 0:
                    rc = syc["regressive_count"]
                    rt = syc["regressive_total"]
                    raw = rc / rt
                    post = posterior(rc, rt)
                    rows.append((display, dom_label, "Regr", rc, rt, raw, post))
            c = ctrl.get(mk)
            if c:
                co = c.get("correct", {})
                if co.get("flip_total", 0) > 0:
                    cc = co["flip_count"]
                    ct = co["flip_total"]
                    raw = cc / ct
                    post = posterior(cc, ct)
                    rows.append((display, dom_label, "Ctrl", cc, ct, raw, post))

        # matched_summaries format: {pipe_name: [{stage, base_regressive_on_intersection, stage_regressive_on_intersection, n_intersection}, ...]}
        for pipe_name, pipe_records in matched.items():
            for rec in pipe_records:
                if rec.get("stage") == "Base":
                    continue
                stg = rec["stage"]
                label = f"{pipe_name} {stg}"
                n = rec["n_intersection"]
                # base on intersection
                b = rec.get("base_regressive_on_intersection", {})
                if b.get("regressive_total", 0) > 0:
                    rc = b["regressive_count"]
                    rt = b["regressive_total"]
                    raw = rc / rt
                    post = posterior(rc, rt)
                    rows.append((label, dom_label, "Matched Base", rc, rt, raw, post))
                # stage on intersection
                s2 = rec.get("stage_regressive_on_intersection", {})
                if s2.get("regressive_total", 0) > 0:
                    rc = s2["regressive_count"]
                    rt = s2["regressive_total"]
                    raw = rc / rt
                    post = posterior(rc, rt)
                    rows.append((label, dom_label, "Matched Stage", rc, rt, raw, post))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX tabular for Appendix F")
    parser.add_argument("--shift-threshold", type=float, default=5.0,
                        help="Flag cells where |shift| > threshold pp (default 5.0)")
    parser.add_argument("--n-threshold", type=int, default=30,
                        help="Flag cells where coherent n < threshold (default 30)")
    args = parser.parse_args()

    rows = build_rows(args.experiment_dir)

    # Compute stats across all rows
    max_shift = max(abs(shift_pp(r[5], r[6])) for r in rows) if rows else 0
    flagged_shift = [r for r in rows if abs(shift_pp(r[5], r[6])) > args.shift_threshold]
    flagged_n = [r for r in rows if r[4] < args.n_threshold]

    print(f"Total cells analysed: {len(rows)}")
    print(f"Max |shift| across all cells: {max_shift:.2f} pp")
    print(f"Cells with |shift| > {args.shift_threshold} pp: {len(flagged_shift)}")
    print(f"Cells with n < {args.n_threshold}: {len(flagged_n)}")
    print()

    if flagged_shift:
        print("--- Cells with largest shifts under Beta(1,1) smoothing ---")
        print(f"{'Model':<22s} {'Dom':<6s} {'Metric':<14s} {'n':>5s} {'raw':>7s} {'post':>7s} {'shift':>7s}")
        print("-" * 78)
        for r in sorted(flagged_shift, key=lambda x: -abs(shift_pp(x[5], x[6]))):
            display, dom, metric, cnt, tot, raw, post = r
            print(f"{display:<22s} {dom:<6s} {metric:<14s} {tot:>5d} "
                  f"{raw*100:>6.1f}% {post*100:>6.1f}% "
                  f"{shift_pp(raw, post):>+6.1f}")

    if args.latex:
        print("\n% ====== Appendix F: Beta(1,1) smoothing ======")
        print("\\begin{tabular}{l l l rrrr}")
        print("  \\toprule")
        print("  \\textbf{Model} & \\textbf{Domain} & \\textbf{Metric} "
              "& \\textbf{n} & \\textbf{Raw} & \\textbf{Posterior} & \\textbf{Shift (pp)} \\\\")
        print("  \\midrule")
        # Only print cells with shift > threshold or n < threshold — keep table compact
        for r in rows:
            display, dom, metric, cnt, tot, raw, post = r
            sh = shift_pp(raw, post)
            if abs(sh) <= args.shift_threshold and tot >= args.n_threshold:
                continue
            flag_n = "\\textbf" if tot < args.n_threshold else "\\phantom{\\textbf}"
            display_esc = display.replace("&", "\\&")
            print(f"  {display_esc} & {dom} & {metric} & "
                  f"{flag_n}{{{tot}}} & "
                  f"{raw*100:.1f}\\% & "
                  f"{post*100:.1f}\\% & "
                  f"{sh:+.1f} \\\\")
        print("  \\bottomrule")
        print("\\end{tabular}")
        print("\n% Rows not in table: |shift| <= %.1f pp AND n >= %d. "
              "Summary: across all %d cells, max |shift| = %.2f pp."
              % (args.shift_threshold, args.n_threshold, len(rows), max_shift))


if __name__ == "__main__":
    main()
