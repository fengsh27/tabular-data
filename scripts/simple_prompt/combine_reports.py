#!/usr/bin/env python3
"""Join the source-grounded self-evaluation to the gold score, one JSON per run.

The question this answers: does a confidence computed WITHOUT the answer key
predict the score computed WITH it? If it does, the fast path can be gated on
confidence for papers that have no gold, and the two-tier design works. If it
does not, the gate is decoration.

    python scripts/simple_prompt/combine_reports.py \\
        --eval scores/eval_qwen38.json --score scores/score_qwen38.json \\
        --out scores/combined_qwen38.json --label qwen3.8
"""

from __future__ import annotations

import argparse
import json
import math


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    ev = json.load(open(args.eval))
    sc = json.load(open(args.score))
    by_pmid = {p["pmid"]: p for p in sc["per_paper"]}

    papers, xs, ys = {}, [], []
    for pmid, e in ev["papers"].items():
        g = by_pmid.get(pmid)
        rec = {"self_eval": {"confidence": e["confidence"],
                             "deterministic": e["deterministic"],
                             "llm": e["llm"]}}
        if g:
            rec_prec, rec_rec = g["row_precision"], g["row_recall"]
            f1 = (2 * rec_prec * rec_rec / (rec_prec + rec_rec)
                  if (rec_prec + rec_rec) else 0.0)
            rec["gold_score"] = {**g, "row_f1": round(f1, 4)}
            xs.append(e["confidence"])
            ys.append(f1)
        else:
            rec["gold_score"] = None  # no manual baseline for this paper
        papers[pmid] = rec

    out = {
        "label": args.label,
        "llm_enabled": ev.get("llm_enabled", False),
        "totals": sc.get("totals", {}),
        "column_accuracy": sc.get("column_accuracy", {}),
        "mean_confidence": ev.get("mean_confidence"),
        "gate_check": {
            "note": "does confidence (gold not used) predict row_f1 (gold used)?",
            "papers_compared": len(xs),
            "pearson_confidence_vs_row_f1": pearson(xs, ys),
            "spearman_confidence_vs_row_f1": spearman(xs, ys),
        },
        "papers": papers,
    }
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n=== combined: {args.label} ===\n")
    print(f"{'PMID':<10} {'confidence':>11} {'row_f1':>8} {'row rec':>8} {'row prec':>9}")
    print("-" * 50)
    for pmid in sorted(papers):
        r = papers[pmid]
        g = r["gold_score"]
        if g:
            print(f"{pmid:<10} {r['self_eval']['confidence']:>11.3f} "
                  f"{g['row_f1']:>8.3f} {100*g['row_recall']:>7.1f}% "
                  f"{100*g['row_precision']:>8.1f}%")
        else:
            print(f"{pmid:<10} {r['self_eval']['confidence']:>11.3f} "
                  f"{'no gold':>8}")
    print("-" * 50)
    gc = out["gate_check"]
    print(f"pearson  {gc['pearson_confidence_vs_row_f1']}   "
          f"spearman {gc['spearman_confidence_vs_row_f1']}   "
          f"(n={gc['papers_compared']})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
