#!/usr/bin/env python3
"""Score pk-individual predictions against the manual gold. No LLM, no network.

Two views, because they answer different questions:

  values  - bag-of-numbers recall/precision. Ignores row identity, so it says
            "did the model read the right numbers off the table" and nothing
            more. Comparable to the earlier 96%/74% figures.
  rows    - one-to-one row matching, then per-column accuracy over the matched
            pairs. This is the one that says whether the schema was filled in
            correctly.

    python scripts/simple_prompt/score_pk_individual.py \
        --pred results/simple_v2 --label simple-v2
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402
import aliases as alias_tables  # noqa: E402

DEFAULT_GOLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "benchmark", "data", "pk-individual", "baseline",
)

# Columns compared on matched rows, with the weight used to pick the best match.
# Patient ID and Analyte are the discriminative ones when many rows share a value.
WEIGHTS = {
    "Population": 1, "Pregnancy stage": 1, "Pediatric/Gestational age": 1,
    "Specimen": 1, "Drug name": 1, "Patient ID": 3, "Parameter type": 1,
    "Analyte": 3, "Time value": 1, "Time unit": 1, "Parameter unit": 1,
}

# The gold spells trimesters both ways and contains one typo; fold them together
# rather than scoring the model down for the gold's inconsistency.
SYNONYMS = {
    "first": "1st", "second": "2nd", "third": "3rd",
    "trimster": "trimester", "postpartum": "postpartum",
}


# U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU both mean "micro"; the gold
# spells it "u". This is spelling, not meaning, so it is always normalised.
MICRO = str.maketrans({"\u00b5": "u", "\u03bc": "u"})

ALIAS = {}  # populated from --aliases


def norm(text: str) -> str:
    s = (text or "").strip().lower().translate(MICRO)
    s = re.sub(r"\s+", " ", s).strip(" .;:")
    for a, b in SYNONYMS.items():
        s = re.sub(rf"\b{a}\b", b, s)
    return s


def norm_unit(text: str) -> str:
    """Units differ only by case, micro sign, and spacing: ug/l == microg/L."""
    return norm(text).replace(" ", "")


def canon(col: str, text: str) -> str:
    s = norm_unit(text) if col == "Parameter unit" else norm(text)
    s = ALIAS.get(col, {}).get(s, s)
    if col == "Specimen" and "/" in s:
        # A ratio names an unordered pair of specimens, and the gold itself
        # writes both orders ("maternal serum/umbilical cord blood" but also
        # "cord blood/maternal plasma"). Compare the pair, not the order.
        parts = [ALIAS.get(col, {}).get(x.strip(), x.strip()) for x in s.split("/")]
        s = "/".join(sorted(p for p in parts if p))
    return s


def cell_eq(col: str, a: str, b: str) -> bool:
    if col in ("Parameter value", "Time value"):
        na, nb = common.number(a), common.number(b)
        if na is not None and nb is not None:
            return abs(na - nb) <= 1e-6 * max(1.0, abs(na), abs(nb))
    return canon(col, a) == canon(col, b)


def numbers_of(rows):
    bag = collections.Counter()
    for r in rows:
        n = common.number(r.get("Parameter value", ""))
        if n is not None:
            bag[round(n, 6)] += 1
    return bag


def match_rows(pred, gold):
    """Greedy one-to-one matching. Equal Parameter value is a hard requirement."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if not cell_eq("Parameter value", p["Parameter value"], g["Parameter value"]):
                continue
            score = sum(w for c, w in WEIGHTS.items() if cell_eq(c, p[c], g[c]))
            pairs.append((score, i, j))
    pairs.sort(key=lambda t: -t[0])

    used_p, used_g, matched = set(), set(), []
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((pred[i], gold[j]))
    return matched


def load_pred(path, pmid):
    for cand in (os.path.join(path, pmid, "combined.csv"),
                 os.path.join(path, f"{pmid}.csv"),
                 os.path.join(path, pmid, "combined_final.csv")):
        if os.path.exists(cand):
            return common.read_csv(cand, pmid=pmid)
    return None


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "    - "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of <pmid>/combined.csv")
    ap.add_argument("--gold", default=os.path.normpath(DEFAULT_GOLD))
    ap.add_argument("--label", default="prediction")
    ap.add_argument("--aliases", choices=["off", "safe", "all"], default="off",
                    help="treat known vocabulary variants as equal "
                         "(see scripts/simple_prompt/aliases.py)")
    ap.add_argument("--gold-glob", default="*_baseline_manual.csv",
                    help="which gold files to score against. baseline/ holds two "
                         "curations per paper: *_baseline_manual.csv (narrower) and "
                         "*_baseline.csv (keeps derived columns such as the infant "
                         "dose percentages). They disagree, so name the one you mean.")
    ap.add_argument("--json", default=None)
    ap.add_argument("--confusions", default=None,
                    help="comma-separated columns: dump the top gold->pred disagreements")
    args = ap.parse_args()

    if args.aliases != "off":
        ALIAS.update(alias_tables.build(include_debatable=args.aliases == "all"))

    gold_files = sorted(glob.glob(os.path.join(args.gold, args.gold_glob)))
    gold_by_pmid = {os.path.basename(f).split("_")[0]: f for f in gold_files}

    per_paper, skipped = [], []
    tot = collections.Counter()
    col_hits = collections.Counter()
    want_conf = {c.strip() for c in (args.confusions or "").split(",") if c.strip()}
    confusions = collections.defaultdict(collections.Counter)

    # Accept either <pred>/<pmid>/combined.csv or a flat <pred>/<pmid>.csv.
    pmids = {d for d in os.listdir(args.pred)
             if os.path.isdir(os.path.join(args.pred, d)) and d.isdigit()}
    pmids |= {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.pred, "*.csv"))
              if os.path.basename(f)[:-4].isdigit()}
    pmids = sorted(pmids)

    for pmid in pmids:
        pred = load_pred(args.pred, pmid)
        if pred is None:
            skipped.append((pmid, "no prediction file"))
            continue
        if pmid not in gold_by_pmid:
            skipped.append((pmid, "no manual gold"))
            continue
        gold = common.read_csv(gold_by_pmid[pmid], pmid=pmid)

        matched = match_rows(pred, gold)
        pb, gb = numbers_of(pred), numbers_of(gold)
        overlap = sum((pb & gb).values())

        tot["pred"] += len(pred)
        tot["gold"] += len(gold)
        tot["matched"] += len(matched)
        tot["val_pred"] += sum(pb.values())
        tot["val_gold"] += sum(gb.values())
        tot["val_hit"] += overlap
        for p, g in matched:
            for c in WEIGHTS:
                if cell_eq(c, p[c], g[c]):
                    col_hits[c] += 1
                elif c in want_conf:
                    confusions[c][(g[c] or "<blank>", p[c] or "<blank>")] += 1

        per_paper.append({
            "pmid": pmid, "pred_rows": len(pred), "gold_rows": len(gold),
            "matched": len(matched),
            "row_recall": len(matched) / len(gold) if gold else 0.0,
            "row_precision": len(matched) / len(pred) if pred else 0.0,
            "value_recall": overlap / sum(gb.values()) if sum(gb.values()) else 0.0,
        })

    print(f"\n=== {args.label}  [aliases={args.aliases}] ===\n")
    print(f"{'PMID':<10} {'pred':>6} {'gold':>6} {'match':>6} "
          f"{'row rec':>8} {'row prec':>9} {'val rec':>8}")
    print("-" * 60)
    for r in per_paper:
        print(f"{r['pmid']:<10} {r['pred_rows']:>6} {r['gold_rows']:>6} "
              f"{r['matched']:>6} {100 * r['row_recall']:>7.1f}% "
              f"{100 * r['row_precision']:>8.1f}% {100 * r['value_recall']:>7.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<10} {tot['pred']:>6} {tot['gold']:>6} {tot['matched']:>6} "
          f"{pct(tot['matched'], tot['gold'])} {pct(tot['matched'], tot['pred']):>9} "
          f"{pct(tot['val_hit'], tot['val_gold']):>8}")

    print(f"\nvalues   recall {pct(tot['val_hit'], tot['val_gold']).strip()}   "
          f"precision {pct(tot['val_hit'], tot['val_pred']).strip()}")
    print(f"rows     recall {pct(tot['matched'], tot['gold']).strip()}   "
          f"precision {pct(tot['matched'], tot['pred']).strip()}")

    if tot["matched"]:
        print(f"\nper-column accuracy over {tot['matched']} matched rows:")
        for c in sorted(WEIGHTS, key=lambda c: -col_hits[c]):
            print(f"  {c:<28} {pct(col_hits[c], tot['matched'])}")

    for col in want_conf:
        top = confusions[col].most_common(12)
        if not top:
            continue
        print(f"\ntop disagreements in '{col}'  (gold -> predicted):")
        for (g, p), n in top:
            print(f"  {n:4d}  {g[:40]!r:<44} -> {p[:46]!r}")

    for pmid, why in skipped:
        print(f"\n[skip] {pmid}: {why}")

    if args.json:
        json.dump({"label": args.label, "totals": dict(tot),
                   "per_paper": per_paper,
                   "column_accuracy": {c: col_hits[c] / tot["matched"]
                                       for c in WEIGHTS} if tot["matched"] else {},
                   "skipped": skipped},
                  open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
