#!/usr/bin/env python3
"""Judge curated rows against the SOURCE TABLE. Never against the gold.

This is the gate in the two-tier design: if it can tell a good extraction from a
bad one without the answer key, the fast path can be trusted on papers that have
no gold. Its output is deliberately kept comparable to the gold score so the two
can be correlated afterwards - that correlation is the real experiment.

Two independent signals:

  deterministic - grounding, coverage, schema and enum conformance, duplicates,
                  citation-shaped IDs. Pure Python, no model, runs on every row.
  llm           - a second pass asking the model whether the rows faithfully
                  represent the table. Optional (--llm), sampled, best-effort.

    python scripts/simple_prompt/evaluate_pk_individual.py \\
        --pred results/simple_v2/<ts> --scratch .../.pk_individual_scratch \\
        --out scores/eval_simple_v2.json --llm --base-url http://127.0.0.1:PORT
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

TABLE_FILES = ("00_markdown_table.md", "inputs.md")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# What the gold actually contains, so "conforms" means "looks like real gold".
ENUMS = {
    "Population": {"", "maternal", "pediatric", "maternal/pediatric"},
    "Pediatric/Gestational age": {"", "maternal", "infant", "fetus", "pediatric",
                                  "maternal/pediatric"},
}
CITATION_RE = re.compile(r"et al\.?|\[\d+\]|\b(19|20)\d\d\b", re.I)


def table_text(scratch: str, pmid: str) -> str:
    root = os.path.join(scratch, pmid)
    if not os.path.isdir(root):
        return ""
    chunks = []
    for name in sorted(os.listdir(root)):
        tdir = os.path.join(root, name)
        if not os.path.isdir(tdir) or not name.startswith("table"):
            continue
        for cand in TABLE_FILES:
            path = os.path.join(tdir, cand)
            if os.path.exists(path):
                chunks.append(open(path, encoding="utf-8", errors="replace").read())
                break
    return "\n\n".join(chunks)


# Some journals set the decimal point as a middle dot ("2\u00b780"). Left alone,
# the regex reads that as two integers and every such value looks ungrounded.
DECIMALS = str.maketrans({"\u00b7": ".", "\u2219": ".", "\u22c5": ".", "\u2027": "."})


def numbers_in(text: str) -> set:
    return {round(float(m), 6)
            for m in NUM_RE.findall((text or "").translate(DECIMALS))}


def deterministic(rows, source: str) -> dict:
    """Checks that need no model. Every one of these is a real defect if it fires."""
    src_nums = numbers_in(source)
    src_lower = (source or "").translate(DECIMALS).lower()

    grounded = ungrounded = 0
    unmatched_examples = []
    for r in rows:
        n = common.number(r.get("Parameter value", ""))
        if n is None:
            continue
        if round(n, 6) in src_nums:
            grounded += 1
        else:
            ungrounded += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append(r.get("Parameter value", ""))

    emitted = {round(common.number(r["Parameter value"]), 6)
               for r in rows if common.number(r.get("Parameter value", "")) is not None}

    enum_bad = collections.Counter()
    for col, allowed in ENUMS.items():
        for r in rows:
            if (r.get(col, "") or "").strip().lower() not in allowed:
                enum_bad[col] += 1

    citations = [r["Patient ID"] for r in rows
                 if CITATION_RE.search(r.get("Patient ID", "") or "")]

    seen, dups = set(), 0
    for r in rows:
        key = tuple(r.get(c, "") for c in common.COLS)
        if key in seen:
            dups += 1
        seen.add(key)

    # A drug name the source never mentions is a strong hallucination signal.
    drugs = {(r.get("Drug name") or "").strip().lower() for r in rows}
    drugs.discard("")
    unknown_drugs = sorted(d for d in drugs if d and d not in src_lower)

    total = len(rows)
    checked = grounded + ungrounded
    return {
        "rows": total,
        "grounding_rate": grounded / checked if checked else 0.0,
        "ungrounded_values": ungrounded,
        "ungrounded_examples": unmatched_examples,
        "source_numbers": len(src_nums),
        "coverage_rate": len(emitted & src_nums) / len(src_nums) if src_nums else 0.0,
        "blank_value_rows": sum(1 for r in rows if not r.get("Parameter value")),
        "blank_specimen_rate": sum(1 for r in rows if not r.get("Specimen")) / total
                               if total else 0.0,
        "enum_violations": dict(enum_bad),
        "enum_violation_rate": sum(enum_bad.values()) / (total * len(ENUMS))
                               if total else 0.0,
        "citation_patient_ids": len(citations),
        "citation_examples": citations[:3],
        "duplicate_rows": dups,
        "unknown_drug_names": unknown_drugs[:5],
        "schema_ok": total > 0,
    }


LLM_PROMPT = """You are checking whether extracted data faithfully represents a source table.

SOURCE TABLE:
{table}

EXTRACTED ROWS (CSV):
{rows}

Judge ONLY whether the extracted rows are faithful to the table. Do not judge
style, column naming, or completeness of the schema. Specifically check:
- every Parameter value appears in the table
- each value is attributed to the right patient, drug, and specimen
- no row invents data the table does not contain

Reply with STRICT JSON and nothing else:
{{"faithful_rows": <int>, "checked_rows": <int>, "score": <0-100>, "problems": ["..."]}}
"""


def llm_check(rows, source, base_url, model, sample, timeout):
    """Ask the model to verify its own extraction against the table."""
    import urllib.request

    subset = rows[:sample]
    if not subset:
        return {"available": False, "reason": "no rows"}
    csv_text = ",".join(common.COLS) + "\n" + "\n".join(
        ",".join((r.get(c, "") or "").replace(",", " ") for c in common.COLS)
        for r in subset
    )
    prompt = LLM_PROMPT.format(table=source[:24000], rows=csv_text)
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": 65536},
    }).encode()
    try:
        req = urllib.request.Request(f"{base_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode()).get("response", "")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}

    text = common.strip_reasoning(raw)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"available": False, "reason": "no JSON in reply", "raw": text[:300]}
    try:
        got = json.loads(m.group())
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"bad JSON: {exc}", "raw": text[:300]}
    got["available"] = True
    got["sampled_rows"] = len(subset)
    return got


def confidence(det: dict, llm: dict | None) -> float:
    """One number in [0,1]. Weighted so hallucination dominates.

    Coverage is deliberately weak: a table holds doses, ages and IDs that are not
    meant to be curated, so low coverage is not by itself a defect.
    """
    if not det["schema_ok"]:
        return 0.0
    parts = [
        (0.50, det["grounding_rate"]),
        (0.15, 1.0 - min(1.0, det["enum_violation_rate"])),
        (0.10, 1.0 if not det["citation_patient_ids"] else 0.0),
        (0.10, 1.0 if not det["unknown_drug_names"] else 0.0),
        (0.05, 1.0 - min(1.0, det["duplicate_rows"] / max(1, det["rows"]))),
        (0.10, min(1.0, det["coverage_rate"] * 2)),
    ]
    score = sum(w * v for w, v in parts)
    if llm and llm.get("available") and isinstance(llm.get("score"), (int, float)):
        score = 0.7 * score + 0.3 * (llm["score"] / 100.0)
    return round(score, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="eval")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.8:27b-t0")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    pmids = sorted(d for d in os.listdir(args.pred)
                   if os.path.isdir(os.path.join(args.pred, d)) and d.isdigit())

    report = {"label": args.label, "llm_enabled": args.llm, "papers": {}}
    print(f"\n=== self-evaluation ({args.label}) — source-grounded, gold not used ===\n")
    print(f"{'PMID':<10} {'rows':>5} {'ground':>7} {'cover':>6} {'enum':>6} "
          f"{'llm':>5} {'conf':>6}")
    print("-" * 52)

    for pmid in pmids:
        path = os.path.join(args.pred, pmid, "combined.csv")
        rows = common.read_csv(path, pmid=pmid) if os.path.exists(path) else []
        source = table_text(args.scratch, pmid)
        det = deterministic(rows, source)
        llm = (llm_check(rows, source, args.base_url.rstrip("/"), args.model,
                         args.sample, args.timeout) if args.llm else None)
        conf = confidence(det, llm)
        report["papers"][pmid] = {"deterministic": det, "llm": llm,
                                  "confidence": conf}
        llm_s = (f"{llm['score']:.0f}" if llm and llm.get("available")
                 and isinstance(llm.get("score"), (int, float)) else "-")
        print(f"{pmid:<10} {det['rows']:>5} {100*det['grounding_rate']:>6.1f}% "
              f"{100*det['coverage_rate']:>5.1f}% {100*det['enum_violation_rate']:>5.1f}% "
              f"{llm_s:>5} {conf:>6.3f}")

    confs = [p["confidence"] for p in report["papers"].values()]
    report["mean_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
    print("-" * 52)
    print(f"{'MEAN':<10} {'':>5} {'':>7} {'':>6} {'':>6} {'':>5} "
          f"{report['mean_confidence']:>6.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
