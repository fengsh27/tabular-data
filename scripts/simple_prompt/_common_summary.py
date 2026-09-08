"""Shared helpers for the simple-prompt pk-summary experiment.

Mirrors _common.py's approach (fence/reasoning stripping, header mapping, a
scored realignment for short rows) but targets the legacy 16-column schema
that benchmark/data/pk-summary/baseline/*.csv and benchmark/configs.py use,
not the newer 19-column pk-summary-curation skill schema. Kept as a separate
module rather than parametrizing _common.py so the already-validated
pk-individual path cannot regress from a pk-summary change.
"""

from __future__ import annotations

import csv
import io
import itertools
import re

from _common import clean, squash, strip_fences, strip_reasoning  # noqa: F401

# The canonical pk-summary schema, matching benchmark/configs.py's
# PK_SUMMARY_COLUMNS_TYPE keys (the columns the existing scorer reads by
# name) rather than the newer skill's 19-column contract.
COLS = [
    "Drug name",
    "Analyte",
    "Specimen",
    "Population",
    "Pregnancy stage",
    "Subject N",
    "Parameter type",
    "Value",
    "Unit",
    "Summary statistics",
    "Variation type",
    "Variation value",
    "Interval type",
    "Lower limit",
    "High limit",
    "P value",
]

ALIASES = {
    "drugname": "Drug name",
    "drug": "Drug name",
    "analyte": "Analyte",
    "specimen": "Specimen",
    "population": "Population",
    "pregnancystage": "Pregnancy stage",
    "subjectn": "Subject N",
    "n": "Subject N",
    "parametertype": "Parameter type",
    "parameter": "Parameter type",
    "value": "Value",
    "parametervalue": "Value",
    "unit": "Unit",
    "parameterunit": "Unit",
    "summarystatistics": "Summary statistics",
    "summarystatistic": "Summary statistics",
    "statistics": "Summary statistics",
    "statisticstype": "Summary statistics",
    "variationtype": "Variation type",
    "variationvalue": "Variation value",
    "intervaltype": "Interval type",
    "lowerlimit": "Lower limit",
    "lowerbound": "Lower limit",
    "highlimit": "High limit",
    "upperbound": "High limit",
    "upperlimit": "High limit",
    "pvalue": "P value",
}

NUMERIC_COLS = {"Subject N", "Value", "Variation value", "Lower limit", "High limit", "P value"}
_TEXT_COLS = ("Drug name", "Analyte", "Specimen", "Parameter type", "Unit")

_VOCAB = {
    "Population": {"maternal", "pediatric", "child", "children", "adult", "adults",
                   "healthy adults", "infant", "infants", "neonate", "neonates"},
    "Pregnancy stage": {"delivery", "lactation", "pregnancy", "postpartum",
                        "1st trimester", "2nd trimester", "3rd trimester"},
    "Summary statistics": {"mean", "median", "geometric mean"},
    "Variation type": {"sd", "cv%", "cv", "sem", "range"},
    "Interval type": {"95% ci", "range", "iqr", "minmax", "min-max"},
}

_VALUE_RE = re.compile(r"^[<>~≈≤≥]?\s*-?\d+(?:\.\d+)?")


def _fits(name: str, cell: str) -> int:
    """How plausible `cell` is under canonical column `name`. See _common._fits."""
    c = (cell or "").strip()
    if name in NUMERIC_COLS:
        if not c:
            return 0
        return 2 if _VALUE_RE.match(c) else -3
    if name in _VOCAB:
        if not c:
            return 0
        return 2 if c.lower() in _VOCAB[name] else -1
    if name in ("Drug name", "Parameter type"):
        if not c:
            return -1  # every row needs at least these
        return -2 if _VALUE_RE.match(c) else 0
    if name in _TEXT_COLS:
        return -2 if _VALUE_RE.match(c) else 0  # a specimen/unit is not a bare number
    return 0


_MAX_GAP = 3


def _map_headers(row):
    mapped = []
    for h in row:
        key = squash(h)
        mapped.append(ALIASES.get(key))
    known = sum(1 for name in mapped if name)
    if known < 4 or known < len(row) / 2:
        return None
    return mapped


def _realign(row, header_names):
    """Restore a field count mismatch. See _common._realign.

    Two shapes turn up. A short row is missing empties the model dropped -
    insert blanks at every combination of positions and keep the arrangement
    that fits its columns best. A long row usually means the model invented
    an extra field for something the schema had no column for (e.g. a
    before/after cohort split it should have folded into an existing text
    column) - try dropping one field from every combination of positions
    instead. Both return None on a tie or an unworkable gap, so the caller
    drops the row rather than inventing one.
    """
    gap = len(row) - len(header_names)
    if gap == 0 or abs(gap) > _MAX_GAP:
        return None
    seen = {}
    if gap < 0:
        missing = -gap
        for combo in itertools.combinations_with_replacement(range(len(row) + 1), missing):
            cand = list(row)
            for pos in reversed(combo):
                cand.insert(pos, "")
            cand = tuple(cand)
            if cand not in seen:
                seen[cand] = sum(_fits(n, c) for n, c in zip(header_names, cand))
    else:
        extra = gap
        for combo in itertools.combinations(range(len(row)), extra):
            drop = set(combo)
            cand = tuple(c for i, c in enumerate(row) if i not in drop)
            if cand not in seen:
                seen[cand] = sum(_fits(n, c) for n, c in zip(header_names, cand))
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return list(ranked[0][0])


def parse_csv(text: str, pmid: str = "", dropped=None):
    """Model output -> list of dicts on the canonical pk-summary schema.

    `pmid` is accepted-but-unused so this has the same call signature as
    _common.parse_csv (the schema has no PMID column to strip).
    """
    text = strip_fences(strip_reasoning(text or ""))
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header = None
    start = 0
    for i, line in enumerate(lines):
        if "," not in line:
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        mapped = _map_headers(row)
        if mapped:
            header = mapped
            start = i + 1
            break
    if header is None:
        return _parse_headerless(lines, dropped=dropped)

    body_rows = [r for r in csv.reader(io.StringIO("\n".join(lines[start:])))
                 if r and any(c.strip() for c in r)]

    if len(header) != len(COLS):
        full = sum(1 for r in body_rows if len(r) == len(COLS))
        if full >= max(2, len(body_rows) // 2):
            header = list(COLS)

    out = []
    for row in body_rows:
        if _map_headers(row):  # a repeated header block
            continue
        while len(row) > len(header) and not row[-1].strip():
            row.pop()
        if len(row) != len(header):
            fixed = _realign(row, header)
            if fixed is None:
                if dropped is not None:
                    dropped.append(row)
                continue
            row = fixed

        rec = {c: "" for c in COLS}
        for name, cell in zip(header, row):
            if name:
                rec[name] = clean(cell)
        if any(rec.values()):
            out.append(rec)
    return out


def _parse_headerless(lines, dropped=None):
    parsed = []
    for line in lines:
        if "," not in line:
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        parsed.append(row)

    exact = sum(1 for r in parsed if len(r) == len(COLS))
    if exact < 2:
        return []

    rows = []
    for row in parsed:
        if len(row) != len(COLS):
            row = _realign(row, COLS)
            if row is None:
                if dropped is not None:
                    dropped.append(row)
                continue
        rec = {c: clean(cell) for c, cell in zip(COLS, row)}
        if any(rec.values()):
            rows.append(rec)
    return rows


def write_csv(path, rows, cols=None):
    cols = list(cols or COLS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
