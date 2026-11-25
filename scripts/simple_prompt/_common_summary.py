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
    # candidate -> (fit total, edits that touch an existing run of empties).
    # The second key breaks fit ties: the slip this repairs is a miscounted run
    # of commas, so an empty inserted next to an existing empty (or an empty
    # field dropped) is the likely truth, while an empty inserted between two
    # non-empty cells is not. Without it, a row whose two candidates differ
    # only in free-text (0-voting) columns - "Vancomycin,,Vancomycin" vs
    # "Vancomycin,Vancomycin,," - ties and is dropped (4 of 8 rows on PMID
    # 35465728, second live round).
    seen = {}
    if gap < 0:
        missing = -gap
        for combo in itertools.combinations_with_replacement(range(len(row) + 1), missing):
            cand = list(row)
            for pos in reversed(combo):
                cand.insert(pos, "")
            cand = tuple(cand)
            run_edits = sum(
                1 for pos in combo
                if (pos > 0 and not (row[pos - 1] or "").strip())
                or (pos < len(row) and not (row[pos] or "").strip())
            )
            if cand not in seen or seen[cand][1] < run_edits:
                seen[cand] = (sum(_fits(n, c) for n, c in zip(header_names, cand)), run_edits)
    else:
        extra = gap
        for combo in itertools.combinations(range(len(row)), extra):
            drop = set(combo)
            cand = tuple(c for i, c in enumerate(row) if i not in drop)
            run_edits = sum(1 for i in combo if not (row[i] or "").strip())
            if cand not in seen or seen[cand][1] < run_edits:
                seen[cand] = (sum(_fits(n, c) for n, c in zip(header_names, cand)), run_edits)
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return list(ranked[0][0])


# A cell scoring this low under its column is in the wrong column (text in a
# numeric column, a bare number as a drug or parameter name), not merely
# unusual. Only such a row is a candidate for _reshift, and only a repair that
# leaves no such cell behind is accepted.
_HARD_MISFIT = -2

_WORDS_RE = re.compile(r"[A-Za-z]{2,}")


def _fits_strict(name: str, cell: str) -> int:
    """`_fits`, except that a cell which merely *starts* with a digit and goes
    on in words - "95% CI", "25th-75th percentiles", "3 Month to < 3 Years" -
    is text, not a number. `_VALUE_RE` only anchors the front, so `_fits`
    rewards "95% CI" sitting in `Lower limit` (+2) and penalises an age group
    in `Specimen` (-2). Inside `_reshift` both are wrong: the first let a
    half-repaired row (PMID 16143486 table 2, two segments shifted, one move
    fixes one) pass as clean, the second triggered a needless move. Used only
    by `_reshift`; `_fits` itself, and `_realign`/Tier 1 with it, are unchanged.
    """
    c = (cell or "").strip()
    if _VALUE_RE.match(c) and _WORDS_RE.search(c):
        if name in NUMERIC_COLS:
            return -3
        if name in _TEXT_COLS or name in ("Drug name", "Parameter type"):
            return 0
    if name in _VOCAB and c and c.lower() not in _VOCAB[name]:
        # A label outside the whitelist ("parturients" for Population) is
        # unusual, not structural evidence. Left at _fits' -1, the search
        # spends a second move parking the label in a free-text column just to
        # dodge the penalty (seen on PMID 16143486) - so it votes 0 here.
        return 0
    return _fits(name, cell)


def _row_consistency(cells, names) -> int:
    """Row-level vote `_reshift` adds to the per-cell `_fits_strict` total.

    An interval has two ends: `Lower limit` filled with `High limit` empty (or
    the reverse) is a shifted tail, not data. Per-cell scoring cannot see it -
    "Lower='' / High=9.42 / P=16.49" and "Lower=9.42 / High=16.49 / P=''" are
    both three plausible numeric cells - and on PMID 16143486 table 2 that
    tie let a half repair win over the full one.
    """
    rec = dict(zip(names, cells))
    lo = (rec.get("Lower limit") or "").strip()
    hi = (rec.get("High limit") or "").strip()
    return -2 if bool(lo) != bool(hi) else 0


_MAX_MOVES = 2


def _moves(cells):
    """Every "take one existing empty cell out at e, put an empty back at p"
    move on `cells` - i.e. shift the segment between e and p by one column.
    Yields (candidate, cells displaced)."""
    original = tuple(cells)
    for e, cell in enumerate(cells):
        if (cell or "").strip():
            continue
        rest = list(cells[:e]) + list(cells[e + 1:])
        for p in range(len(rest) + 1):
            cand = tuple(rest[:p] + [""] + rest[p:])
            if cand != original:
                yield cand, abs(p - e)


def _reshift(row, names):
    """Repair a row with the right field count whose cells are shifted.

    `_realign` handles a row that is short or long. A third shape slipped past
    it on PMID 35465728 (pk_summary, table 2): the model dropped the empty
    `Pregnancy stage` cell and wrote an extra empty after `median`, so the
    count matched the header while `60 / Initial Cavg / 12.9 / mg/L / median`
    all sat one column left of home - a whole table of confidently wrong rows
    that scored 0 against gold, with Tier 3 reporting no error.

    Candidates are up to `_MAX_MOVES` such moves (PMID 16143486 table 2 has
    two shifted segments per row, one dropped comma in each). Each candidate
    is scored by `_fits_strict` per cell plus `_row_consistency`. Accept the
    best candidate only when it (a) beats the original row, (b) leaves no cell
    at or below `_HARD_MISFIT`, and (c) wins outright on (score, fewer moves,
    fewer cells displaced) - candidates that differ only in where an empty
    lands among free-text (0-voting) columns tie on score, and the one that
    disturbs the row least is the one the model's slip actually produced. On
    anything else return None and the caller keeps the row as written, so
    Tier 1 flags it exactly as before.
    """
    if len(row) != len(names):
        return None

    def score(cells):
        per_cell = [_fits_strict(n, c) for n, c in zip(names, cells)]
        return sum(per_cell) + _row_consistency(cells, names), min(per_cell)

    base_total, base_min = score(row)
    if base_min > _HARD_MISFIT:
        return None
    original = tuple(row)
    seen = {}  # candidate -> (total, min cell fit, moves, cells displaced)
    frontier = [(original, 0, 0)]
    for depth in range(1, _MAX_MOVES + 1):
        next_frontier = []
        for cells, _, displaced in frontier:
            for cand, moved in _moves(cells):
                key = (depth, displaced + moved)
                if cand == original or (cand in seen and seen[cand][2:] <= key):
                    continue
                total, lowest = score(cand)
                seen[cand] = (total, lowest, depth, displaced + moved)
                next_frontier.append((cand, depth, displaced + moved))
        frontier = next_frontier
    if not seen:
        return None
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1][0], kv[1][2], kv[1][3]))
    best, (best_total, best_min, best_moves, best_displaced) = ranked[0]
    if best_total <= base_total or best_min <= _HARD_MISFIT:
        return None
    if len(ranked) > 1 and ranked[1][1][0] == best_total \
            and ranked[1][1][2:] == (best_moves, best_displaced):
        return None
    return list(best)


def parse_csv(text: str, pmid: str = "", dropped=None):
    """Model output -> list of dicts on the canonical pk-summary schema.

    `pmid` is accepted-but-unused so this has the same call signature as
    _common.parse_csv (the schema has no PMID column to strip).

    A row whose field count mismatches the header goes through `_realign`; a
    row with the right count but a cell clearly in the wrong column goes
    through `_reshift`. Neither invents data - both only move empties.
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

    body_rows = [_pop_trailing_empties(r) for r in csv.reader(io.StringIO("\n".join(lines[start:])))
                 if r and any(c.strip() for c in r)]

    if len(header) != len(COLS):
        # The model dropped a name from the header but wrote full-width rows
        # (34183327: a 15-name header over 16/17-field rows): the rows are
        # positional on the canonical schema, so read them as such. A row one
        # field too long is a stray empty that _realign removes.
        full = sum(1 for r in body_rows if len(r) in (len(COLS), len(COLS) + 1))
        if full >= max(2, len(body_rows) // 2):
            header = list(COLS)

    out = []
    for row in body_rows:
        if _map_headers(row):  # a repeated header block
            continue
        row = _pop_trailing_empties(row, len(header))
        if len(row) != len(header):
            fixed = _realign(row, header)
            if fixed is None:
                if dropped is not None:
                    dropped.append(row)
                continue
            row = fixed
        else:
            fixed = _reshift(row, header)
            if fixed is not None:
                row = fixed

        rec = {c: "" for c in COLS}
        for name, cell in zip(header, row):
            if name:
                rec[name] = clean(cell)
        if any(rec.values()):
            out.append(rec)
    return out


def _pop_trailing_empties(row, width=None):
    """Drop empty fields past the schema width (a trailing comma per row is
    the commonest way a model ends up one field too wide)."""
    width = len(COLS) if width is None else width
    row = list(row)
    while len(row) > width and not row[-1].strip():
        row.pop()
    return row


def _parse_headerless(lines, dropped=None):
    """Bare data rows with no header line, read by column position.

    At least two rows must be exactly schema-width (after trailing empties are
    dropped) to count as a CSV block; otherwise nothing is parsed and every
    comma-bearing line is reported in `dropped`, so a lost table is visible to
    Tier 1 instead of silently becoming an empty one (30825333 table 3: eight
    17-field rows parsed to nothing and nothing dropped).
    """
    parsed = []
    for line in lines:
        if "," not in line:
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        parsed.append(_pop_trailing_empties(row))

    exact = sum(1 for r in parsed if len(r) == len(COLS))
    if exact < 2:
        if dropped is not None:
            dropped.extend(parsed)
        return []

    rows = []
    for row in parsed:
        if len(row) != len(COLS):
            fixed = _realign(row, COLS)
            if fixed is None:
                if dropped is not None:
                    dropped.append(row)
                continue
            row = fixed
        else:
            fixed = _reshift(row, COLS)
            if fixed is not None:
                row = fixed
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
