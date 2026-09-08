"""Shared helpers for the simple-prompt pk-individual experiment.

Parsing model output and normalising cell values live here so the runner and the
scorer agree byte-for-byte on what a row is.
"""

from __future__ import annotations

import csv
import io
import itertools
import re

# The canonical pk-individual schema, exactly as the manual gold writes it.
COLS = [
    "Patient ID",
    "Drug name",
    "Analyte",
    "Specimen",
    "Population",
    "Pregnancy stage",
    "Pediatric/Gestational age",
    "Parameter type",
    "Parameter unit",
    "Parameter value",
    "Time value",
    "Time unit",
]

# Header spellings we accept, keyed by the header squashed to bare alphanumerics.
ALIASES = {
    # "pmid" is deliberately absent: the schema has no PMID column, and leaving
    # the header unmapped makes parse_csv skip that cell in place rather than
    # shift every column after it.
    "population": "Population",
    "pregnancystage": "Pregnancy stage",
    "gestationalage": "Pediatric/Gestational age",
    "pediatricgestationalage": "Pediatric/Gestational age",
    "pediatricage": "Pediatric/Gestational age",
    "specimen": "Specimen",
    "drugname": "Drug name",
    "drug": "Drug name",
    "patientid": "Patient ID",
    "subjectid": "Patient ID",
    "parametertype": "Parameter type",
    "parameter": "Parameter type",
    "analyte": "Analyte",
    "timevalue": "Time value",
    "time": "Time value",
    "timeunit": "Time unit",
    "parametervalue": "Parameter value",
    "value": "Parameter value",
    "parameterunit": "Parameter unit",
    "unit": "Parameter unit",
}

# parameter1 / value2 / timeunit3 ... the wide layout the v1 prompt produces.
WIDE_RE = re.compile(r"^(parameter|drugname|time|timeunit|value|unit)(\d+)$")

WIDE_TO_CANON = {
    "parameter": "Parameter type",
    "drugname": "Analyte",
    "time": "Time value",
    "timeunit": "Time unit",
    "value": "Parameter value",
    "unit": "Parameter unit",
}

# Cells that mean "the table does not report this". The gold uses an empty cell.
NULLS = {
    "", "na", "n/a", "n.a.", "none", "null", "nan", "-", "--", "notreported",
    "notavailable", "notapplicable", "notstated", "unknown", "nr", "ns",
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def squash(header: str) -> str:
    """'Time  unit_1' -> 'timeunit1'. Used to look headers up in ALIASES."""
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def clean(cell: str) -> str:
    """Normalise one cell: strip, and map every spelling of 'missing' to ''."""
    s = (cell or "").strip().strip('"').strip()
    if not s:
        return ""
    # squash() drops every non-alphanumeric, so a bare "%" would squash to ""
    # and be read as missing. It is a real unit: only consult the squashed form
    # when something survives the squashing.
    k = squash(s)
    if s.lower() in NULLS or (k and k in NULLS):
        return ""
    return s


def number(cell: str):
    """First number in a cell, or None. '1.2 (0.8-1.9)' -> 1.2."""
    if not cell:
        return None
    m = _NUM_RE.search(cell.replace(",", ""))
    return float(m.group()) if m else None


def strip_reasoning(text: str) -> str:
    """Drop qwen3-style <think> blocks, which otherwise swamp the CSV."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    # An unterminated block means the model never stopped reasoning.
    text = re.sub(r"<think>.*\Z", "", text, flags=re.S | re.I)
    return text


def strip_fences(text: str) -> str:
    """Unwrap ```csv ... ``` without discarding output that has no fence."""
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    return "\n".join(blocks) if blocks else text


def _map_headers(row):
    """Map a raw header row onto canonical names. Returns None if it isn't one."""
    mapped, wide = [], False
    for h in row:
        key = squash(h)
        if key in ALIASES:
            mapped.append((ALIASES[key], None))
            continue
        m = WIDE_RE.match(key)
        if m:
            wide = True
            mapped.append((WIDE_TO_CANON[m.group(1)], int(m.group(2))))
            continue
        mapped.append((None, None))
    known = sum(1 for name, _ in mapped if name)
    # Require a real majority of recognised headers so a data row isn't mistaken
    # for the header.
    if known < 5 or known < len(row) / 2:
        return None
    return mapped, wide


_TEXT_COLS = ("Specimen", "Parameter type")

# The two closed vocabularies the prompt actually specifies. Used only to tell
# competing realignments apart - `maternal/pediatric` is a Population, never a
# Pregnancy stage - never to validate or rewrite a cell.
_VOCAB = {
    "Population": {"maternal", "pediatric", "maternal/pediatric"},
    "Pregnancy stage": {"delivery", "lactation", "pregnancy", "postpartum",
                        "1st trimester", "2nd trimester", "3rd trimester"},
    "Pediatric/Gestational age": {"maternal", "fetus", "infant", "pediatric",
                                  "maternal/pediatric"},
}

_VALUE_RE = re.compile(r"^[<>~\u2248\u2264\u2265]?\s*-?\d+(?:\.\d+)?")


def _fits(name: str, cell: str) -> int:
    """How plausible `cell` is under canonical column `name`.

    Only columns with a recognisable shape vote. Free-text columns score 0, so
    they never decide an alignment on their own.
    """
    c = (cell or "").strip()
    if name == "Parameter value":
        return 3 if _VALUE_RE.match(c) else -3
    if name == "Parameter unit":
        if not c:
            return 0
        return -2 if _VALUE_RE.match(c) else 2
    if name == "Time value":
        return 1 if (not c or _VALUE_RE.match(c)) else -2
    if name == "Patient ID":
        return -2 if len(c.split()) > 2 else 1  # ids are short: "1", "Case 5"
    if name in _VOCAB:
        if not c:
            return 0
        return 2 if c.lower() in _VOCAB[name] else -1
    if name in ("Drug name", "Analyte"):
        if not c:
            return -1  # the prompt requires both; blank means the row slipped
        return -2 if _VALUE_RE.match(c) else 0
    if name in _TEXT_COLS:
        return -2 if _VALUE_RE.match(c) else 0  # a specimen is not a number
    return 0


# Beyond this many missing fields the row is too broken to reconstruct: the
# number of arrangements explodes and the winner stops meaning anything.
_MAX_GAP = 3


def _realign(row, header):
    """Restore the empty fields a model dropped from `a,,,b`-style output.

    Two shapes turn up. A model that stops after the last value it has leaves
    the trailing fields off; a model writing `a,,,b` miscounts a run of empties
    and drops an interior one. Right-padding fixes the first and corrupts the
    second - it shifts every column past the gap, so one missing comma becomes
    a whole row of confidently wrong data.

    So treat both the same way: put the missing empties at every combination of
    positions and keep the arrangement whose cells fit their columns. Padding on
    the right is simply the candidate that puts them all at the end, and it wins
    when it deserves to. Return None when nothing wins outright, so the caller
    drops the row rather than inventing one.
    """
    missing = len(header) - len(row)
    if missing < 1 or missing > _MAX_GAP:
        return None
    if any(idx is not None for _, idx in header):
        return None  # wide layout: positions are not the canonical schema
    names = [n for n, _ in header]
    seen = {}
    for combo in itertools.combinations_with_replacement(range(len(row) + 1), missing):
        cand = list(row)
        for pos in reversed(combo):  # descending, so earlier indices stay valid
            cand.insert(pos, "")
        cand = tuple(cand)
        if cand not in seen:  # inserting into a run of empties repeats a row
            seen[cand] = sum(_fits(n, c) for n, c in zip(names, cand))
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return list(ranked[0][0])


def parse_csv(text: str, pmid: str = "", dropped=None):
    """Model output -> list of dicts on the canonical schema.

    Tolerates preamble prose, code fences, reasoning blocks, repeated headers,
    and the wide `parameter1/value1/...` layout (which is melted to long).

    A row whose field count does not match the header is realigned if it is
    short by exactly one field, and otherwise discarded - appended to `dropped`
    when the caller passes a list. It is never padded or truncated into shape:
    that silently shifts the columns.
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
        got = _map_headers(row)
        if got:
            header, wide = got
            start = i + 1
            break
    if header is None:
        # The model was told "output only the CSV" and took that to mean no
        # header row either. Fall back to positional mapping when the rows are
        # exactly the schema width - the data is fine, only the labels are gone.
        return _parse_headerless(lines, pmid, dropped=dropped)

    body_rows = [r for r in csv.reader(io.StringIO("\n".join(lines[start:])))
                 if r and any(c.strip() for c in r)]

    # A header naming fewer columns than the rows actually carry means the model
    # dropped a name from the header line, not that the rows carry extra data.
    # Believing the header shifts every column after the gap, so when the body is
    # mostly full-width, ignore the header and map by position instead.
    if len(header) != len(COLS):
        full = sum(1 for r in body_rows if len(r) == len(COLS))
        if full >= max(2, len(body_rows) // 2):
            header, wide = [(c, None) for c in COLS], False

    out = []
    for row in body_rows:
        if _map_headers(row):  # a repeated header block
            continue
        while len(row) > len(header) and not row[-1].strip():
            row.pop()  # a trailing comma costs nothing
        if len(row) != len(header):
            fixed = _realign(row, header)
            if fixed is None:
                if dropped is not None:
                    dropped.append(row)
                continue
            row = fixed

        base = {c: "" for c in COLS}
        groups = {}
        for (name, idx), cell in zip(header, row):
            if not name:
                continue
            if idx is None:
                base[name] = clean(cell)
            else:
                groups.setdefault(idx, {})[name] = clean(cell)

        if not wide or not groups:
            rec = dict(base)
            if any(rec.values()):
                out.append(rec)
            continue

        # Melt: one output row per parameter block that carries anything.
        for idx in sorted(groups):
            grp = groups[idx]
            if not any(grp.values()):
                continue
            rec = dict(base)
            rec.update(grp)
            out.append(rec)
    return out


def _parse_headerless(lines, pmid: str = "", dropped=None):
    """Recover bare data rows emitted without a header, by column position.

    Two passes. The first keeps only exact-width lines, which is what proves
    this really is a CSV block and not prose that happens to contain commas.
    Only if that succeeds does the second pass admit short lines through
    _realign - otherwise a comma-heavy sentence could be bent into a row.
    """
    parsed = []
    for line in lines:
        if "," not in line:
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        # Models trained on the old schema still lead with the PMID. Drop it
        # rather than reject the row - but only when it really is this paper's.
        if pmid and len(row) == len(COLS) + 1 and row[0].strip() == str(pmid):
            row = row[1:]
        parsed.append(row)

    exact = sum(1 for r in parsed if len(r) == len(COLS))
    # One stray comma-bearing line is not a table; require a real block.
    if exact < 2:
        return []

    header = [(c, None) for c in COLS]
    rows = []
    for row in parsed:
        if len(row) != len(COLS):
            row = _realign(row, header)
            if row is None:
                if dropped is not None:
                    dropped.append(row)
                continue
        rec = {c: clean(cell) for c, cell in zip(COLS, row)}
        if any(rec.values()):
            rows.append(rec)
    return rows


def write_csv(path, rows, cols=None):
    """Write `rows` on the canonical schema.

    `cols` overrides the field list. The per-paper `combined.csv` uses COLS, so
    it is column-identical to the manual gold; the pooled `all_predictions.csv`
    passes COLS + ["PMID"] because otherwise nothing says which paper a row
    came from.
    """
    cols = list(cols or COLS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def read_csv(path, pmid: str = ""):
    """Read a CSV already on (or close to) the canonical schema."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = fh.read()
    return parse_csv(raw, pmid=pmid)
