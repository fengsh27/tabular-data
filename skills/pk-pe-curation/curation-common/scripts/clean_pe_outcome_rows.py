#!/usr/bin/env python3
"""Deterministic row cleanup for the pe-study-outcome skill (ver2 pipeline).

Ports `pe_study_outcome_ver2.pe_study_out_row_cleanup_step.RowCleanupStep`: a
chain of business rules over the assembled outcome rows, then a numeric filter,
column renames, and reorder to the final 12-column schema.

INPUT columns (the "working" names produced by the assembly stage):
  Characteristic, Exposure, Outcome, Main value, Main value unit, Statistics type,
  Variation type, Variation value, Interval type, Lower bound, Upper bound, P value
OUTPUT columns (final schema):
  Characteristic, Exposure, Outcome, Parameter unit, Parameter statistic,
  Parameter value, Variation type, Variation value, Interval type, Lower bound,
  Upper bound, P value

The rules run in the SAME order as the legacy pandas code (which matters — the
conditional rules act on raw values *before* the empty/sentinel normalization):

  1. if (Statistics type == Interval type OR == "N/A") AND
        (Main value == Lower bound OR == Upper bound): Main value, Statistics type := "N/A"
  2. if Lower/Upper != "N/A" and both appear as substrings of Main value: Main value := "N/A"
  3. if Main value == "N/A": Statistics type := "N/A"
  4. if Lower == Upper == "N/A": Interval type := "N/A"
  5. if Lower != "N/A" and Upper != "N/A": Interval type := "Range"
  6. if Variation value == "N/A": Variation type := "N/A"
  7. normalize cells: blank/whitespace -> "N/A"; n/a, unknown, Unknown, nan -> "N/A";
     Standard Deviation (SD)/s.d./S.D. -> "SD"; a lone "," -> " "
  8. drop rows where NONE of [Main value, Variation type, Lower bound, Upper bound]
     contains a digit
  9. rename Main value->Parameter value, Statistics type->Parameter statistic,
     Main value unit->Parameter unit; reorder to the final schema

Usage:
    python clean_pe_outcome_rows.py 04_assembled.csv            # -> stdout
    python clean_pe_outcome_rows.py 04_assembled.csv -o 05_final.csv
"""

import argparse
import csv
import io
import re
import sys

NA = "N/A"

WORKING_COLUMNS = [
    "Characteristic", "Exposure", "Outcome", "Main value", "Main value unit",
    "Statistics type", "Variation type", "Variation value", "Interval type",
    "Lower bound", "Upper bound", "P value",
]

FINAL_COLUMNS = [
    "Characteristic", "Exposure", "Outcome", "Parameter unit",
    "Parameter statistic", "Parameter value", "Variation type",
    "Variation value", "Interval type", "Lower bound", "Upper bound", "P value",
]

RENAME = {
    "Main value": "Parameter value",
    "Statistics type": "Parameter statistic",
    "Main value unit": "Parameter unit",
}

_CELL_MAP = {
    "n/a": NA, "unknown": NA, "Unknown": NA, "nan": NA,
    "Standard Deviation (SD)": "SD", "s.d.": "SD", "S.D.": "SD",
    ",": " ",
}
_WS = re.compile(r"^\s*$")


def _g(r, col):
    return r.get(col, "") if r.get(col) is not None else ""


def _normalize_cell(v):
    s = v if v is not None else ""
    if _WS.match(s):
        return NA
    return _CELL_MAP.get(s, s)


def _has_digit(s):
    return any(ch.isdigit() for ch in (s or ""))


def clean_rows(rows):
    out = []
    for r in rows:
        r = dict(r)

        stat, interval = _g(r, "Statistics type"), _g(r, "Interval type")
        main = _g(r, "Main value")
        lower, upper = _g(r, "Lower bound"), _g(r, "Upper bound")

        # 1
        if (stat == interval or stat == NA) and (main == lower or main == upper):
            r["Main value"] = NA
            r["Statistics type"] = NA
        # 2
        main = _g(r, "Main value")
        if lower.strip() != NA and upper.strip() != NA and lower in main and upper in main:
            r["Main value"] = NA
        # 3
        if _g(r, "Main value") == NA:
            r["Statistics type"] = NA
        # 4 / 5
        if lower == NA and upper == NA:
            r["Interval type"] = NA
        if lower != NA and upper != NA:
            r["Interval type"] = "Range"
        # 6
        if _g(r, "Variation value") == NA:
            r["Variation type"] = NA

        # 7 normalize every cell
        r = {k: _normalize_cell(v) for k, v in r.items()}

        # 8 numeric filter
        if not any(_has_digit(_g(r, c)) for c in
                   ["Main value", "Variation type", "Lower bound", "Upper bound"]):
            continue

        # 9 rename to final names
        out.append({RENAME.get(k, k): v for k, v in r.items()})
    return out


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_csv", help="the assembled 12-column (working-name) PE outcome CSV")
    ap.add_argument("-o", "--output", default=None, help="output CSV (default: stdout)")
    args = ap.parse_args(argv[1:])

    try:
        with open(args.input_csv, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    cleaned = clean_rows(rows)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FINAL_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in cleaned:
        writer.writerow({c: r.get(c, "") for c in FINAL_COLUMNS})
    text = buf.getvalue()

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        sys.stderr.write(f"wrote {len(cleaned)} rows to {args.output}\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
