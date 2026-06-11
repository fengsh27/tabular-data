#!/usr/bin/env python3
"""Deterministic row cleanup for the PK specimen skills (summary + individual).

Ports the legacy `RowCleanupStep.execute_directly` of both
`pk_specimen_summary` and `pk_specimen_individual` (identical logic) to a single
stdlib-only script with a regression test. Both specimen skills call it on their
assembled 9-column CSV to produce the final curated CSV.

Two rules, in order, mirroring the legacy pandas code exactly:

  1. remove_sum_row — if every `Sample N` is an integer and exactly one row's
     value equals half the column total (i.e. total - v == v, v != 0), drop the
     first such row. This removes a redundant "summed total" row when the
     individual parts are also present.
  2. filter_max_sample_n — if every `Sample N` is an integer, collapse rows that
     are identical across all columns EXCEPT `Sample N`, `Population N`, and the
     traceability column (`Note` here; `Source text` in the legacy pre-rename
     table), keeping the row with the largest `Sample N` (ties -> earliest).

If any `Sample N` is non-integer (e.g. "N/A"), BOTH rules no-op — exactly like the
legacy code, whose `astype(int)` raises and returns the frame unchanged. Original
row order is always preserved among survivors. Columns are never reordered.

Usage:
    python clean_specimen_rows.py 04_assembled.csv            # -> stdout
    python clean_specimen_rows.py 04_assembled.csv -o 05_final.csv
"""

import argparse
import csv
import io
import sys

# Columns excluded from the "identical except sample count" comparison. Mirrors
# the legacy exclude lists: summary excluded {Sample N, Population N, Source text},
# individual excluded {Sample N, Source text}. The traceability column is `Note`
# in the skills' final schema. Population N is simply absent in the individual
# schema, so a single unified exclude set reproduces both variants.
COMPARE_EXCLUDE = {"Sample N", "Population N", "Note"}

SAMPLE_N = "Sample N"


def _all_int(values):
    """True iff every value parses as a Python int (matches pandas astype(int)
    on a string column: "20" ok, "20.5"/"N/A"/"" raise)."""
    try:
        for v in values:
            int((v if v is not None else "").strip())
        return True
    except (ValueError, TypeError):
        return False


def _sn(row):
    return int((row.get(SAMPLE_N) or "").strip())


def remove_sum_row(rows):
    if SAMPLE_N not in (rows[0] if rows else {}):
        return rows
    if not rows or not _all_int(r.get(SAMPLE_N) for r in rows):
        return rows
    total = sum(_sn(r) for r in rows)
    for i, r in enumerate(rows):
        v = _sn(r)
        if total - v == v and v != 0:
            return rows[:i] + rows[i + 1:]
    return rows


def filter_max_sample_n(rows, fieldnames):
    if SAMPLE_N not in (rows[0] if rows else {}):
        return rows
    if not rows or not _all_int(r.get(SAMPLE_N) for r in rows):
        return rows
    compare_cols = [c for c in fieldnames if c not in COMPARE_EXCLUDE]
    best = {}  # compare-key -> (original_index, sample_n)
    for i, r in enumerate(rows):
        key = tuple(r.get(c, "") for c in compare_cols)
        sn = _sn(r)
        if key not in best or sn > best[key][1]:
            best[key] = (i, sn)
    keep_idx = {idx for idx, _ in best.values()}
    return [r for i, r in enumerate(rows) if i in keep_idx]


def clean_rows(rows, fieldnames):
    rows = remove_sum_row(rows)
    rows = filter_max_sample_n(rows, fieldnames)
    return rows


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_csv", help="the assembled 9-column specimen CSV")
    ap.add_argument("-o", "--output", default=None, help="output CSV (default: stdout)")
    args = ap.parse_args(argv[1:])

    try:
        with open(args.input_csv, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except OSError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    cleaned = clean_rows(rows, fieldnames)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in cleaned:
        writer.writerow({c: r.get(c, "") for c in fieldnames})
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
