#!/usr/bin/env python3
"""Deterministic row cleanup for the pk-population-individual skill.

Ports the one active rule of the legacy
`pk_popu_ind_row_cleanup_step.RowCleanupStep.execute_directly`: drop any row
whose characteristic value is blank or N/A. The legacy code operates on the
pre-rename column `Main value`; in the skill's final schema that column is
`Characteristic value`, so this script keys on `Characteristic value`.

The legacy emptiness test, reproduced exactly:
    mv = value.strip()                       # trim
    mv = re.sub(r'\\s*/\\s*', '/', mv)        # normalize spaced slashes ("4 / 5" -> "4/5")
    mv = mv.upper()
    keep row iff len(mv) > 0 and mv not in {'N/A', 'NA'}
The normalization only affects the keep/drop decision; the original cell value is
written through unchanged. Original row order is preserved.

(The pk-population-summary cleanup is a no-op in the legacy code, so it has no
script — its assembled table is already final.)

Usage:
    python clean_population_individual_rows.py 04_assembled.csv            # -> stdout
    python clean_population_individual_rows.py 04_assembled.csv -o 05_final.csv
"""

import argparse
import csv
import io
import re
import sys

VALUE_COLUMN = "Characteristic value"
DROP_IF_IN = {"N/A", "NA"}
_SLASH = re.compile(r"\s*/\s*")


def _is_empty_value(cell):
    mv = (cell if cell is not None else "").strip()
    mv = _SLASH.sub("/", mv).upper()
    return len(mv) == 0 or mv in DROP_IF_IN


def clean_rows(rows):
    if rows and VALUE_COLUMN not in rows[0]:
        return rows  # column absent -> nothing to filter (tolerant)
    return [r for r in rows if not _is_empty_value(r.get(VALUE_COLUMN))]


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_csv", help="the assembled 9-column population-individual CSV")
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

    cleaned = clean_rows(rows)

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
