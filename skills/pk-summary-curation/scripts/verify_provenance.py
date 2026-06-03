#!/usr/bin/env python3
"""Deterministic provenance check for a curated table.

Verifies that every NUMBER appearing in a curated CSV also appears in the
source material (the source markdown table + caption/footnotes). This is the
model-independent half of the verify/correct step: it catches hallucinated or
mistyped numeric values cold, without asking any model to judge itself.

It deliberately checks **numbers only**, not text. Categorical columns
(Population, Statistics type, Parameter type, …) are normalized/derived by the
curation stages and legitimately do NOT appear verbatim in the source, so
matching their text would produce false positives. Numbers, by contrast, are
required to be copied verbatim from the source ("no calculations" rule), so any
number that is absent from the source is a real provenance failure.

The check is schema-agnostic: it does not know or care about column names. It
extracts numbers from whatever cells exist. Use --skip-columns / --value-columns
to scope which columns are checked when a normalized label embeds a number
(e.g. "Trimester 1", "Method A1").

Usage:
    python verify_provenance.py FINAL_CSV SOURCE [SOURCE ...]
    python verify_provenance.py 13_final.csv 00_markdown_table.md inputs.md
    python verify_provenance.py 13_final.csv source.md --skip-columns "Pregnancy stage,Parameter type"
    python verify_provenance.py 13_final.csv source.md --json

Exit code: 0 if every checked number is supported by the source, 1 otherwise
(2 on usage/IO error).
"""

import argparse
import csv
import json
import re
import sys

# Numbers WITHOUT a leading sign, so that an ASCII-hyphen range like "0-12"
# yields {0, 12} rather than {0, -12}, and en-dash ranges "5.39-8.17" split too.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|\.\d+")

# Cells that never carry a checkable value.
NA_TOKENS = {"", "n/a", "na", "nd", "none", "-", "–", "—"}


def extract_numbers(text):
    """Return the set of distinct numeric values (as floats) found in `text`."""
    out = set()
    for tok in NUMBER_RE.findall(text or ""):
        try:
            out.add(round(float(tok), 6))
        except ValueError:
            continue
    return out


def load_source_numbers(source_paths):
    """Union of all numbers present across the source files."""
    nums = set()
    raw = []
    for p in source_paths:
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        raw.append(text)
        nums |= extract_numbers(text)
    return nums, "\n".join(raw)


def check_csv(csv_path, source_numbers, value_columns=None, skip_columns=None):
    """Return a list of unsupported-number findings.

    Each finding: {row, column, cell, number}.
    """
    skip_columns = set(skip_columns or [])
    value_columns = set(value_columns) if value_columns else None

    findings = []
    checked_count = 0
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            for col, cell in row.items():
                if col is None:
                    continue
                col_name = col.strip()
                # csv may produce an unnamed index column (key ""): keep it skippable
                if value_columns is not None and col_name not in value_columns:
                    continue
                if col_name in skip_columns:
                    continue
                if (cell or "").strip().lower() in NA_TOKENS:
                    continue
                for num in extract_numbers(cell):
                    checked_count += 1
                    if num not in source_numbers:
                        findings.append(
                            {
                                "row": row_idx,
                                "column": col_name,
                                "cell": cell.strip(),
                                "number": num,
                            }
                        )
    return findings, checked_count


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("final_csv", help="the curated CSV to verify")
    ap.add_argument("source", nargs="+",
                    help="source file(s): the source markdown table and caption/inputs")
    ap.add_argument("--value-columns", default=None,
                    help="comma-separated columns to check (default: all)")
    ap.add_argument("--skip-columns", default=None,
                    help="comma-separated columns to exclude from checking")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args(argv[1:])

    def split(s):
        return [c.strip() for c in s.split(",")] if s else None

    try:
        source_numbers, _ = load_source_numbers(args.source)
        findings, checked = check_csv(
            args.final_csv,
            source_numbers,
            value_columns=split(args.value_columns),
            skip_columns=split(args.skip_columns),
        )
    except (OSError, csv.Error) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    supported = checked - len(findings)
    if args.json:
        print(json.dumps({
            "checked_numbers": checked,
            "supported": supported,
            "unsupported": len(findings),
            "findings": findings,
        }, indent=2))
    else:
        print(f"Provenance check: {supported}/{checked} numbers supported by source.")
        if findings:
            print(f"\n{len(findings)} UNSUPPORTED number(s) — not found in source:")
            for f in findings:
                print(f"  row {f['row']}, column '{f['column']}': "
                      f"{f['number']:g}  (cell: {f['cell']!r})")
            print("\nThese values may be hallucinated, mistyped, or mis-transcribed. "
                  "Re-check them against the source before accepting the row.")
        else:
            print("All checked numbers trace back to the source. ✓")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
