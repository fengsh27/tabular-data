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

The default (existence) check proves a number is PRESENT in the source; it
cannot prove the number is on the RIGHT row. The optional --attribution mode
adds that: it parses the source table(s), and for each curated row checks that
its values appear under the source label (column header or row label) that best
matches the row's --label-columns. This catches misattribution — e.g. a
cord-blood value swapped with a maternal one — which existence cannot, since
both numbers exist. Attribution only fires for rows with usable label tokens and
values that are table-sourced; everything else falls back to existence.

Usage:
    python verify_provenance.py FINAL_CSV SOURCE [SOURCE ...]
    python verify_provenance.py 13_final.csv 00_markdown_table.md inputs.md
    python verify_provenance.py 13_final.csv source.md --skip-columns "Pregnancy stage,Parameter type"
    python verify_provenance.py 13_final.csv source.md --json
    python verify_provenance.py 13_final.csv 00_markdown_table.md \
        --value-columns "Parameter value,Lower bound,Upper bound" \
        --attribution --label-columns "Parameter type,Analyte,Specimen,Population"

Exit code: 0 if every checked number is supported (and, with --attribution,
correctly attributed), 1 otherwise (2 on usage/IO error).
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

# Word tokens used to attribute a value to a labeled source row/column.
TOKEN_RE = re.compile(r"[a-z]{2,}")
# Pure measurement units are dropped so they don't create spurious label
# overlap (e.g. every concentration column shares "ng"/"ml").
UNIT_STOPWORDS = {
    "ng", "ml", "mg", "kg", "mcg", "ug", "dl", "ul", "pg", "nmol", "umol",
    "mmol", "mol", "min", "mins", "hr", "hrs", "sec",
}


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


def tokenize(text):
    """Lowercase word tokens (>=2 letters), minus pure measurement units."""
    return {t for t in TOKEN_RE.findall((text or "").lower()) if t not in UNIT_STOPWORDS}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_markdown_tables(text):
    """Parse every pipe-delimited markdown table in `text`.

    Returns a list of {"headers": [...], "rows": [[cells], ...]}. Stacked
    headers are assumed already collapsed (the bundled converter does this), so
    the header is the row immediately above the `| --- |` separator.
    """
    def is_row(line):
        return line.strip().startswith("|")

    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def is_separator(cs):
        nonempty = [c for c in cs if c != ""]
        return bool(nonempty) and all(re.fullmatch(r":?-{2,}:?", c) for c in nonempty)

    tables = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if not is_row(lines[i]):
            i += 1
            continue
        block = []
        while i < n and is_row(lines[i]):
            block.append(cells(lines[i]))
            i += 1
        sep = next((j for j, r in enumerate(block) if is_separator(r)), None)
        if sep is None or sep == 0:
            headers, rows = block[0], block[1:]
        else:
            headers, rows = block[sep - 1], block[sep + 1:]
        rows = [r for r in rows if not is_separator(r)]
        tables.append({"headers": headers, "rows": rows})
    return tables


def build_groups(tables):
    """Build labeled number-groups from source tables, in BOTH orientations.

    For a column-oriented PK table the discriminator is the column header; for
    a row-oriented one it is the row's first cell. We index both so attribution
    works regardless of orientation:
      - one group per column: (column header, numbers in that column's body)
      - one group per row:    (row's first cell, numbers in the rest of the row)
    """
    groups = []
    for t in tables:
        headers, rows = t["headers"], t["rows"]
        for ci, h in enumerate(headers):
            nums = set()
            for r in rows:
                if ci < len(r):
                    nums |= extract_numbers(r[ci])
            if nums:
                groups.append((h, nums))
        for r in rows:
            if not r:
                continue
            nums = set()
            for cell in r[1:]:
                nums |= extract_numbers(cell)
            if nums:
                groups.append((r[0], nums))
    return groups


def check_attribution(csv_path, groups, label_columns,
                      value_columns=None, skip_columns=None):
    """Attribution check: a value must appear in the source group whose label
    BEST matches the curated row's label columns — not merely somewhere.

    This catches misattribution (e.g. a cord-blood value swapped with maternal)
    that the existence check cannot: both numbers exist, but each lands under
    the wrong label. We pick the argmax-overlap group, which breaks ties toward
    the specific discriminator ("cord" vs "maternal") despite shared tokens
    ("blood").

    A finding fires only when (a) the row HAS label tokens, (b) some source
    group overlaps them, and (c) the value exists in the source but NOT in any
    best-matching group. Rows with no usable label, or values absent from every
    table group (e.g. caption-only Subject N), are left to the existence check.
    """
    value_columns = set(value_columns) if value_columns else None
    skip_columns = set(skip_columns or [])
    label_columns = set(label_columns or [])

    group_tokens = [(tokenize(lbl), nums) for lbl, nums in groups]
    all_group_numbers = set()
    for _, nums in groups:
        all_group_numbers |= nums

    findings = []
    checked = 0
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            ltok = set()
            for col, cell in row.items():
                if col is None:
                    continue
                if col.strip() in label_columns and (cell or "").strip().lower() not in NA_TOKENS:
                    ltok |= tokenize(cell)
            if not ltok:
                continue

            scored = [(jaccard(gt, ltok), nums) for gt, nums in group_tokens]
            best = max((s for s, _ in scored), default=0.0)
            if best <= 0:
                continue
            best_numbers = set()
            for s, nums in scored:
                if s == best:
                    best_numbers |= nums

            for col, cell in row.items():
                if col is None:
                    continue
                col_name = col.strip()
                if value_columns is not None and col_name not in value_columns:
                    continue
                if col_name in skip_columns or col_name in label_columns:
                    continue
                if (cell or "").strip().lower() in NA_TOKENS:
                    continue
                for num in extract_numbers(cell):
                    if num not in all_group_numbers:
                        continue  # not table-sourced; existence check owns it
                    checked += 1
                    if num not in best_numbers:
                        findings.append({
                            "row": row_idx,
                            "column": col_name,
                            "cell": cell.strip(),
                            "number": num,
                            "row_labels": sorted(ltok),
                        })
    return findings, checked


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
    ap.add_argument("--attribution", action="store_true",
                    help="also check each value appears under the source label "
                         "that best matches the row (needs --label-columns)")
    ap.add_argument("--label-columns", default=None,
                    help="comma-separated discriminator columns for --attribution "
                         "(e.g. 'Parameter type,Analyte,Specimen,Population')")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args(argv[1:])

    def split(s):
        return [c.strip() for c in s.split(",")] if s else None

    try:
        source_numbers, source_raw = load_source_numbers(args.source)
        findings, checked = check_csv(
            args.final_csv,
            source_numbers,
            value_columns=split(args.value_columns),
            skip_columns=split(args.skip_columns),
        )
        attr_findings, attr_checked = [], 0
        if args.attribution:
            if not args.label_columns:
                sys.stderr.write("error: --attribution requires --label-columns\n")
                return 2
            groups = build_groups(parse_markdown_tables(source_raw))
            attr_findings, attr_checked = check_attribution(
                args.final_csv,
                groups,
                label_columns=split(args.label_columns),
                value_columns=split(args.value_columns),
                skip_columns=split(args.skip_columns),
            )
    except (OSError, csv.Error) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    supported = checked - len(findings)
    if args.json:
        out = {
            "checked_numbers": checked,
            "supported": supported,
            "unsupported": len(findings),
            "findings": findings,
        }
        if args.attribution:
            out["attribution"] = {
                "checked_numbers": attr_checked,
                "misattributed": len(attr_findings),
                "findings": attr_findings,
            }
        print(json.dumps(out, indent=2))
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
        if args.attribution:
            ok = attr_checked - len(attr_findings)
            print(f"\nAttribution check: {ok}/{attr_checked} numbers under the "
                  f"best-matching source label.")
            if attr_findings:
                print(f"\n{len(attr_findings)} MISATTRIBUTED number(s) — present "
                      f"in the source but not under this row's label:")
                for f in attr_findings:
                    print(f"  row {f['row']}, column '{f['column']}': "
                          f"{f['number']:g}  (row labels: {', '.join(f['row_labels'])})")
                print("\nThese numbers exist in the source but appear to belong to "
                      "a different row/cohort/specimen. Re-check the attribution.")
            else:
                print("Every checked number sits under its row's best-matching "
                      "source label. ✓")

    return 1 if (findings or attr_findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
