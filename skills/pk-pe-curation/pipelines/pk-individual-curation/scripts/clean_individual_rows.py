#!/usr/bin/env python3
"""Stage 13 of the pk-individual-curation skill: deterministic row cleanup.

Applies the mechanical normalization rules of the individual pipeline to the
assembled 12-column CSV, producing the final curated CSV. These rules are pure
bookkeeping (blanking, dropping, de-duping) — exactly the kind of thing an LLM
does inconsistently — so they live in code, with a regression test, rather than
in a prompt.

Rules (in order):
  1. Drop any row containing the literal "ERROR".
  2. Long time units (weeks / months / years, any spelling) are not sampling
     times -> blank the Time value.
  3. Couple Time value and Time unit: if either is N/A, set both N/A.
  4. If Parameter value is N/A, set Parameter type and Parameter unit N/A.
  5. Cmax / Tmax / Cavg carry no sampling time -> blank Time value and unit.
  6. Normalize blanks/sentinels to "N/A" (empty, n/a, unknown, nan); turn a
     stray comma cell into a space.
  7. Drop rows whose Parameter value is N/A (nothing measured).
  8. Drop duplicate rows.
  9. Put Patient ID first, then the canonical column order.

Schema-aware but tolerant: a rule is skipped if its column is absent.

Usage:
    python clean_individual_rows.py 12_assembled.csv            # -> stdout
    python clean_individual_rows.py 12_assembled.csv -o 13_final.csv
"""

import argparse
import csv
import io
import sys

EXPECTED_COLUMNS = [
    "Patient ID", "Drug name", "Analyte", "Specimen",
    "Population", "Pregnancy stage", "Pediatric/Gestational age",
    "Parameter type", "Parameter unit", "Parameter value",
    "Time value", "Time unit",
]

NA = "N/A"

# Time units that denote an age/duration rather than a sampling time point.
LONG_TIME_UNITS = {
    "week", "weeks", "wk", "wks", "w",
    "month", "months", "mo", "mos",
    "year", "years", "yr", "yrs", "y",
}

# Cell values normalized to N/A.
NA_LIKE = {"", "n/a", "na", "unknown", "nan", "none"}

# Parameter types that never carry a sampling time.
NO_TIME_PARAMS = {"cmax", "tmax", "cavg"}


def _norm_cell(v):
    s = (v if v is not None else "").strip()
    if s.lower() in NA_LIKE:
        return NA
    if s == ",":
        return " "
    return s


def clean_rows(rows):
    """Apply the cleanup rules to a list of dict rows; return cleaned list."""
    out = []
    for row in rows:
        r = {k: (v if v is not None else "") for k, v in row.items()}

        # 1. drop ERROR rows
        if any(str(v).strip() == "ERROR" for v in r.values()):
            continue

        def get(col):
            return (r.get(col) or "").strip()

        # 2. long time units -> blank time value
        if "Time unit" in r and get("Time unit").lower() in LONG_TIME_UNITS:
            r["Time value"] = NA

        # 5. Cmax/Tmax/Cavg -> no time  (before coupling so unit is blanked too)
        if "Parameter type" in r and get("Parameter type").lower() in NO_TIME_PARAMS:
            if "Time value" in r:
                r["Time value"] = NA
            if "Time unit" in r:
                r["Time unit"] = NA

        # 6. normalize sentinels first so the couplings below see N/A
        r = {k: _norm_cell(v) for k, v in r.items()}

        # 3. couple time value/unit
        if "Time value" in r and "Time unit" in r:
            if r["Time value"] == NA or r["Time unit"] == NA:
                r["Time value"] = NA
                r["Time unit"] = NA

        # 4. no value -> no type/unit
        if "Parameter value" in r and r["Parameter value"] == NA:
            if "Parameter type" in r:
                r["Parameter type"] = NA
            if "Parameter unit" in r:
                r["Parameter unit"] = NA

        out.append(r)

    # 7. drop rows with N/A parameter value
    if out and "Parameter value" in out[0]:
        out = [r for r in out if r.get("Parameter value") != NA]

    # 8. drop duplicates (preserve first occurrence, stable order)
    seen = set()
    deduped = []
    for r in out:
        key = tuple(sorted(r.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def order_columns(fieldnames):
    """Patient ID first, then expected order, then any extras (stable)."""
    present = list(fieldnames)
    ordered = [c for c in EXPECTED_COLUMNS if c in present]
    extras = [c for c in present if c not in EXPECTED_COLUMNS]
    return ordered + extras


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_csv", help="the assembled 12-column CSV (stage 12 output)")
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
    columns = order_columns(fieldnames)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in cleaned:
        writer.writerow({c: r.get(c, "") for c in columns})
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
