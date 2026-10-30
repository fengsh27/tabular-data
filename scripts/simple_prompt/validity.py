"""Post-hoc Tier-1 structural/plausibility scoring for a finished curated CSV.

`_common.py`/`_common_summary.py` already score per-cell plausibility via
`_fits()`, but only as a private helper `_realign()` uses at parse time, to
pick the least-bad gap-insertion when a model's output row is short a field.
This module re-exposes a scoring rule as a standalone, importable check
runnable against *any* finished CSV on the pk_individual or pk_summary
schema - including one that never went through this project's own
parse_csv() at all, e.g. the multi-agent pipeline's output - not just the
simple-prompt arm's own output.

Parsing itself is unchanged: `_common.py`/`_common_summary.py` keep using
their own private `_fits`/`_realign` exactly as before. Nothing here rewrites,
drops, or realigns a row - it only scores what is already on disk.

For pk_summary, this module's own score deliberately diverges from
`_common_summary._fits()`, after a real-data investigation (PMID 16143486)
showed `_fits()` manufacturing a false structural-failure signal on the
multi-agent arm's output, severe enough to flip the cascade's A-vs-B pick
away from the arm that was actually closer to gold. Two things were wrong,
and `_structural_fits()` below fixes both:

- `_fits()`'s per-cell check is done by `csv.DictReader` column *name*, not
  position. The multi-agent arm spells 5 of the 16 canonical columns
  differently (`"Parameter value"` for `"Value"`, `"Parameter unit"` for
  `"Unit"`, `"Parameter statistic"` for `"Summary statistics"`, `"Lower
  bound"`/`"Upper bound"` for `"Lower limit"`/`"High limit"`) - `_fits()`
  then can't find those columns at all and silently scores them as blank,
  even though the real values are sitting right there under a different
  spelling. Not fixed here (the mismatch still means those 5 columns go
  unscored), but no longer compounded by...
- `_fits()`'s vocabulary-whitelist check (Population, Pregnancy stage,
  Summary statistics, Variation type, Interval type: "is this cell one of a
  short list of expected words") turns out to rubber-stamp a wrong label as
  readily as a correct one - a real case had the source paper say "95% CI"
  and the multi-agent arm write "Range," both equally on the whitelist, both
  scored +2, with no way to tell them apart without reading the source
  paper, which this check never does. It also penalizes a *correct* label
  phrased as a synonym ("Parturition" for "delivery") exactly as hard as a
  genuinely wrong one. `_structural_fits()` drops this check entirely for
  pk_summary (scores those 5 columns as 0, no opinion) rather than try to
  tune an unwinnable whitelist - judging label correctness against the
  source is Tier 3's job (it actually reads the source), not Tier 1's.
  Kept: the numeric-vs-garbage check (a real signal - it is what catches an
  actual column-shift bug, e.g. a parameter name like "ClT/F" landing in a
  numeric "Value" cell) and the required-identifier check (Drug name/
  Parameter type must not be blank or a bare number), both of which don't
  depend on matching specific wording. Also: a "N/A"-style placeholder in a
  numeric column now scores as blank (0), not as garbage (-3) - a model
  writing "N/A" for "not applicable" is not a sign of a structural error.

pk_individual is NOT changed here - this hasn't been investigated against
real pk_individual data the way pk_summary was, and pk_individual isn't
wired into the live cascade (`curate_orchestrator.SUPPORTED_SCHEMAS`) at all
yet, so there's no decision this would currently affect either way.
"""

from __future__ import annotations

import csv
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Same schema -> module mapping run_simple_prompt.py uses, so --schema picks
# the same column set / _fits() rule here as it does during parsing.
SCHEMA_MODULES = {"pk_individual": "_common", "pk_summary": "_common_summary"}

# A model writing "not applicable" this way is not reporting garbage - treat
# it the same as a blank cell. See module docstring.
_NA_LIKE = {"n/a", "na", "n / a", "none", "not applicable"}


def _schema_module(schema: str):
    try:
        mod_name = SCHEMA_MODULES[schema]
    except KeyError:
        raise ValueError(
            f"unknown schema {schema!r}; expected one of {sorted(SCHEMA_MODULES)}"
        )
    return importlib.import_module(mod_name)


def schema_columns(schema: str) -> list[str]:
    """The canonical column list for `schema` (`_common.COLS` / `_common_summary.COLS`)."""
    return list(_schema_module(schema).COLS)


def _structural_fits(schema: str, name: str, cell: str) -> int:
    """Tier 1's own per-cell score - see module docstring for why this isn't
    just `mod._fits()` for pk_summary. pk_individual is unchanged."""
    mod = _schema_module(schema)
    if schema != "pk_summary":
        return mod._fits(name, cell)

    c = (cell or "").strip()
    if c.lower() in _NA_LIKE:
        c = ""

    if name in mod.NUMERIC_COLS:
        if not c:
            return 0
        return 2 if mod._VALUE_RE.match(c) else -3
    if name in ("Drug name", "Parameter type"):
        if not c:
            return -1
        return -2 if mod._VALUE_RE.match(c) else 0
    if name in mod._TEXT_COLS:
        return -2 if mod._VALUE_RE.match(c) else 0
    return 0  # vocabulary-governed column - not checked here, see module docstring


def score_row(schema: str, row: dict) -> int:
    """Sum of `_structural_fits(column, cell)` over every canonical column, for one row dict."""
    mod = _schema_module(schema)
    return sum(_structural_fits(schema, name, row.get(name, "")) for name in mod.COLS)


def score_table_validity(csv_path, schema: str) -> dict:
    """Score every row of an already-produced curated CSV.

    `csv_path` is read with `csv.DictReader` against the schema's own
    canonical headers (the same `combined.csv` shape both the simple-prompt
    and multi-agent arms produce) - a row missing or misnamed columns simply
    scores those cells as missing, the same as `_structural_fits()` does for
    an empty cell. Returns `total_rows`, each row's fit score, and their
    average - the raw signal; what counts as "too many errors" is a decision
    for the caller (`ValidityChecker`), not this module.
    """
    mod = _schema_module(schema)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    row_scores = [sum(_structural_fits(schema, name, row.get(name, "")) for name in mod.COLS)
                  for row in rows]
    total_rows = len(rows)
    avg_fit_score = (sum(row_scores) / total_rows) if total_rows else 0.0

    return {
        "schema": schema,
        "total_rows": total_rows,
        "avg_fit_score": avg_fit_score,
        "row_scores": row_scores,
    }
