"""Post-hoc Tier-1 structural/plausibility scoring for a finished curated CSV.

`_common.py`/`_common_summary.py` already score per-cell plausibility via
`_fits()`, but only as a private helper `_realign()` uses at parse time, to
pick the least-bad gap-insertion when a model's output row is short a field.
This module re-exposes that same per-cell scoring rule as a standalone,
importable check runnable against *any* finished CSV on the pk_individual or
pk_summary schema - including one that never went through this project's own
parse_csv() at all, e.g. the multi-agent pipeline's output - not just the
simple-prompt arm's own output.

Parsing itself is unchanged: `_common.py`/`_common_summary.py` keep using
their own private `_fits`/`_realign` exactly as before. Nothing here rewrites,
drops, or realigns a row - it only scores what is already on disk.
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


def score_row(schema: str, row: dict) -> int:
    """Sum of `_fits(column, cell)` over every canonical column, for one row dict."""
    mod = _schema_module(schema)
    return sum(mod._fits(name, row.get(name, "")) for name in mod.COLS)


def score_table_validity(csv_path, schema: str) -> dict:
    """Score every row of an already-produced curated CSV.

    `csv_path` is read with `csv.DictReader` against the schema's own
    canonical headers (the same `combined.csv` shape both the simple-prompt
    and multi-agent arms produce) - a row missing or misnamed columns simply
    scores those cells as missing, the same as `_fits()` does for an empty
    cell. Returns `total_rows`, each row's fit score, and their average -
    the raw signal; what counts as "too many errors" is a decision for the
    caller (`ValidityChecker`), not this module.
    """
    mod = _schema_module(schema)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    row_scores = [sum(mod._fits(name, row.get(name, "")) for name in mod.COLS)
                  for row in rows]
    total_rows = len(rows)
    avg_fit_score = (sum(row_scores) / total_rows) if total_rows else 0.0

    return {
        "schema": schema,
        "total_rows": total_rows,
        "avg_fit_score": avg_fit_score,
        "row_scores": row_scores,
    }
