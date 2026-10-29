"""Tiers 1-2 of the ValidityChecker, per docs/VERIFICATION_AGENT_DESIGN.md.

Combines the Tier 1 structural/plausibility score (`scripts/simple_prompt/
validity.py`, wrapping `_fits()`) and the Tier 2 provenance-attribution check
(`provenance_adapter.check_attribution`, wrapping `verify_provenance.py`) into
one `ValidityReport` per candidate. Tier 2 is skipped - not merely zero - on a
zero-row table: there is nothing to check provenance on, and the table is
already invalid regardless of what Tier 2 would say.

Tier 3 (plausibility rules) and Tier 4 (LLM semantic check) are out of scope
here; see the design doc's Phase 2+.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMPLE_PROMPT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "scripts", "simple_prompt"))
if _SIMPLE_PROMPT_DIR not in sys.path:
    sys.path.insert(0, _SIMPLE_PROMPT_DIR)

import validity as _validity  # scripts/simple_prompt/validity.py

from extractor.curate_agent import provenance_adapter

# Not calibrated yet - Phase 5 picks this against benchmark_results/
# 20261007-baseline-fix/'s simple_qwen38 vs multiagent_gemma4 outcomes. This
# default only needs to separate the two already-diagnosed simple_qwen38 bugs
# from a clean table for Phase 1's regression test, not decide a production
# value.
DEFAULT_THRESHOLD = 0.2

# A row scoring at or below this under `_fits()` stands in for "dropped" when
# the caller has no externally-tracked drop count - e.g. the multi-agent arm,
# which never goes through parse_csv()'s drop step at all, so it has no
# manifest.json entry to read a real count from.
DEFAULT_LOW_FIT_THRESHOLD = 0


def tier1_structural(csv_path, schema, extra_dropped_rows=0,
                      low_fit_threshold=DEFAULT_LOW_FIT_THRESHOLD):
    """Structural/plausibility signal: dropped_rows, total_rows, avg_fit_score.

    Two distinct failure shapes both count, additively, not either/or:

    - `extra_dropped_rows`: rows missing from `csv_path` entirely because the
      parser could not realign them. These never reach the file at all, so
      they also never show up in `score_table_validity`'s row scores - pass
      the real count when the caller has one (Arm A's manifest.json already
      records how many rows `parse_csv` dropped; Arm B has no equivalent
      step, so it defaults to 0).
    - Rows that DID reach the file but still score at or below
      `low_fit_threshold` under `_fits()` - a count-correct-but-shifted row,
      see docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 1, which a parser never
      drops because its field count matches. Derived fresh every time, for
      both arms, from the per-row fit score.

    `total_rows` is incremented by `extra_dropped_rows` too: the denominator
    is "how many rows were attempted," not just "how many survived to the
    file" - otherwise a paper that drops most of its rows could show an
    artificially small denominator.
    """
    scored = _validity.score_table_validity(csv_path, schema)
    low_fit_rows = sum(1 for s in scored["row_scores"] if s <= low_fit_threshold)
    return {
        "dropped_rows": extra_dropped_rows + low_fit_rows,
        "total_rows": scored["total_rows"] + extra_dropped_rows,
        "avg_fit_score": scored["avg_fit_score"],
    }


def tier2_provenance(csv_path, source_paths, label_columns,
                      value_columns=None, skip_columns=None):
    """Provenance/attribution signal: checked_numbers, unsupported, findings.

    Uses attribution mode, not existence mode: existence only catches
    hallucinated values, attribution also catches column-shift/misattribution
    (docs/VERIFICATION_AGENT_DESIGN.md, Tier 2).
    """
    report = provenance_adapter.check_attribution(
        csv_path, source_paths, label_columns,
        value_columns=value_columns, skip_columns=skip_columns,
    )
    return {
        "checked_numbers": report["checked_numbers"],
        "unsupported": report["misattributed"],
        "findings": report["findings"],
    }


def check_validity(csv_path, schema, source_paths, label_columns,
                    extra_dropped_rows=0, low_fit_threshold=DEFAULT_LOW_FIT_THRESHOLD,
                    threshold=DEFAULT_THRESHOLD,
                    value_columns=None, skip_columns=None):
    """Full Tier 1 + Tier 2 ValidityReport for one candidate's finished CSV.

    Shape matches docs/VERIFICATION_AGENT_DESIGN.md's ValidityReport contract:
    {"tier1": {...}, "tier2": {...}, "error_rate": float, "escalate": bool}.
    `tier2` is None when Tier 1 already found a zero-row table.
    """
    tier1 = tier1_structural(csv_path, schema, extra_dropped_rows=extra_dropped_rows,
                              low_fit_threshold=low_fit_threshold)

    if tier1["total_rows"] == 0:
        return {"tier1": tier1, "tier2": None, "error_rate": 1.0, "escalate": True}

    tier2 = tier2_provenance(csv_path, source_paths, label_columns,
                              value_columns=value_columns, skip_columns=skip_columns)

    error_rate = (tier1["dropped_rows"] + tier2["unsupported"]) / tier1["total_rows"]
    return {
        "tier1": tier1,
        "tier2": tier2,
        "error_rate": error_rate,
        "escalate": error_rate > threshold,
    }
