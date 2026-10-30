"""The ValidityChecker, per docs/VERIFICATION_AGENT_DESIGN.md.

Three tiers, cheapest first:

- Tier 1 (structural): `scripts/simple_prompt/validity.py`'s `_fits()`-based
  per-row plausibility score. Deterministic, free.
- Tier 2 (existence): does every checkable number in the curated table
  appear somewhere in the source? Deterministic, free -
  `verify_provenance.py`'s existence mode. Catches hallucinated/typo'd
  numbers; cannot catch misattribution (a value that exists but sits on the
  wrong row) - Tier 3 covers that.
- Tier 3 (LLM verification): `llm_verification.verify()` - asks the same
  verification agent multi-agent mode's own verify/correct loop uses to
  check the curated table against its source, once, no correction loop.
  Catches misattribution and unit/magnitude implausibility without a
  maintained matching heuristic or rule table.

Both Tier 2 and Tier 3 are skipped - not merely zero - on a zero-row table:
there is nothing to check, and the table is already invalid regardless, so
no LLM call is spent on it either.

Earlier versions of this checker had a Tier 2 based on Jaccard label-matching
and a Tier 3 based on a hand-maintained unit/magnitude rule table; both were
removed after real-data calibration runs
(`scripts/calibrate_validity_threshold.py`) showed the label-matching
approach flags mostly false positives on multi-level-header source tables,
and the rule table had a real false-positive bug on the micro-sign unit
character. See git history for that code if needed.
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd
from TabFuncFlow.utils.table_utils import dataframe_to_markdown

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMPLE_PROMPT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "scripts", "simple_prompt"))
if _SIMPLE_PROMPT_DIR not in sys.path:
    sys.path.insert(0, _SIMPLE_PROMPT_DIR)
import validity as _validity  # scripts/simple_prompt/validity.py

_CURATION_COMMON_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "..", "skills", "pk-pe-curation", "curation-common", "scripts"))
if _CURATION_COMMON_DIR not in sys.path:
    sys.path.insert(0, _CURATION_COMMON_DIR)
import verify_provenance as _verify_provenance  # skills/.../curation-common/scripts/verify_provenance.py

from extractor.curate_agent import llm_verification

# Not calibrated yet - Phase 5 picks this against real benchmark data (see
# scripts/calibrate_validity_threshold.py). This default only needs to
# separate the already-diagnosed simple_qwen38 bugs from a clean table for
# Phase 1's regression test, not decide a production value.
DEFAULT_THRESHOLD = 0.2

# A row scoring at or below this under `score_table_validity()` stands in for
# "dropped" when the caller has no externally-tracked drop count - e.g. the
# multi-agent arm, which never goes through parse_csv()'s drop step at all,
# so it has no manifest.json entry to read a real count from.
#
# pk_summary gets its own, lower default: `validity.py`'s pk_summary scorer
# no longer hands out a reward for matching its vocabulary whitelist (see
# that module's docstring for why), so a row with nothing structurally wrong
# and nothing in those reward categories now nets to a neutral 0, not a
# comfortably positive score the way it used to. Keeping the cutoff at 0
# would flag that neutral row as "dropped" - only an actual negative signal
# (0 passed `_structural_fits()` through its NUMERIC_COLS or
# Drug-name/Parameter-type/_TEXT_COLS checks and still came back wrong)
# should count. pk_individual's own `_fits()` is unchanged, so its default
# stays where it was.
DEFAULT_LOW_FIT_THRESHOLD = 0
_LOW_FIT_THRESHOLD_BY_SCHEMA = {"pk_summary": -1}


def tier1_structural(csv_path, schema, extra_dropped_rows=0, low_fit_threshold=None):
    """Structural/plausibility signal: dropped_rows, total_rows, avg_fit_score.

    Two distinct failure shapes both count, additively, not either/or:

    - `extra_dropped_rows`: rows missing from `csv_path` entirely because the
      parser could not realign them. These never reach the file at all, so
      they also never show up in `score_table_validity`'s row scores - pass
      the real count when the caller has one (Arm A's manifest.json already
      records how many rows `parse_csv` dropped; Arm B has no equivalent
      step, so it defaults to 0).
    - Rows that DID reach the file but still score at or below
      `low_fit_threshold` - a count-correct-but-shifted row, see
      docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 1, which a parser never drops
      because its field count matches. Derived fresh every time, for both
      arms, from the per-row fit score.

    `low_fit_threshold=None` (the default) resolves to
    `_LOW_FIT_THRESHOLD_BY_SCHEMA`'s per-schema value - see that dict's
    comment. Pass an explicit value to override.

    `total_rows` is incremented by `extra_dropped_rows` too: the denominator
    is "how many rows were attempted," not just "how many survived to the
    file" - otherwise a paper that drops most of its rows could show an
    artificially small denominator.
    """
    if low_fit_threshold is None:
        low_fit_threshold = _LOW_FIT_THRESHOLD_BY_SCHEMA.get(schema, DEFAULT_LOW_FIT_THRESHOLD)
    scored = _validity.score_table_validity(csv_path, schema)
    low_fit_rows = sum(1 for s in scored["row_scores"] if s <= low_fit_threshold)
    return {
        "dropped_rows": extra_dropped_rows + low_fit_rows,
        "total_rows": scored["total_rows"] + extra_dropped_rows,
        "avg_fit_score": scored["avg_fit_score"],
    }


def tier2_existence(csv_path, source_paths, value_columns=None, skip_columns=None):
    """Existence signal: checked_numbers, unsupported, findings.

    Deliberately existence mode, not attribution mode (removed - see module
    docstring): this only asks "does this number appear anywhere in the
    source," which can't catch misattribution - Tier 3's LLM check covers
    that instead. Kept because it still catches pure hallucinations/typos
    for free, before paying for an LLM call.
    """
    source_numbers, _ = _verify_provenance.load_source_numbers(source_paths)
    findings, checked = _verify_provenance.check_csv(
        csv_path, source_numbers, value_columns=value_columns, skip_columns=skip_columns,
    )
    return {"checked_numbers": checked, "unsupported": len(findings), "findings": findings}


def check_validity(csv_path, schema, source_paths, paper_title, paper_abstract,
                    extra_dropped_rows=0, low_fit_threshold=None,
                    threshold=DEFAULT_THRESHOLD,
                    value_columns=None, skip_columns=None,
                    verify_fn=llm_verification.verify, llm=None,
                    domain="pharmacokinetics", include_tier1_rate=True):
    """Full Tier 1 + Tier 2 + Tier 3 ValidityReport for one candidate's CSV.

    Shape: {"tier1": {...}, "tier2": {...}, "tier3": {...}, "error_rate":
    float, "escalate": bool}. `tier2`/`tier3` are both None when Tier 1
    already found a zero-row table - no LLM call is made on an empty table.

    `verify_fn` defaults to the real `llm_verification.verify` (a live LLM
    call); tests substitute a fake so this can be exercised without one.

    `error_rate` averages each tier's own 0-1 failure fraction (not a sum of
    raw counts over `total_rows` - see git history for why that broke).
    Tier 1's rate is dropped_rows/total_rows, included only when
    `include_tier1_rate` is true (see below). Tier 2's rate is
    unsupported/checked_numbers, left out of the average (not counted as 0)
    when there was nothing table-sourced to check. Tier 3's rate is
    (error_row_count + missed_row_count)/(total_rows + missed_row_count):
    missed_row_count joins the denominator too, not just the numerator -
    those rows were never attempted by the parser, so Tier 1's total_rows
    (rows that made it to the CSV plus rows the parser attempted and
    dropped) never counted them in the first place. Leaving the denominator
    as plain total_rows let this rate exceed 1.0 whenever the agent reported
    more missing rows than the table even has - not a deliberate property,
    just the numerator counting something the denominator didn't.

    `include_tier1_rate=False` excludes Tier 1's rate from `error_rate`
    (Tier 1 itself is still computed - `total_rows` feeds Tier 3's
    denominator and the zero-row short-circuit below either way). A
    real-data investigation (PMID 16143486, see docs/cascade_mode_results.md
    and scripts/simple_prompt/validity.py's docstring) showed Tier 1's
    structural check - built around the simple-prompt arm's own output
    conventions - manufacturing a false failure signal severe enough to flip
    which of two candidates `curate_orchestrator.py` picks, when scored
    against the multi-agent arm's differently-formatted (but not actually
    wrong) output. `curate_orchestrator.py` passes `include_tier1_rate=False`
    for the multi-agent arm's own check for exactly this reason - Tier 1 was
    never validated against that arm's conventions, and that arm's own
    output never goes through the realignment step Tier 1's structural bug
    (a count-correct-but-column-shifted row) can even occur in.
    """
    tier1 = tier1_structural(csv_path, schema, extra_dropped_rows=extra_dropped_rows,
                              low_fit_threshold=low_fit_threshold)

    if tier1["total_rows"] == 0:
        return {"tier1": tier1, "tier2": None, "tier3": None, "error_rate": 1.0, "escalate": True}

    tier2 = tier2_existence(csv_path, source_paths, value_columns=value_columns,
                             skip_columns=skip_columns)

    source_tables_text = "\n\n".join(
        open(p, encoding="utf-8").read() for p in source_paths
    )
    curated_table_md = dataframe_to_markdown(pd.read_csv(csv_path))
    tier3_result = verify_fn(
        paper_title=paper_title, paper_abstract=paper_abstract,
        source_tables=source_tables_text, curated_table=curated_table_md,
        llm=llm, domain=domain,
    )
    tier3 = {
        "correct": tier3_result["correct"],
        "error_row_count": tier3_result["error_row_count"],
        "missed_row_count": tier3_result["missed_row_count"],
        "explanation": tier3_result["explanation"],
    }

    rates = []
    if include_tier1_rate:
        rates.append(tier1["dropped_rows"] / tier1["total_rows"])
    if tier2["checked_numbers"] > 0:
        rates.append(tier2["unsupported"] / tier2["checked_numbers"])
    tier3_total = tier1["total_rows"] + tier3["missed_row_count"]
    rates.append((tier3["error_row_count"] + tier3["missed_row_count"]) / tier3_total)
    error_rate = sum(rates) / len(rates)

    return {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "error_rate": error_rate,
        "escalate": error_rate > threshold,
    }
