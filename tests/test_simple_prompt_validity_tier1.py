"""Tests for scripts/simple_prompt/validity.py's pk_summary Tier 1 scoring.

Covers the fix made after a real-data investigation (PMID 16143486, see
validity.py's own module docstring and docs/cascade_mode_results.md): the
vocabulary-whitelist check is dropped for pk_summary (it rewards a wrong
label as readily as a correct one and penalizes a correct synonym), and an
"N/A"-style placeholder in a numeric column now scores as blank, not as
garbage. pk_individual is untouched and not covered here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))

import validity as v  # noqa: E402


def test_na_in_numeric_column_scores_as_blank():
    assert v._structural_fits("pk_summary", "Subject N", "N/A") == 0
    assert v._structural_fits("pk_summary", "Subject N", "") == 0
    assert v._structural_fits("pk_summary", "Subject N", "n / a") == 0


def test_na_case_insensitive_and_variants():
    for variant in ("N/A", "n/a", "NA", "na", "None", "Not Applicable"):
        assert v._structural_fits("pk_summary", "P value", variant) == 0


def test_garbage_in_numeric_column_still_penalized():
    # Real text that isn't a recognized "not applicable" placeholder and
    # isn't a number either - still a genuine structural problem.
    assert v._structural_fits("pk_summary", "Value", "unknown") == -3


def test_numeric_value_still_rewarded():
    assert v._structural_fits("pk_summary", "Value", "91") == 2


def test_vocabulary_column_is_not_scored_either_way():
    # Neither a listed word, a synonym, nor an outright wrong label gets an
    # opinion here any more - that's Tier 3's job (it reads the source).
    assert v._structural_fits("pk_summary", "Pregnancy stage", "delivery") == 0
    assert v._structural_fits("pk_summary", "Pregnancy stage", "Parturition") == 0
    assert v._structural_fits("pk_summary", "Interval type", "Range") == 0
    assert v._structural_fits("pk_summary", "Interval type", "95% CI") == 0


def test_required_identifier_columns_still_checked():
    # Not vocabulary-governed - "must be present, must not be a bare number."
    assert v._structural_fits("pk_summary", "Drug name", "") == -1
    assert v._structural_fits("pk_summary", "Drug name", "123") == -2
    assert v._structural_fits("pk_summary", "Drug name", "Ketamine") == 0


def test_text_column_rejects_bare_number():
    assert v._structural_fits("pk_summary", "Unit", "91") == -2
    assert v._structural_fits("pk_summary", "Unit", "mg/ml") == 0


def test_column_shift_bug_still_caught():
    # The real Arm A bug this check exists to catch: a parameter name
    # landing in the numeric "Value" cell.
    assert v._structural_fits("pk_summary", "Value", "ClT/F") == -3


def test_pk_individual_scoring_is_unchanged():
    # pk_individual wasn't touched - its own _fits() still runs as-is,
    # vocabulary branch included.
    import _common as common_mod
    assert v._structural_fits("pk_individual", "Population", "maternal") == \
        common_mod._fits("Population", "maternal")
    assert v._structural_fits("pk_individual", "Population", "maternal") == 2


def test_real_arm_b_row_no_longer_scores_deeply_negative():
    # The exact row from PMID 16143486's multi-agent output that motivated
    # this fix (see module docstring) - previously totaled -7.
    row = {
        "Drug name": "rac-lorazepam", "Analyte": "Lorazepam", "Specimen": "Plasma",
        "Population": "Maternal", "Pregnancy stage": "Parturition", "Subject N": "N/A",
        "Parameter type": "AUC0–∞",
        # "Value"/"Unit"/"Summary statistics"/"Lower limit"/"High limit" absent -
        # this arm spells them differently; see module docstring.
        "Variation type": "N/A", "Variation value": "N/A", "Interval type": "Range",
        "P value": "N/A",
    }
    assert v.score_row("pk_summary", row) == 0
