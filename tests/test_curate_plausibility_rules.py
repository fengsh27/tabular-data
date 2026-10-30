"""Unit tests for extractor.curate_agent.plausibility_rules (Tier 3's rule table).

Covers the three outcomes check_row() can return: no opinion (parameter type
not in the table), clean (recognized but fine), and flagged (unit mismatch or
out-of-range magnitude) - independent of the CSV-reading wrapper in
validity_checker.py, which tests/test_curate_validity_checker.py covers.
"""

from extractor.curate_agent import plausibility_rules as pr


def test_unrecognized_parameter_type_has_no_opinion():
    assert pr.check_row("Total area under curve", "Mg/min/L", "91") is None


def test_recognized_clean_row_is_not_flagged():
    assert pr.check_row("Cmax", "ng/mL", "120") is None
    assert pr.check_row("Half-life", "h", "8") is None


def test_unit_mismatch_is_flagged():
    issues = pr.check_row("Cmax", "hr", "91")
    assert issues is not None
    assert any("unit" in i for i in issues)


def test_out_of_range_magnitude_is_flagged():
    issues = pr.check_row("Half-life", "h", "9000")
    assert issues is not None
    assert any("range" in i for i in issues)


def test_missing_unit_or_value_is_not_flagged_for_that_check():
    # An empty cell is a Tier 1 concern (structural completeness), not Tier 3's.
    assert pr.check_row("Cmax", "", "") is None
