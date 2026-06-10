"""Deterministic regression test for the pe-study-outcome cleanup script.

`clean_pe_outcome_rows.py` ports the legacy `pe_study_outcome_ver2` row-cleanup:
interval/statistic business rules + sentinel normalization + numeric filter +
rename/reorder. Deterministic, so asserted byte-exactly in CI.

Run:  poetry run pytest skills_e2e_tests/test_clean_pe_outcome_rows.py
"""
import importlib.util
import os

SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-pe-curation", "curation-common",
    "scripts",
    "clean_pe_outcome_rows.py",
)


def load_script():
    spec = importlib.util.spec_from_file_location("clean_pe_outcome_rows", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pe = load_script()


def wrow(**kw):
    base = {c: "N/A" for c in pe.WORKING_COLUMNS}
    base.update(kw)
    return base


def test_renames_and_reorders_to_final_schema():
    out = pe.clean_rows([wrow(**{"Main value": "3.2", "Statistics type": "Mean",
                                 "Main value unit": "kg", "Characteristic": "age"})])
    assert len(out) == 1
    r = out[0]
    assert r["Parameter value"] == "3.2"
    assert r["Parameter statistic"] == "Mean"
    assert r["Parameter unit"] == "kg"
    # all final columns are present
    assert set(pe.FINAL_COLUMNS).issubset(r.keys())


def test_drops_row_with_no_numeric_value():
    # nothing numeric in Main value / Variation type / Lower / Upper -> dropped
    out = pe.clean_rows([wrow(**{"Characteristic": "age", "Outcome": "sleep",
                                 "Main value": "N/A", "Variation type": "SD"})])
    assert out == []


def test_keeps_row_with_digit_in_bound():
    out = pe.clean_rows([wrow(**{"Main value": "N/A", "Lower bound": "1.0",
                                 "Upper bound": "2.0"})])
    assert len(out) == 1
    assert out[0]["Interval type"] == "Range"  # both bounds present -> Range


def test_both_bounds_present_sets_interval_range():
    out = pe.clean_rows([wrow(**{"Main value": "5", "Lower bound": "1",
                                 "Upper bound": "9", "Interval type": "N/A"})])
    assert out[0]["Interval type"] == "Range"


def test_main_value_equal_to_bound_is_blanked():
    out = pe.clean_rows([wrow(**{"Main value": "1", "Statistics type": "N/A",
                                 "Lower bound": "1", "Upper bound": "9"})])
    assert len(out) == 1  # survives on the bound digits
    assert out[0]["Parameter value"] == "N/A"
    assert out[0]["Parameter statistic"] == "N/A"


def test_main_value_containing_both_bounds_is_blanked():
    out = pe.clean_rows([wrow(**{"Main value": "1.5 (1.0-2.0)", "Lower bound": "1.0",
                                 "Upper bound": "2.0"})])
    assert out[0]["Parameter value"] == "N/A"


def test_sentinel_normalization():
    out = pe.clean_rows([wrow(**{"Main value": "3", "Main value unit": "",
                                 "Variation type": "Standard Deviation (SD)",
                                 "Variation value": "0.5"})])
    assert out[0]["Parameter unit"] == "N/A"      # blank -> N/A
    assert out[0]["Variation type"] == "SD"       # normalized


def test_na_variation_value_blanks_variation_type():
    out = pe.clean_rows([wrow(**{"Main value": "3", "Variation type": "SD",
                                 "Variation value": "N/A"})])
    assert out[0]["Variation type"] == "N/A"


def test_preserves_order_of_surviving_rows():
    rows = [wrow(**{"Main value": "10", "Characteristic": "a"}),
            wrow(**{"Main value": "N/A", "Characteristic": "b"}),   # dropped
            wrow(**{"Main value": "20", "Characteristic": "c"})]
    out = pe.clean_rows(rows)
    assert [r["Characteristic"] for r in out] == ["a", "c"]


def test_empty_input():
    assert pe.clean_rows([]) == []
