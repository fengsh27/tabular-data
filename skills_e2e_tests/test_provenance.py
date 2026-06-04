"""Deterministic regression test for the provenance checker.

Covers the script's contract: numbers in the curated CSV must trace to the
source; categorical/text cells are ignored; ranges split into their bounds;
and --skip-columns / --value-columns scope which columns are checked.

The script is model-independent, so these are true byte-level asserts (safe for
CI), unlike the LLM-driven curation stages.

Run:  poetry run pytest skills_e2e_tests/test_provenance.py
"""
import importlib.util
import os

import pytest

SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-summary-curation",
    "scripts",
    "verify_provenance.py",
)


def load_script():
    spec = importlib.util.spec_from_file_location("verify_provenance", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vp = load_script()


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# A small source: caption + a 2-row summary table with interval bounds.
SOURCE = """
Table X. Lorazepam in maternal and cord blood (n = 8); mean (CI 95%).
| Specimen | Mean CI 95% |
| --- | --- |
| Cord blood | 6.78 (5.39-8.17) |
| Maternal blood | 9.91 (7.68-12.14) |
"""

GOOD_CSV = """Drug,Specimen,Subject N,Parameter value,Interval type,Lower bound,Upper bound
Lorazepam,Cord blood,8,6.78,95% CI,5.39,8.17
Lorazepam,Maternal blood,8,9.91,95% CI,7.68,12.14
"""


def test_extract_numbers_splits_ranges_and_ignores_sign():
    # en-dash and ascii-hyphen ranges both split into two positive numbers
    assert vp.extract_numbers("6.78 (5.39-8.17)") == {6.78, 5.39, 8.17}
    assert vp.extract_numbers("0-12") == {0.0, 12.0}
    assert vp.extract_numbers(".67") == {0.67}


def test_all_numbers_supported(tmp_path):
    csv_path = write(tmp_path, "good.csv", GOOD_CSV)
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", SOURCE)])
    findings, checked = vp.check_csv(csv_path, src_nums)
    assert findings == []
    # per row: Subject N(8), Parameter value, 95 (from "95% CI"), Lower, Upper = 5; x2 rows
    assert checked == 10


def test_hallucinated_number_flagged(tmp_path):
    bad = GOOD_CSV.replace("6.78", "6.99")  # not present in source
    csv_path = write(tmp_path, "bad.csv", bad)
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", SOURCE)])
    findings, _ = vp.check_csv(csv_path, src_nums)
    assert len(findings) == 1
    assert findings[0]["number"] == 6.99
    assert findings[0]["column"] == "Parameter value"
    assert findings[0]["row"] == 0


def test_na_and_text_cells_ignored(tmp_path):
    csv = ("Drug,Specimen,Parameter value,P value\n"
           "Lorazepam,Cord blood,N/A,N/A\n")  # no numbers to check
    csv_path = write(tmp_path, "na.csv", csv)
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", SOURCE)])
    findings, checked = vp.check_csv(csv_path, src_nums)
    assert findings == []
    assert checked == 0


def test_skip_columns_suppresses_normalized_label(tmp_path):
    # "Trimester 1" embeds a 1 that is not in the source -> flagged by default,
    # suppressed when the column is skipped.
    csv = ("Specimen,Pregnancy stage,Parameter value\n"
           "Cord blood,Trimester 1,6.78\n")
    csv_path = write(tmp_path, "label.csv", csv)
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", SOURCE)])

    flagged, _ = vp.check_csv(csv_path, src_nums)
    assert any(f["column"] == "Pregnancy stage" and f["number"] == 1.0 for f in flagged)

    suppressed, _ = vp.check_csv(csv_path, src_nums, skip_columns=["Pregnancy stage"])
    assert suppressed == []


def test_value_columns_restricts_scope(tmp_path):
    csv_path = write(tmp_path, "good.csv", GOOD_CSV)
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", SOURCE)])
    # Only check "Parameter value"; tamper an UNCHECKED column -> still clean.
    tampered = GOOD_CSV.replace(",8,", ",999,")  # Subject N column
    tcsv = write(tmp_path, "t.csv", tampered)
    findings, _ = vp.check_csv(tcsv, src_nums, value_columns=["Parameter value"])
    assert findings == []


def test_main_exit_codes(tmp_path):
    src = write(tmp_path, "src.md", SOURCE)
    good = write(tmp_path, "good.csv", GOOD_CSV)
    bad = write(tmp_path, "bad.csv", GOOD_CSV.replace("6.78", "6.99"))
    assert vp.main(["prog", good, src]) == 0
    assert vp.main(["prog", bad, src]) == 1
