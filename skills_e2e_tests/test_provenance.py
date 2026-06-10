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
    "pk-pe-curation", "curation-common",
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


# --- Attribution check (the misattribution gap existence cannot close) -------
#
# A column-oriented source table (the original, pre-transpose shape) whose
# discriminator lives in the COLUMN HEADERS (Cord blood vs Maternal blood), and
# a curated (transposed) output where each row claims a specimen. The deliberate
# swap below moves the maternal values onto the cord-blood row and vice versa —
# every number still EXISTS in the source, so the existence check passes; only
# the attribution check can catch it.

ATTR_SOURCE = """
Table 4. Transplacental distribution of lorazepam at delivery (n = 8); mean (CI 95%).
| Parturient | Cord blood (ng/ml) | Maternal blood (ng/ml) | Collection time(min) | Cord blood/maternal blood |
| --- | --- | --- | --- | --- |
| Mean CI 95% | 6.78 (5.39-8.17) | 9.91 (7.68-12.14) | 293.4 (163.2-423) | 0.73 (0.52-0.94) |
"""

ATTR_GOOD = (
    "Parameter type,Specimen,Parameter value,Lower bound,Upper bound\n"
    "Cord blood concentration,Cord blood,6.78,5.39,8.17\n"
    "Maternal blood concentration,Maternal blood,9.91,7.68,12.14\n"
)

# cord row gets maternal's numbers and vice versa
ATTR_SWAPPED = (
    "Parameter type,Specimen,Parameter value,Lower bound,Upper bound\n"
    "Cord blood concentration,Cord blood,9.91,7.68,12.14\n"
    "Maternal blood concentration,Maternal blood,6.78,5.39,8.17\n"
)

ATTR_LABELS = ["Parameter type", "Specimen"]
ATTR_VALUES = ["Parameter value", "Lower bound", "Upper bound"]


def _groups():
    return vp.build_groups(vp.parse_markdown_tables(ATTR_SOURCE))


def test_parse_markdown_tables_reads_headers_and_rows():
    tables = vp.parse_markdown_tables(ATTR_SOURCE)
    assert len(tables) == 1
    assert tables[0]["headers"][1] == "Cord blood (ng/ml)"
    assert tables[0]["rows"][0][0] == "Mean CI 95%"


def test_build_groups_indexes_columns_by_header():
    groups = dict((lbl, nums) for lbl, nums in _groups())
    assert groups["Cord blood (ng/ml)"] == {6.78, 5.39, 8.17}
    assert groups["Maternal blood (ng/ml)"] == {9.91, 7.68, 12.14}


def test_attribution_passes_correct_rows(tmp_path):
    csv_path = write(tmp_path, "good.csv", ATTR_GOOD)
    findings, checked = vp.check_attribution(
        csv_path, _groups(), label_columns=ATTR_LABELS, value_columns=ATTR_VALUES
    )
    assert findings == []
    assert checked == 6  # 3 values x 2 rows


def test_attribution_catches_value_swap(tmp_path):
    swapped = write(tmp_path, "swapped.csv", ATTR_SWAPPED)

    # Existence check is BLIND to the swap — every number is still in the source.
    src_nums, _ = vp.load_source_numbers([write(tmp_path, "src.md", ATTR_SOURCE)])
    existence, _ = vp.check_csv(swapped, src_nums, value_columns=ATTR_VALUES)
    assert existence == [], "existence check should not catch a pure swap"

    # Attribution check DOES catch it, on both rows.
    findings, _ = vp.check_attribution(
        swapped, _groups(), label_columns=ATTR_LABELS, value_columns=ATTR_VALUES
    )
    assert {f["row"] for f in findings} == {0, 1}
    assert len(findings) == 6  # all three swapped values flagged on each row


def test_attribution_skips_rows_without_label_tokens(tmp_path):
    # No label columns named -> nothing to attribute -> no findings.
    swapped = write(tmp_path, "swapped.csv", ATTR_SWAPPED)
    findings, checked = vp.check_attribution(
        swapped, _groups(), label_columns=[], value_columns=ATTR_VALUES
    )
    assert findings == []
    assert checked == 0


def test_attribution_main_exit_codes(tmp_path):
    src = write(tmp_path, "src.md", ATTR_SOURCE)
    good = write(tmp_path, "good.csv", ATTR_GOOD)
    swapped = write(tmp_path, "swapped.csv", ATTR_SWAPPED)
    args = ["--attribution", "--label-columns", "Parameter type,Specimen",
            "--value-columns", "Parameter value,Lower bound,Upper bound"]
    assert vp.main(["prog", good, src] + args) == 0
    assert vp.main(["prog", swapped, src] + args) == 1
    # --attribution without --label-columns is a usage error
    assert vp.main(["prog", good, src, "--attribution"]) == 2
