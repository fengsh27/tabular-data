"""Deterministic regression test for the shared specimen row-cleanup script.

`clean_specimen_rows.py` ports the legacy `RowCleanupStep` of both specimen
pipelines (remove-half-total-row + keep-max-Sample-N), so it is a fully
deterministic stage we can assert byte-exactly in CI — like
test_clean_individual_rows for the individual pipeline.

Run:  poetry run pytest skills_e2e_tests/test_clean_specimen_rows.py
"""
import importlib.util
import os

SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-pe-curation", "curation-common",
    "scripts",
    "clean_specimen_rows.py",
)


def load_script():
    spec = importlib.util.spec_from_file_location("clean_specimen_rows", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = load_script()

SUMMARY_COLS = [
    "Specimen", "Sample N", "Population", "Pregnancy stage",
    "Pediatric/Gestational age", "Population N", "Sample time", "Time unit",
    "Note",
]
INDIVIDUAL_COLS = [
    "Patient ID", "Specimen", "Sample N", "Population", "Pregnancy stage",
    "Pediatric/Gestational age", "Sample time", "Time unit", "Note",
]


def srow(sn, **kw):
    base = {c: "N/A" for c in SUMMARY_COLS}
    base["Sample N"] = sn
    base.update(kw)
    return base


def irow(sn, **kw):
    base = {c: "N/A" for c in INDIVIDUAL_COLS}
    base["Sample N"] = sn
    base.update(kw)
    return base


# --- remove_sum_row ---------------------------------------------------------

def test_remove_sum_row_drops_half_total():
    rows = [srow("16", Specimen="Urine", **{"Sample time": "T1"}),
            srow("18", Specimen="Urine", **{"Sample time": "T2"}),
            srow("34", Specimen="Urine", **{"Sample time": "Total"})]  # 16+18+34=68, 34==68/2
    out = cs.remove_sum_row(rows)
    assert [r["Sample N"] for r in out] == ["16", "18"]


def test_remove_sum_row_noop_when_no_half():
    rows = [srow("10"), srow("20"), srow("25")]  # total 55, no value == 27.5
    assert len(cs.remove_sum_row(rows)) == 3


def test_remove_sum_row_noop_when_non_integer():
    rows = [srow("10"), srow("N/A"), srow("20")]
    assert len(cs.remove_sum_row(rows)) == 3  # astype(int) would raise -> no-op


def test_remove_sum_row_ignores_zero():
    rows = [srow("0"), srow("0")]  # total 0; v!=0 guard prevents a drop
    assert len(cs.remove_sum_row(rows)) == 2


# --- filter_max_sample_n ----------------------------------------------------

def test_keeps_max_sample_n_for_duplicate_combo():
    rows = [srow("10", Specimen="Urine", **{"Sample time": "0"}),
            srow("20", Specimen="Urine", **{"Sample time": "0"})]
    out = cs.filter_max_sample_n(rows, SUMMARY_COLS)
    assert len(out) == 1 and out[0]["Sample N"] == "20"


def test_population_n_excluded_from_comparison():
    # rows identical except Population N (and Sample N) -> collapse to max Sample N
    rows = [srow("10", Specimen="Urine", **{"Population N": "5"}),
            srow("20", Specimen="Urine", **{"Population N": "8"})]
    out = cs.filter_max_sample_n(rows, SUMMARY_COLS)
    assert len(out) == 1 and out[0]["Sample N"] == "20"


def test_note_excluded_from_comparison():
    rows = [srow("10", Specimen="Urine", Note="sentence A"),
            srow("20", Specimen="Urine", Note="sentence B")]
    out = cs.filter_max_sample_n(rows, SUMMARY_COLS)
    assert len(out) == 1 and out[0]["Sample N"] == "20"


def test_distinct_specimens_not_merged():
    rows = [srow("10", Specimen="Urine"), srow("20", Specimen="Blood")]
    out = cs.filter_max_sample_n(rows, SUMMARY_COLS)
    assert len(out) == 2


def test_individual_distinct_patients_not_merged():
    # Patient ID participates in the comparison, so two patients stay separate
    rows = [irow("10", **{"Patient ID": "1", "Specimen": "Urine"}),
            irow("10", **{"Patient ID": "2", "Specimen": "Urine"})]
    out = cs.filter_max_sample_n(rows, INDIVIDUAL_COLS)
    assert len(out) == 2


def test_original_order_preserved_through_merge():
    rows = [srow("20", Specimen="X", **{"Sample time": "0"}),
            srow("5", Specimen="Y"),
            srow("10", Specimen="X", **{"Sample time": "0"})]  # merges with row 0
    out = cs.filter_max_sample_n(rows, SUMMARY_COLS)
    assert [r["Specimen"] for r in out] == ["X", "Y"]
    assert out[0]["Sample N"] == "20"


# --- clean_rows end-to-end --------------------------------------------------

def test_clean_rows_removes_total_then_dedupes():
    rows = [srow("16", Specimen="U", **{"Sample time": "T1"}),
            srow("16", Specimen="U", **{"Sample time": "T1"}),  # exact dup
            srow("32", Specimen="U", **{"Sample time": "Total"})]  # 16+16+32=64, 32==64/2
    out = cs.clean_rows(rows, SUMMARY_COLS)
    assert len(out) == 1 and out[0]["Sample N"] == "16"
