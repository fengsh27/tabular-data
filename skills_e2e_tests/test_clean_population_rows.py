"""Deterministic regression test for the pk-population-individual cleanup script.

`clean_population_individual_rows.py` ports the one active rule of the legacy
`pk_popu_ind_row_cleanup_step` — drop rows whose characteristic value is blank or
N/A, with the exact slash-normalization the legacy code uses. Deterministic, so
asserted byte-exactly in CI.

Run:  poetry run pytest skills_e2e_tests/test_clean_population_rows.py
"""
import importlib.util
import os

SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "skills",
    "pk-pe-curation", "curation-common",
    "scripts",
    "clean_population_individual_rows.py",
)


def load_script():
    spec = importlib.util.spec_from_file_location("clean_population_individual_rows", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = load_script()

COLS = [
    "Patient ID", "Characteristic", "Characteristic subcategory",
    "Characteristic unit", "Characteristic value", "Population",
    "Pregnancy stage", "Pediatric/Gestational age", "Note",
]


def row(value, pid="1"):
    base = {c: "N/A" for c in COLS}
    base["Patient ID"] = pid
    base["Characteristic"] = "Weight"
    base["Characteristic value"] = value
    return base


def test_drops_na_value_rows():
    rows = [row("76.8", pid="1"), row("N/A", pid="2")]
    out = cp.clean_rows(rows)
    assert len(out) == 1 and out[0]["Patient ID"] == "1"


def test_drops_blank_and_whitespace_value_rows():
    rows = [row("23", pid="1"), row("", pid="2"), row("   ", pid="3")]
    out = cp.clean_rows(rows)
    assert [r["Patient ID"] for r in out] == ["1"]


def test_drops_na_variants_case_insensitive():
    for v in ["na", "n/a", "N/A", "Na", "n / a"]:
        assert cp.clean_rows([row(v)]) == [], f"{v!r} should be dropped"


def test_keeps_real_value_and_preserves_it_verbatim():
    out = cp.clean_rows([row("4 / 5 / 4")])  # a real ratio, not empty
    assert len(out) == 1
    assert out[0]["Characteristic value"] == "4 / 5 / 4"  # written through unchanged


def test_preserves_order():
    rows = [row("10", "1"), row("N/A", "2"), row("20", "3")]
    out = cp.clean_rows(rows)
    assert [r["Patient ID"] for r in out] == ["1", "3"]


def test_tolerant_when_value_column_absent():
    rows = [{"Patient ID": "1", "Characteristic": "Weight"}]  # no value column
    assert cp.clean_rows(rows) == rows
