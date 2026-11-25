"""Tests for the generic simple-prompt schemas (scripts/simple_prompt/schema_specs.py,
_common_generic.py, schema_registry.py) and their Tier 1 scoring in validity.py.

The two hand-written schemas (pk_individual, pk_summary) are covered by
test_simple_prompt_parsers.py; here the eight data-described schemas get the
same checks: the column sets match what the production workflows emit (and
benchmark/configs.py scores), a header in any accepted spelling maps, a row
short or long a field is realigned, a headerless block parses, "N/A" cleans
to "", and a lost block is reported as dropped rather than silently empty.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))

import validity as v  # noqa: E402
from schema_registry import ALL_SCHEMAS, SPECS, UNIT_KINDS, load_schema_module  # noqa: E402
from benchmark import configs  # noqa: E402

GENERIC = sorted(SPECS)


def test_every_cascade_schema_has_a_module_and_a_unit_kind():
    assert set(ALL_SCHEMAS) == {
        "pk_summary", "pk_individual", "pk_drug_summary", "pk_drug_individual",
        "pk_specimen_summary", "pk_specimen_individual", "pk_population_summary",
        "pk_population_individual", "pe_study_info", "pe_study_outcome",
    }
    for schema in ALL_SCHEMAS:
        mod = load_schema_module(schema)
        assert list(mod.COLS) == v.schema_columns(schema)
        assert UNIT_KINDS[schema] in ("table", "text", "table_or_text")
    assert UNIT_KINDS["pe_study_outcome"] == "table"
    assert UNIT_KINDS["pk_drug_summary"] == "text"
    assert UNIT_KINDS["pk_population_summary"] == "table_or_text"


@pytest.mark.parametrize("schema, cols", [
    ("pk_drug_summary", configs.PK_DRUG_SUMMARY_COLUMNS_TYPE),
    ("pk_drug_individual", configs.PK_DRUG_INDIVIDUAL_COLUMNS_TYPE),
    ("pk_specimen_summary", configs.PK_SPECIMEN_SUMMARY_COLUMNS_TYPE),
    ("pk_specimen_individual", configs.PK_SPECIMEN_INDIVIDUAL_COLUMNS_TYPE),
    ("pe_study_info", configs.PE_STUDY_INFO_COLUMNS_TYPE),
])
def test_columns_match_the_benchmark_configs(schema, cols):
    assert list(SPECS[schema].cols) == list(cols)


def test_population_and_pe_outcome_columns_match_the_workflows():
    # benchmark/configs.py is stale for these two (9 columns with a Patient ID for the
    # summary, "Outcomes" for the outcome); the workflows' final rename is the contract.
    assert list(SPECS["pk_population_summary"].cols) == [
        "Characteristic", "Characteristic subcategory", "Characteristic unit", "Characteristic value",
        "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound",
        "Upper bound", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N", "Note"]
    assert list(SPECS["pe_study_outcome"].cols) == [
        "Characteristic", "Exposure", "Outcome", "Parameter unit", "Parameter statistic",
        "Parameter value", "Variation type", "Variation value", "Interval type", "Lower bound",
        "Upper bound", "P value"]


def _row(schema, **values):
    cols = SPECS[schema].cols
    return [values.get(c, "") for c in cols]


def test_header_aliases_and_na_cleaning():
    p = load_schema_module("pk_drug_summary")
    raw = ("Drug name,Dose,Unit,Frequency,Schedule,Route,Population,Pregnancy stage,"
           "Gestational age,Population N,Source text\n"
           "Buprenorphine,8,mg,Multiple,once daily,Sublingual,Maternal,Trimester 3,N/A,10,\"8 mg, daily\"\n")
    rows = p.parse_csv(raw)
    assert rows == [{"Drug/Metabolite name": "Buprenorphine", "Dose amount": "8", "Dose unit": "mg",
                     "Dose frequency": "Multiple", "Dose schedule": "once daily", "Dose route": "Sublingual",
                     "Population": "Maternal", "Pregnancy stage": "Trimester 3",
                     "Pediatric/Gestational age": "", "Subject N": "10", "Note": "8 mg, daily"}]


def test_trailing_comma_and_headerless_block_parse():
    p = load_schema_module("pk_specimen_summary")
    raw = ("Plasma,24,Maternal,Trimester 3,,12,0-24,Hour,note,\n"
           "Cord blood,12,Pediatric,Delivery,,12,0,Hour,note,\n")
    dropped = []
    rows = p.parse_csv(raw, dropped=dropped)
    assert [r["Specimen"] for r in rows] == ["Plasma", "Cord blood"] and dropped == []


def test_short_row_is_realigned_and_unfixable_row_is_dropped():
    p = load_schema_module("pk_population_individual")
    header = ",".join(p.COLS)
    short = "1,Age,,year,25,Maternal,Trimester 3,"  # one empty missing (Note)
    raw = "\n".join([header, short, "garbage line, with two fields"])
    dropped = []
    rows = p.parse_csv(raw, dropped=dropped)
    assert len(rows) == 1 and rows[0]["Characteristic value"] == "25" and rows[0]["Patient ID"] == "1"
    assert len(dropped) == 1


def test_lost_headerless_block_is_reported_as_dropped():
    p = load_schema_module("pe_study_outcome")
    raw = "a,b\nc,d\n"  # nothing of schema width
    dropped = []
    assert p.parse_csv(raw, dropped=dropped) == [] and len(dropped) == 2


def test_fenced_single_row_pe_study_info():
    p = load_schema_module("pe_study_info")
    raw = ("```csv\n" + ",".join(p.COLS) + "\n"
           "Pharmacoepidemiology,Maternal,Prospective cohort study,Trimester 3,Zidovudine,"
           "\"Hospital, Lima\",\"HIV-infected, >18 y\",\"Multiple pregnancy\",\"Anemia, neutropenia\",120\n```")
    rows = p.parse_csv(raw)
    assert len(rows) == 1 and rows[0]["Data source"] == "Hospital, Lima" and rows[0]["Subject N"] == "120"


def test_generic_tier1_scoring_uses_the_spec_roles():
    assert v._structural_fits("pk_drug_summary", "Subject N", "N/A") == 0
    assert v._structural_fits("pk_drug_summary", "Subject N", "twelve") == -3
    assert v._structural_fits("pk_drug_summary", "Drug/Metabolite name", "") == -1
    assert v._structural_fits("pk_drug_summary", "Drug/Metabolite name", "12") == -2
    assert v._structural_fits("pk_drug_summary", "Dose route", "8") == -2
    assert v._structural_fits("pk_drug_summary", "Population", "anything") == 0  # vocabulary: not scored
    assert v._structural_fits("pe_study_outcome", "Parameter value", "0.52") == 2
    row = {c: "" for c in SPECS["pe_study_outcome"].cols}
    row.update({"Outcome": "birth weight", "Parameter value": "3.1", "Parameter unit": "kg"})
    assert v.score_row("pe_study_outcome", row) == 2


def test_score_table_validity_on_a_generic_schema(tmp_path):
    p = load_schema_module("pk_specimen_individual")
    csv_path = tmp_path / "t.csv"
    p.write_csv(str(csv_path), [
        {"Patient ID": "1", "Specimen": "Plasma", "Sample N": "3", "Sample time": "0, 2, 4", "Time unit": "Hour"},
        {"Patient ID": "2", "Specimen": "12", "Sample N": "three"},  # shifted-looking row
    ])
    scored = v.score_table_validity(str(csv_path), "pk_specimen_individual")
    assert scored["total_rows"] == 2 and scored["row_scores"][0] > 0 and scored["row_scores"][1] < 0
