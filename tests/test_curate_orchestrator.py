"""Tests for extractor.curate_agent.curate_orchestrator (the v2 per-table cascade).

Exercises the cascade's own logic - route/select wiring, per-table gate,
per-table escalation, judge-always, combine order - with every LLM-backed
step (route, selection, both arms, Tier 3, judge) swapped for a fake, since
the real ones need a live Ollama server and a multi-agent run. Tier 1 and
Tier 2 run for real on the fixture CSVs: a column-shifted row is what
trips the gate here (the exact bug shape from docs/SIMPLE_MODE_QWEN38_ISSUES.md).
"""

import csv
import json
import os

import pandas as pd
import pytest

from extractor.curate_agent import curate_orchestrator as co
from extractor.database.pmid_db import PMIDDB

PMID = "99999999"

TABLE_1 = pd.DataFrame({
    "Drug": ["Ketamine"], "Parameter": ["Total area under curve"],
    "N": [19], "Value": [91], "Unit": ["Mg/min/L"], "SD": [30],
})
TABLE_2 = pd.DataFrame({
    "Drug": ["Ketamine"], "Parameter": ["Clearance"],
    "N": [12], "Value": [55], "Unit": ["L/h"], "SD": [7],
})

PK_SUMMARY_HEADER = [
    "Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage",
    "Subject N", "Parameter type", "Value", "Unit", "Summary statistics",
    "Variation type", "Variation value", "Interval type", "Lower limit",
    "High limit", "P value",
]

CLEAN_ROW_1 = ["Ketamine", "Ketamine", "Plasma", "children", "", "19",
               "Total area under curve", "91", "Mg/min/L", "mean", "SD", "30", "", "", "", ""]
CLEAN_ROW_2 = ["Ketamine", "Ketamine", "Plasma", "children", "", "12",
               "Clearance", "55", "L/h", "mean", "SD", "7", "", "", "", ""]
# Table 2's "Subject N" never written - every later field slid one column over.
SHIFTED_ROW_2 = ["Ketamine", "Ketamine", "", "children", "", "Clearance",
                 "55", "L/h", "mean", "SD", "7", "", "", "", "", ""]


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PK_SUMMARY_HEADER)
        for row in rows:
            w.writerow(row)


def _pmid_db(tmp_path):
    db = PMIDDB(tmp_path / "pmid_info.db")
    tables = [
        {"caption": "Table 1", "footnote": "", "table": TABLE_1.copy()},
        {"caption": "Table 2", "footnote": "", "table": TABLE_2.copy()},
    ]
    db.insert_pmid_info(PMID, "title", "abstract", "full text", tables, [])
    return db


def _clean_verify(**kwargs):
    return {"correct": True, "error_row_count": 0, "missed_row_count": 0,
            "explanation": "All values match."}


def _route_pk_summary(**kwargs):
    return {"paper_type": "PK", "schemas": ["pk_summary"]}


def _select_all(tables, schema, **kwargs):
    return list(range(len(tables)))


def _cheap_arm_writing(rows_by_table):
    """Fake Arm A: writes the given rows for table_1/table_2 (keyed by the
    source table's directory name, which is how the orchestrator hands the
    table over)."""
    calls = []

    def fake_cheap(pmid, schema, table_md_path, out_dir, **kw):
        tname = os.path.basename(os.path.dirname(table_md_path))  # "table_1" / "table_2"
        calls.append(tname)
        csv_path = os.path.join(out_dir, "arm_a.csv")
        _write_csv(csv_path, rows_by_table[tname])
        return {"csv_path": csv_path, "rows": len(rows_by_table[tname]), "dropped_rows": 0,
                "seconds": 0.0, "llm_calls": 1}
    fake_cheap.calls = calls
    return fake_cheap


def _expensive_arm_writing(rows_by_caption):
    calls = []

    def fake_expensive(pmid, schema, table, out_dir, **kw):
        calls.append(table["caption"])
        csv_path = os.path.join(out_dir, "arm_b.csv")
        _write_csv(csv_path, rows_by_caption[table["caption"]])
        return {"csv_path": csv_path, "rows": len(rows_by_caption[table["caption"]]),
                "seconds": 0.0, "error": None}
    fake_expensive.calls = calls
    return fake_expensive


def _never(name):
    def fn(*args, **kwargs):
        raise AssertionError(f"{name} should not have been called")
    return fn


def _judge(winner):
    calls = []

    def fake_compare(**kwargs):
        calls.append(kwargs)
        return {"winner": winner, "explanation": "fake verdict", "used_full_tables": False}
    fake_compare.calls = calls
    return fake_compare


def test_all_tables_pass_never_escalates(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [CLEAN_ROW_2]})

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    assert result["route"] == {"paper_type": "PK", "schemas": ["pk_summary"]}
    assert result["not_handled"] == []
    schema = result["schemas"]["pk_summary"]
    assert cheap.calls == ["table_1", "table_2"]
    assert [t["winner"] for t in schema["tables"]] == ["A", "A"]
    assert schema["escalated_table_indexes"] == []
    combined = pd.read_csv(schema["csv_path"])
    assert list(combined.columns) == PK_SUMMARY_HEADER
    assert combined["Parameter type"].tolist() == ["Total area under curve", "Clearance"]
    assert schema["llm_calls"] == {"select": 1, "arm_a": 2, "tier3_a": 2}
    assert os.path.exists(os.path.join(tmp_path / "work", "cascade_report.json"))


def test_only_the_failing_table_escalates_and_judge_decides(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    expensive = _expensive_arm_writing({"Table 2": [CLEAN_ROW_2]})
    judge = _judge("B")

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=judge,
    )

    schema = result["schemas"]["pk_summary"]
    assert expensive.calls == ["Table 2"]  # Arm B ran for the failing table only
    assert len(judge.calls) == 1  # and the judge ran because Arm B ran - no tie needed
    t1, t2 = schema["tables"]
    assert (t1["escalated"], t1["winner"]) == (False, "A")
    assert (t2["escalated"], t2["winner"]) == (True, "B")
    assert t2["report_a"]["tier1"]["dropped_rows"] == 1
    assert t2["report_b"]["error_rate"] == 0.0
    assert schema["escalated_table_indexes"] == [2]
    # Table 2's source table only was handed to the judge.
    assert "Clearance" in judge.calls[0]["source_tables"]
    assert "Total area under curve" not in judge.calls[0]["source_tables"]
    combined = pd.read_csv(schema["csv_path"])
    assert combined["Subject N"].tolist() == [19, 12]  # table 1 from A, table 2 from B, in order
    assert schema["llm_calls"] == {"select": 1, "arm_a": 2, "tier3_a": 2, "tier3_b": 1, "judge": 1}


def test_judge_can_keep_arm_a(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    expensive = _expensive_arm_writing({"Table 2": [CLEAN_ROW_2]})

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=_judge("A"),
    )

    t2 = result["schemas"]["pk_summary"]["tables"][1]
    assert t2["winner"] == "A"
    assert t2["csv_path"] == t2["csv_a"]
    combined = pd.read_csv(result["schemas"]["pk_summary"]["csv_path"])
    assert len(combined) == 2
    assert combined.iloc[1]["Subject N"] == "Clearance"  # A's shifted row kept as-is, shift and all


def _empty_has_data(**kwargs):
    return {"has_curatable_data": True, "expected_rows": ["Ketamine Clearance 55 L/h"],
            "explanation": "a PK parameter row with n, value, unit"}


def _empty_no_data(**kwargs):
    return {"has_curatable_data": False, "expected_rows": [],
            "explanation": "rows are group means with no patient identifiers"}


def test_justified_empty_arm_a_wins_without_arm_b_or_judge(tmp_path):
    # Tier 4 audit cases 18426260 t2 / 34746508 t3: Arm A returns nothing for
    # a table that has no data of this schema's kind. The gate accepts it; no
    # Arm B, no judge; the combined table carries only the other table's rows.
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": []})

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"), empty_fn=_empty_no_data,
    )

    sch = result["schemas"]["pk_summary"]
    t2 = sch["tables"][1]
    assert t2["escalated"] is False and t2["winner"] == "A"
    assert t2["empty_table_justified"] is True
    assert t2["report_a"]["error_rate"] == 0.0
    assert t2["llm_calls"] == {"arm_a": 1, "tier3_a": 1}
    assert sch["escalated_table_indexes"] == []
    combined = pd.read_csv(sch["csv_path"])
    assert len(combined) == 1 and combined.iloc[0]["Parameter type"] == "Total area under curve"
    assert sch["tables"][0]["empty_table_justified"] is False


def test_unsupported_schema_is_reported_not_run(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [CLEAN_ROW_2]})

    def route_two(**kwargs):
        # every PipelineTypeEnum value is supported now; only an unknown name is not
        return {"paper_type": "Both", "schemas": ["pk_summary", "pk_something_new"]}

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=route_two, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    assert result["not_handled"] == ["pk_something_new"]
    assert set(result["schemas"]) == {"pk_summary"}


def test_explicit_schemas_skip_route(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [CLEAN_ROW_2]})

    result = co.curate(
        PMID, schemas=["pk_summary"], pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_never("route"), select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    assert result["route"] is None
    assert set(result["schemas"]) == {"pk_summary"}


def test_neither_paper_runs_nothing(tmp_path):
    def route_neither(**kwargs):
        return {"paper_type": "Neither", "schemas": []}

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=route_neither, select_tables_fn=_never("select"),
        run_cheap_arm_fn=_never("Arm A"), run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    assert result["schemas"] == {}
    assert result["not_handled"] == []


def test_selector_picks_a_subset_in_source_order(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [CLEAN_ROW_2]})

    def select_second_only(tables, schema, **kwargs):
        return [1]

    result = co.curate(
        PMID, schemas=["pk_summary"], pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        select_tables_fn=select_second_only,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    schema = result["schemas"]["pk_summary"]
    assert cheap.calls == ["table_2"]
    assert schema["selected_table_indexes"] == [2]
    assert pd.read_csv(schema["csv_path"])["Parameter type"].tolist() == ["Clearance"]


def test_cascade_report_is_json_serializable(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    expensive = _expensive_arm_writing({"Table 2": [CLEAN_ROW_2]})

    co.curate(
        PMID, schemas=["pk_summary"], pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=_judge("B"),
    )

    report = json.load(open(tmp_path / "work" / "cascade_report.json", encoding="utf-8"))
    assert report["schemas"]["pk_summary"]["escalated_table_indexes"] == [2]
    assert report["schemas"]["pk_summary"]["tables"][1]["judge"]["winner"] == "B"


def test_flatten_header_handles_db_round_tripped_multiindex_labels():
    # Exactly the labels PMID 11849190's table 1 carried after the pmid_info.db
    # JSON round-trip (string reprs of the MultiIndex tuples).
    assert co._flatten_header("('Unnamed: 0_level_0', 'Patient')") == "Patient"
    assert co._flatten_header("('Venlafaxine', 'Cmax1(µg l−1)')") == "Venlafaxine - Cmax1(µg l−1)"
    assert co._flatten_header("('Unnamed: 4_level_0', 'M/PAUC')") == "M/PAUC"
    assert co._flatten_header(("Venlafaxine", "Venlafaxine", "Cmax")) == "Venlafaxine - Cmax"  # colspan repeat
    assert co._flatten_header("Value") == "Value"  # plain labels untouched
    assert co._flatten_header("(not a tuple") == "(not a tuple"


def test_write_table_markdown_flattens_headers(tmp_path):
    df = pd.DataFrame([[1, 2.5]], columns=["('Unnamed: 0_level_0', 'Patient')", "('Venlafaxine', 'Cmax1(µg l−1)')"])
    (path,) = co.write_table_markdown([{"caption": "Table 1", "footnote": "", "table": df}], str(tmp_path))
    header = [l for l in open(path, encoding="utf-8") if l.startswith("|")][0]
    assert header.strip() == "| Patient | Venlafaxine - Cmax1(µg l−1) |"
    assert "Unnamed" not in header


def test_as_dataframes_decodes_first_call_json_tables():
    # extract_pmid_info_to_db()'s first call for a pmid returns the JSON string
    # insert_pmid_info() serialized in place, not a DataFrame.
    tables = [{"caption": "A", "table": TABLE_1.to_json()}, {"caption": "B", "table": TABLE_2}]
    out = co._as_dataframes(tables)
    assert isinstance(out[0]["table"], pd.DataFrame)
    assert out[0]["table"]["Value"].tolist() == [91]
    assert out[1]["table"] is TABLE_2


def test_positions_of_matches_by_identity_then_caption():
    tables = [{"caption": "A", "table": TABLE_1}, {"caption": "B", "table": TABLE_2}]
    assert co._positions_of(tables, [tables[1]]) == [1]
    copies = [{"caption": "B", "table": TABLE_2.copy()}]
    assert co._positions_of(tables, copies) == [1]
    assert co._positions_of(tables, [tables[1], tables[0]]) == [0, 1]


def test_select_tables_dispatches_the_selector_by_schema(monkeypatch):
    tables = [{"caption": "T1", "table": TABLE_1.copy()}, {"caption": "T2", "table": TABLE_2.copy()}]
    seen = []

    def fake_summary(html_tables, llm):
        seen.append("summary")
        return [html_tables[0]], ["0"], "summary reasoning", {}

    def fake_individual(html_tables, llm):
        seen.append("individual")
        return [html_tables[1]], ["1"], "individual reasoning", {}

    monkeypatch.setitem(co._SELECTORS, "pk_summary", fake_summary)
    monkeypatch.setitem(co._SELECTORS, "pk_individual", fake_individual)

    assert co.select_tables(tables, "pk_summary", llm="x") == [0]
    assert co.select_tables(tables, "pk_individual", llm="x") == [1]
    assert seen == ["summary", "individual"]


def _empty_report(justified):
    return {"tier1": {"total_rows": 0, "dropped_rows": 0, "avg_fit_score": 0.0}, "tier2": None,
            "tier3": {"correct": justified, "error_row_count": 0,
                      "missed_row_count": 0 if justified else 3, "explanation": "x",
                      "empty_table_justified": justified, "expected_rows": []},
            "error_rate": 0.0 if justified else 1.0, "escalate": not justified}


def _rows_report(n):
    return {"tier1": {"total_rows": n, "dropped_rows": 0, "avg_fit_score": 5.0},
            "tier2": {"checked_numbers": 4, "unsupported": 0, "findings": []},
            "tier3": {"correct": False, "error_row_count": 1, "missed_row_count": 2, "explanation": "y"},
            "error_rate": 0.3, "escalate": True}


def test_rule_verdict_unjustified_empty_loses_to_rows():
    v = co._rule_verdict(_rows_report(12), _empty_report(justified=False))
    assert v["winner"] == "A" and v["rule"] == "unjustified_empty_loses"
    v = co._rule_verdict(_empty_report(justified=False), _rows_report(5))
    assert v["winner"] == "B" and v["rule"] == "unjustified_empty_loses"


def test_rule_verdict_leaves_justified_empty_and_two_row_tables_to_the_judge():
    assert co._rule_verdict(_rows_report(12), _empty_report(justified=True)) is None
    assert co._rule_verdict(_rows_report(12), _rows_report(7)) is None
    assert co._rule_verdict(_empty_report(False), _empty_report(False))["rule"] == "both_empty"


def test_unjustified_empty_arm_b_loses_without_a_judge_call(tmp_path):
    # Second live round, 22050870 table 3: Arm A 12 rows, Arm B 0 rows with
    # the assessment saying the table has data - the LLM judge picked B.
    # Now decided by rule, no judge call, llm_calls["judge"] == 0.
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    expensive = _expensive_arm_writing({"Table 2": []})

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=_never("judge"), empty_fn=_empty_has_data,
    )

    t2 = result["schemas"]["pk_summary"]["tables"][1]
    assert t2["winner"] == "A" and t2["judge"]["rule"] == "unjustified_empty_loses"
    assert t2["llm_calls"]["judge"] == 0
    assert pd.read_csv(result["schemas"]["pk_summary"]["csv_path"])["Parameter type"].tolist() \
        == ["Total area under curve", "55"]  # A's (shifted) table 2 row is kept, not an empty B


def test_justified_empty_arm_b_forces_the_full_table_judge(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    expensive = _expensive_arm_writing({"Table 2": []})
    judge = _judge("B")

    co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=judge, empty_fn=_empty_no_data,
    )

    assert len(judge.calls) == 1
    assert judge.calls[0]["force_full_tables"] is True


def test_arm_b_mode_is_passed_to_arm_b_and_recorded(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    seen = []

    def expensive(pmid, schema, table, out_dir, **kw):
        seen.append(kw)
        csv_path = os.path.join(out_dir, "arm_b.csv")
        _write_csv(csv_path, [CLEAN_ROW_2])
        return {"csv_path": csv_path, "rows": 1, "seconds": 0.0, "error": None,
                "mode": kw["mode"], "final_answer": "Correct", "token_usage": {"total_tokens": 3}}

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=_judge("B"), arm_b_mode="multiagent",
    )

    assert seen[0]["mode"] == "multiagent"
    assert seen[0]["title"] and "abstract" in seen[0]  # the loop needs the paper's title/abstract
    schema = result["schemas"]["pk_summary"]
    assert schema["arm_b_mode"] == "multiagent"
    t2 = schema["tables"][1]
    assert t2["arm_b_mode"] == "multiagent" and t2["arm_b_final_answer"] == "Correct"
    assert "arm_b_mode" not in schema["tables"][0]  # not escalated: Arm B never ran


def test_default_arm_b_mode_is_pipeline(tmp_path):
    cheap = _cheap_arm_writing({"table_1": [CLEAN_ROW_1], "table_2": [SHIFTED_ROW_2]})
    seen = []

    def expensive(pmid, schema, table, out_dir, **kw):
        seen.append(kw)
        csv_path = os.path.join(out_dir, "arm_b.csv")
        _write_csv(csv_path, [CLEAN_ROW_2])
        return {"csv_path": csv_path, "rows": 1, "seconds": 0.0, "error": None}

    result = co.curate(
        PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
        route_fn=_route_pk_summary, select_tables_fn=_select_all,
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=expensive,
        verify_fn=_clean_verify, compare_fn=_judge("A"),
    )

    assert seen[0]["mode"] == "pipeline"
    schema = result["schemas"]["pk_summary"]
    assert schema["arm_b_mode"] == "pipeline"
    assert schema["tables"][1]["arm_b_mode"] == "pipeline"
    assert "arm_b_final_answer" not in schema["tables"][1]


def test_unknown_arm_b_mode_is_rejected_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="unknown Arm B mode"):
        co.curate(PMID, pmid_db=_pmid_db(tmp_path), work_dir=str(tmp_path / "work"),
                  route_fn=_never("route"), arm_b_mode="ensemble")


# --- units: full-text schemas and the population fallback ----------------------

DRUG_SUMMARY_HEADER = list(co.arms._validity.schema_columns("pk_drug_summary"))
DRUG_ROW = ["Buprenorphine", "8", "mg", "Multiple", "once daily", "Sublingual",
            "Maternal", "Trimester 3", "", "10", "Buprenorphine 8 mg once daily"]


def _write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _cheap_arm_recording(header, rows):
    """Fake Arm A that records the source file it was handed and writes `rows`."""
    calls = []

    def fake_cheap(pmid, schema, table_md_path, out_dir, **kw):
        calls.append((schema, table_md_path, open(table_md_path, encoding="utf-8").read()))
        csv_path = os.path.join(out_dir, "arm_a.csv")
        _write_rows(csv_path, header, rows)
        return {"csv_path": csv_path, "rows": len(rows), "dropped_rows": 0, "seconds": 0.0, "llm_calls": 1}
    fake_cheap.calls = calls
    return fake_cheap


def _pmid_db_with_sections(tmp_path, tables=None):
    db = PMIDDB(tmp_path / "pmid_info.db")
    sections = [{"section": "Methods", "content": "Ten (10) mothers (mean age 25.4 years, SD 0.5) "
                                                   "received buprenorphine 8 mg once daily."},
                {"section": "References", "content": "1. Someone et al."}]
    db.insert_pmid_info(PMID, "title", "abstract", "full text", tables or [], sections)
    return db


def test_full_text_schema_runs_one_text_unit_without_a_selector(tmp_path):
    cheap = _cheap_arm_recording(DRUG_SUMMARY_HEADER, [DRUG_ROW])

    result = co.curate(
        PMID, schemas=["pk_drug_summary"], pmid_db=_pmid_db_with_sections(tmp_path),
        work_dir=str(tmp_path / "work"), select_tables_fn=_never("select"),
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )

    sch = result["schemas"]["pk_drug_summary"]
    assert sch["unit_kind"] == "text"
    assert sch["selected_table_indexes"] == []
    assert sch["llm_calls"] == {"select": 0, "arm_a": 1, "tier3_a": 1}
    (unit,) = sch["tables"]
    assert (unit["unit_kind"], unit["table_index"], unit["winner"]) == ("text", 0, "A")
    # Arm A was handed the article text (sections joined, references cut), as a file
    schema_name, path, text = cheap.calls[0]
    assert schema_name == "pk_drug_summary" and path.endswith(os.path.join("source_text", "full_text.md"))
    assert "buprenorphine 8 mg once daily" in text and "Someone et al." not in text
    combined = pd.read_csv(sch["csv_path"])
    assert list(combined.columns) == DRUG_SUMMARY_HEADER
    assert combined["Drug/Metabolite name"].tolist() == ["Buprenorphine"]


def test_full_text_unit_escalates_with_a_text_payload(tmp_path):
    # Tier 2 fails: the dose 999 is nowhere in the text -> Arm B gets the text unit
    bad = list(DRUG_ROW); bad[1] = "999"
    cheap = _cheap_arm_recording(DRUG_SUMMARY_HEADER, [bad])
    seen = []

    def fake_expensive(pmid, schema, table, out_dir, **kw):
        seen.append(table)
        csv_path = os.path.join(out_dir, "arm_b.csv")
        _write_rows(csv_path, DRUG_SUMMARY_HEADER, [DRUG_ROW])
        return {"csv_path": csv_path, "rows": 1, "seconds": 0.0, "error": None}

    judge = _judge("B")
    result = co.curate(
        PMID, schemas=["pk_drug_summary"], pmid_db=_pmid_db_with_sections(tmp_path),
        work_dir=str(tmp_path / "work"), select_tables_fn=_never("select"),
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=fake_expensive,
        verify_fn=_clean_verify, compare_fn=judge,
    )

    sch = result["schemas"]["pk_drug_summary"]
    assert sch["escalated_table_indexes"] == [0]
    assert seen[0]["table"] is None and "buprenorphine" in seen[0]["text"].lower()
    assert "buprenorphine 8 mg once daily" in judge.calls[0]["source_tables"]
    assert judge.calls[0]["domain"] == "pharmacokinetic drug dosing (study-level)"
    assert pd.read_csv(sch["csv_path"])["Dose amount"].tolist() == [8]


def test_population_schema_uses_tables_when_selected_and_text_otherwise(tmp_path):
    header = list(co.arms._validity.schema_columns("pk_population_summary"))
    # Tier 2 checks the numeric cells against the unit's own source: the table
    # unit's row carries TABLE_1's numbers, the text unit's row the sections'.
    table_row = ["Age", "", "year", "91", "Mean", "SD", "30", "", "", "", "Maternal", "", "", "19", "x"]
    row = ["Age", "", "year", "25.4", "Mean", "SD", "0.5", "", "", "", "Maternal", "", "", "10", "x"]
    tables = [{"caption": "Table 1 Demographics", "footnote": "", "table": TABLE_1.copy()}]

    cheap = _cheap_arm_recording(header, [table_row])
    result = co.curate(
        PMID, schemas=["pk_population_summary"], pmid_db=_pmid_db_with_sections(tmp_path, tables),
        work_dir=str(tmp_path / "work"), select_tables_fn=lambda tables, schema, **kw: [0],
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )
    sch = result["schemas"]["pk_population_summary"]
    assert sch["selected_table_indexes"] == [1] and sch["tables"][0]["unit_kind"] == "table"
    assert "Total area under curve" in cheap.calls[0][2]  # the table's markdown, not the text

    cheap = _cheap_arm_recording(header, [row])
    # insert_pmid_info serialises the table DataFrames in place: hand the second db fresh copies
    tables = [{**t, "table": TABLE_1.copy()} for t in tables]
    (tmp_path / "b").mkdir()  # PMIDDB does not create the directory its file lives in
    result = co.curate(
        PMID, schemas=["pk_population_summary"], pmid_db=_pmid_db_with_sections(tmp_path / "b", tables),
        work_dir=str(tmp_path / "work_b"), select_tables_fn=lambda tables, schema, **kw: [],
        run_cheap_arm_fn=cheap, run_expensive_arm_fn=_never("Arm B"),
        verify_fn=_clean_verify, compare_fn=_never("judge"),
    )
    sch = result["schemas"]["pk_population_summary"]
    assert sch["selected_table_indexes"] == [] and sch["tables"][0]["unit_kind"] == "text"
    assert sch["llm_calls"]["select"] == 1  # the selector ran and found nothing
    assert "buprenorphine" in cheap.calls[0][2].lower()


def test_select_tables_dispatches_per_schema(monkeypatch):
    seen = {}

    def fake_selector(name):
        def fn(tables, llm):
            seen[name] = True
            return [tables[0]], [0], "because", {}
        return fn

    monkeypatch.setattr(co, "_SELECTORS", {
        "pk_summary": fake_selector("summary"), "pk_individual": fake_selector("individual"),
        "pe_study_outcome": fake_selector("pe"), "pk_population_summary": fake_selector("demo"),
        "pk_population_individual": fake_selector("demo"),
    })
    tables = [{"caption": "T1", "footnote": "", "table": TABLE_1.copy()}]
    for schema, key in (("pe_study_outcome", "pe"), ("pk_population_individual", "demo"),
                        ("pk_individual", "individual")):
        seen.clear()
        assert co.select_tables(tables, schema, llm=object()) == [0]
        assert seen == {key: True}


def test_article_text_of_prefers_sections_and_cuts_references():
    sections = [{"section": "Methods", "content": "<p>Given 8 mg.</p>"},
                {"section": "References", "content": "1. X"}]
    text = co.article_text_of("T", "A", sections, "full")
    assert "Given 8 mg." in text and "1. X" not in text and "<p>" not in text
    assert co.article_text_of("T", "A", [], "full") == "full"
    assert co.article_text_of("T", "A", None, None).startswith("T")
