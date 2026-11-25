"""Tests for extractor.curate_agent.arms - the per-table Arm A / Arm B wrappers.

The model call (Arm A) and the LangGraph workflow (Arm B) are stubbed; what is
under test is the CSV each arm leaves behind for the ValidityChecker: canonical
columns, dropped-row count, the multi-agent -> canonical column mapping, and
the "a failed table is an empty table, not an aborted paper" contract.
"""

import os

import pandas as pd

from extractor.curate_agent import arms

PK_SUMMARY_COLS = arms._validity.schema_columns("pk_summary")
PK_INDIVIDUAL_COLS = arms._validity.schema_columns("pk_individual")


def test_normalize_maps_multiagent_pk_summary_columns_to_canonical():
    df = pd.DataFrame([{
        "Drug name": "lorazepam", "Parameter type": "Cmax", "Parameter value": 12.96,
        "Parameter unit": "ng/ml", "Parameter statistic": "Mean", "Interval type": "Range",
        "Lower bound": 9.42, "Upper bound": 16.49,
        "Pediatric/Gestational age": None, "Time value": None, "Time unit": None,
    }])
    out = arms.normalize_to_schema(df, "pk_summary")
    assert list(out.columns) == PK_SUMMARY_COLS
    row = out.iloc[0]
    assert row["Value"] == 12.96 and row["Unit"] == "ng/ml"
    assert row["Summary statistics"] == "Mean"
    assert row["Lower limit"] == 9.42 and row["High limit"] == 16.49
    assert "Time value" not in out.columns


def test_normalize_pk_individual_keeps_canonical_columns():
    df = pd.DataFrame([{c: "x" for c in PK_INDIVIDUAL_COLS}])
    out = arms.normalize_to_schema(df, "pk_individual")
    assert list(out.columns) == PK_INDIVIDUAL_COLS


def test_cheap_arm_writes_canonical_csv_and_counts_dropped(tmp_path):
    table_md = tmp_path / "00_markdown_table.md"
    table_md.write_text("| Drug | Value |\n| --- | --- |\n| Ketamine | 91 |\n", encoding="utf-8")
    header = ",".join(PK_SUMMARY_COLS)
    good = ",".join(["Ketamine", "Ketamine", "Plasma", "children", "", "19",
                     "AUC", "91", "Mg/min/L", "mean", "SD", "30", "", "", "", ""])
    bad = "only,three,fields"

    def call_fn(prompt, **kw):
        assert "| Ketamine | 91 |" in prompt  # the table really went into the prompt
        return {"response": f"{header}\n{good}\n{bad}\n", "prompt_eval_count": 7, "eval_count": 3}

    res = arms.run_cheap_arm_table("123", "pk_summary", str(table_md), str(tmp_path / "out"),
                                   call_fn=call_fn)

    df = pd.read_csv(res["csv_path"])
    assert list(df.columns) == PK_SUMMARY_COLS
    assert len(df) == 1 and res["rows"] == 1
    assert res["dropped_rows"] == 1
    assert res["llm_calls"] == 1
    assert (tmp_path / "out" / "arm_a.raw.txt").exists()


class _FakeWorkflow:
    def __init__(self, df=None, exc=None):
        self.df, self.exc, self.calls = df, exc, []

    def go_md_table(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.df


def test_expensive_arm_pk_summary_writes_canonical_columns(tmp_path):
    wf = _FakeWorkflow(df=pd.DataFrame([{
        "Drug name": "lorazepam", "Parameter type": "Cmax", "Parameter value": 12.96,
        "Parameter unit": "ng/ml", "Parameter statistic": "Mean",
    }]))
    table = {"caption": "Table 1", "footnote": "fn", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("123", "pk_summary", table, str(tmp_path / "b"),
                                       title="T", full_text="ignored", workflow=wf)

    df = pd.read_csv(res["csv_path"])
    assert list(df.columns) == PK_SUMMARY_COLS
    assert df.iloc[0]["Value"] == 12.96 and df.iloc[0]["Unit"] == "ng/ml"
    assert res["rows"] == 1 and res["error"] is None
    assert wf.calls[0]["caption_and_footnote"] == "Table 1\nfn"
    assert "full_text" not in wf.calls[0]  # pk_summary's go_md_table has no such argument


def test_expensive_arm_pk_individual_passes_full_text(tmp_path):
    wf = _FakeWorkflow(df=pd.DataFrame([{c: "x" for c in PK_INDIVIDUAL_COLS}]))
    table = {"caption": "Table 1", "footnote": "", "table": pd.DataFrame({"a": [1]})}

    arms.run_expensive_arm_table("123", "pk_individual", table, str(tmp_path / "b"),
                                 title="T", full_text="the paper", workflow=wf)

    assert wf.calls[0]["full_text"] == "the paper"


def test_expensive_arm_failure_yields_empty_table_not_exception(tmp_path):
    wf = _FakeWorkflow(exc=ValueError("No valid entries found"))
    table = {"caption": "Table 1", "footnote": "", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("123", "pk_summary", table, str(tmp_path / "b"), workflow=wf)

    df = pd.read_csv(res["csv_path"])
    assert list(df.columns) == PK_SUMMARY_COLS and df.empty
    assert res["rows"] == 0
    assert "No valid entries found" in res["error"]


# --- Arm B modes: "pipeline" (execution only) vs "multiagent" (+ verify/correct loop) ---

def _ind_df(*values):
    return pd.DataFrame([{**{c: "x" for c in PK_INDIVIDUAL_COLS}, "Parameter value": v} for v in values])


def _loop_returning(df, final_answer="Correct"):
    calls = []

    def fake_loop(pmid, schema, exec_df, source_md, **kw):
        calls.append({"pmid": pmid, "schema": schema, "exec_df": exec_df, "source_md": source_md, **kw})
        return {"df": df, "final_answer": final_answer, "token_usage": {"total_tokens": 5}}
    fake_loop.calls = calls
    return fake_loop


def test_arm_b_rejects_unknown_mode(tmp_path):
    table = {"caption": "T", "footnote": "", "table": pd.DataFrame({"a": [1]})}
    import pytest
    with pytest.raises(ValueError, match="unknown Arm B mode"):
        arms.run_expensive_arm_table("1", "pk_individual", table, str(tmp_path), mode="ensemble",
                                     workflow=_FakeWorkflow(df=_ind_df("1")))


def test_arm_b_pipeline_mode_is_execution_only(tmp_path):
    wf = _FakeWorkflow(df=_ind_df("1", "2"))
    loop = _loop_returning(_ind_df("9"))
    table = {"caption": "T", "footnote": "", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("1", "pk_individual", table, str(tmp_path / "b"),
                                       workflow=wf, verify_correct_fn=loop)  # default mode

    assert res["mode"] == "pipeline" and res["final_answer"] is None and res["token_usage"] is None
    assert loop.calls == []  # the loop never ran
    assert pd.read_csv(res["csv_path"])["Parameter value"].tolist() == [1, 2]
    assert not os.path.exists(tmp_path / "b" / "arm_b.execution.csv")


def test_arm_b_multiagent_mode_runs_loop_on_execution_output(tmp_path):
    wf = _FakeWorkflow(df=_ind_df("1", "2"))
    loop = _loop_returning(_ind_df("1", "2", "3"), final_answer="Correct")
    table = {"caption": "Table 2", "footnote": "fn", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("77", "pk_individual", table, str(tmp_path / "b"),
                                       title="Title", abstract="Abstract", full_text="Body",
                                       mode="multiagent", workflow=wf, verify_correct_fn=loop)

    assert res["mode"] == "multiagent" and res["final_answer"] == "Correct"
    assert res["token_usage"] == {"total_tokens": 5} and res["error"] is None
    call = loop.calls[0]
    assert (call["pmid"], call["schema"]) == ("77", "pk_individual")
    assert call["exec_df"]["Parameter value"].tolist() == ["1", "2"]  # the loop saw execution's table
    assert "| a |" in call["source_md"]  # ... and the source table it came from
    assert (call["title"], call["abstract"], call["full_text"]) == ("Title", "Abstract", "Body")
    # the corrected table is what Arm B returns; execution's table is kept beside it
    assert res["rows"] == 3
    assert pd.read_csv(res["csv_path"])["Parameter value"].tolist() == [1, 2, 3]
    assert pd.read_csv(tmp_path / "b" / "arm_b.execution.csv")["Parameter value"].tolist() == [1, 2]


def test_arm_b_multiagent_mode_skips_loop_on_empty_execution(tmp_path):
    wf = _FakeWorkflow(df=pd.DataFrame())
    loop = _loop_returning(_ind_df("9"))
    table = {"caption": "T", "footnote": "", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("1", "pk_individual", table, str(tmp_path / "b"),
                                       mode="multiagent", workflow=wf, verify_correct_fn=loop)

    assert loop.calls == [] and res["rows"] == 0 and res["final_answer"] is None


def test_arm_b_multiagent_loop_failure_keeps_execution_result(tmp_path):
    wf = _FakeWorkflow(df=_ind_df("1", "2"))

    def broken_loop(*a, **kw):
        raise RuntimeError("verifier exploded")
    table = {"caption": "T", "footnote": "", "table": pd.DataFrame({"a": [1]})}

    res = arms.run_expensive_arm_table("1", "pk_individual", table, str(tmp_path / "b"),
                                       mode="multiagent", workflow=wf, verify_correct_fn=broken_loop)

    assert res["rows"] == 2 and res["final_answer"] == "VerificationError"
    assert "verifier exploded" in res["error"]
    assert pd.read_csv(res["csv_path"])["Parameter value"].tolist() == [1, 2]


def test_verify_correct_table_seeds_production_state_and_returns_corrected_df():
    class FakeStep:
        def __init__(self):
            self.calls = []

        def _run_one_table(self, state, table_md, source_md, table_label):
            self.calls.append((state, table_md, source_md, table_label))
            from extractor.agents.pk_pe_agents.pk_pe_agents_types import FinalAnswerEnum
            corrected = table_md.replace("| 2 |", "| 20 |")
            return corrected, FinalAnswerEnum.Correct, {"total_tokens": 42}

    step = FakeStep()
    df = pd.DataFrame({"Patient ID": ["p1"], "Parameter value": [2]})
    out = arms.verify_correct_table("55", "pk_individual", df, "| a |\n| --- |\n| 1 |",
                                    title="T", abstract="A", full_text="F", step=step)

    state, table_md, source_md, label = step.calls[0]
    assert state["pmid"] == "55" and state["paper_title"] == "T" and state["paper_abstract"] == "A"
    assert state["full_text"] == "F" and state["step_output_callback"] is None and state["step_count"] == 0
    assert "| p1 | 2 |" in table_md and source_md.startswith("| a |")
    assert out["final_answer"] == "Correct" and out["token_usage"] == {"total_tokens": 42}
    assert out["df"]["Parameter value"].astype(str).tolist() == ["20"]


def test_expensive_arm_placeholder_row_is_an_empty_table(tmp_path):
    # The full-text workflows answer "nothing found" with one all-N/A row (PMID
    # 18701886, pk_specimen_individual): that must reach the gate as zero rows.
    cols = arms._validity.schema_columns("pk_specimen_individual")
    placeholder = pd.DataFrame([{c: "N/A" for c in cols}])

    class _FullTextWorkflow:
        def go_full_text(self, **kwargs):
            self.kwargs = kwargs
            return placeholder

    wf = _FullTextWorkflow()
    unit = {"caption": "full text", "footnote": "", "table": None, "text": "Infant A gave meconium."}
    res = arms.run_expensive_arm_table("18701886", "pk_specimen_individual", unit, str(tmp_path / "b"),
                                       title="t", workflow=wf)
    assert res["rows"] == 0 and res["error"] is None
    assert wf.kwargs["full_text"] == "Infant A gave meconium."  # the text unit went to go_full_text
    assert list(pd.read_csv(res["csv_path"]).columns) == cols
    assert len(pd.read_csv(res["csv_path"])) == 0

    real = pd.DataFrame([{c: "N/A" for c in cols} | {"Patient ID": "A", "Specimen": "meconium"}])
    assert len(arms.drop_placeholder_rows(pd.concat([placeholder, real]))) == 1
