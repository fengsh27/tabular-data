"""VerifyScopeEnum.PerTable: verify+correct each source table on its own, then combine,
instead of combining first and verifying the whole paper (VerifyScopeEnum.Combined, the
existing default - see pk_pe_agenttool_task.py for the graph shapes).

These are pure orchestration tests: PKPECuratedTablesVerificationStep and
PKPECuratedTablesCorrectionCodeStep are monkeypatched (their own prompts/parsing are
unchanged and already covered elsewhere), so what's under test here is
PKPEPerTableVerifyCorrectStep's looping (one scoped sub-state per table, capped rounds,
skip a table with no data), the paper-level verdict precedence across tables, table
combination, AgentTool.run()'s backward-compatible 2-tuple/3-tuple handling, and the
runtime fallback-to-Combined routing when a tool doesn't populate curated_tables.
"""
from types import SimpleNamespace

import pandas as pd
import pytest
from TabFuncFlow.utils.table_utils import dataframe_to_markdown

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import AgentTool
from extractor.agents.pk_pe_agents.pk_pe_agents_types import FinalAnswerEnum
from extractor.agents.pk_pe_agents.pk_pe_per_table_verify_step import (
    PKPEPerTableVerifyCorrectStep,
    combine_markdown_tables,
    worst_case_final_answer,
)
from extractor.agents_manager.pk_pe_agenttool_task import tool_supports_per_table
from extractor.constants import MAX_PER_TABLE_STEP_COUNT

# --------------------------------------------------------------------------
# worst_case_final_answer
# --------------------------------------------------------------------------


def test_worst_case_all_correct_is_correct():
    assert worst_case_final_answer([FinalAnswerEnum.Correct, FinalAnswerEnum.Correct]) == FinalAnswerEnum.Correct


def test_worst_case_one_incorrect_wins_over_correct():
    assert worst_case_final_answer([FinalAnswerEnum.Correct, FinalAnswerEnum.Incorrect]) == FinalAnswerEnum.Incorrect


def test_worst_case_error_outranks_incorrect():
    # a hard error further upstream is worse news than "the model checked and disagreed"
    assert worst_case_final_answer(
        [FinalAnswerEnum.Incorrect, FinalAnswerEnum.PipelineError]
    ) == FinalAnswerEnum.PipelineError


def test_worst_case_max_step_reached_outranks_incorrect():
    assert worst_case_final_answer(
        [FinalAnswerEnum.Correct, FinalAnswerEnum.MaxStepReached]
    ) == FinalAnswerEnum.MaxStepReached


def test_worst_case_empty_list_is_no_table():
    # every table produced nothing (all None, skipped) - same conceptual outcome as the
    # combined-mode execution_step producing no curated_table at all.
    assert worst_case_final_answer([]) == FinalAnswerEnum.NoTable


def test_worst_case_single_value_passthrough():
    assert worst_case_final_answer([FinalAnswerEnum.Unverified]) == FinalAnswerEnum.Unverified


# --------------------------------------------------------------------------
# combine_markdown_tables
# --------------------------------------------------------------------------


def _md(*rows):
    return dataframe_to_markdown(pd.DataFrame(rows))


def test_combine_concatenates_in_order():
    t1 = _md({"a": 1, "b": 2}, {"a": 3, "b": 4})
    t2 = _md({"a": 5, "b": 6})
    combined = combine_markdown_tables([t1, t2])
    from TabFuncFlow.utils.table_utils import markdown_to_dataframe
    df = markdown_to_dataframe(combined)
    # markdown round-trips numbers as strings; that's fine, only order/content matters here.
    assert df["a"].astype(str).tolist() == ["1", "3", "5"]
    assert df["b"].astype(str).tolist() == ["2", "4", "6"]


def test_combine_skips_blank_entries():
    t1 = _md({"a": 1})
    combined = combine_markdown_tables([t1, "", None])
    from TabFuncFlow.utils.table_utils import markdown_to_dataframe
    assert len(markdown_to_dataframe(combined)) == 1


def test_combine_all_blank_is_empty_table():
    assert combine_markdown_tables([None, ""]) is not None  # doesn't raise; empty df markdown


# --------------------------------------------------------------------------
# tool_supports_per_table (the runtime fallback-to-Combined check)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "curated_tables, expected",
    [
        (["t1", "t2"], True),
        (["t1", None], True),  # one real table is enough to take the per-table path
        ([None, None], False),  # every table failed extraction - nothing to scope to
        ([], False),
        (None, False),  # tool never populated it - the un-upgraded-pipeline case
    ],
)
def test_tool_supports_per_table(curated_tables, expected):
    assert tool_supports_per_table({"curated_tables": curated_tables}) is expected


def test_tool_supports_per_table_missing_key():
    assert tool_supports_per_table({}) is False


# --------------------------------------------------------------------------
# AgentTool.run(): backward-compatible 2-tuple / 3-tuple _run() results
# --------------------------------------------------------------------------


class _TwoTupleTool(AgentTool):
    def _run(self, previous_errors=None):
        return pd.DataFrame({"a": [1]}), ["source md"]


class _ThreeTupleTool(AgentTool):
    def _run(self, previous_errors=None):
        return pd.DataFrame({"a": [1]}), ["source md"], [pd.DataFrame({"a": [1]})]


def test_run_defaults_per_table_dfs_to_none_for_unupgraded_tool():
    df, source_tables, final_answer, per_table_dfs = _TwoTupleTool().run()
    assert final_answer is None
    assert per_table_dfs is None
    assert source_tables == ["source md"]


def test_run_passes_through_per_table_dfs_for_upgraded_tool():
    df, source_tables, final_answer, per_table_dfs = _ThreeTupleTool().run()
    assert final_answer is None
    assert per_table_dfs is not None and len(per_table_dfs) == 1


# --------------------------------------------------------------------------
# PKPEPerTableVerifyCorrectStep orchestration (verification/correction monkeypatched)
# --------------------------------------------------------------------------


def _base_state(curated_tables, source_tables):
    return {
        "pmid": "0000000",
        "paper_title": "t",
        "paper_abstract": "a",
        "full_text": "N/A",
        "curated_tables": curated_tables,
        "source_tables": source_tables,
        "final_answer": None,
        "step_output_callback": None,
    }


def _make_step(verify_fn, correct_fn=None):
    step = PKPEPerTableVerifyCorrectStep(llm=None, pmid="0000000", domain="test")
    step.verification_step._execute_directly = verify_fn
    if correct_fn is not None:
        step.correction_step._execute_directly = correct_fn
    return step


def test_per_table_step_verifies_each_table_independently():
    """Two tables, both correct on the first pass: no correction calls at all."""
    seen_source_tables = []

    def verify(sub_state):
        seen_source_tables.append(sub_state["source_tables"])
        sub_state["final_answer"] = FinalAnswerEnum.Correct
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    def correct(sub_state):
        raise AssertionError("correction should not run when verification says Correct")

    table_one, table_two = _md({"a": "one"}), _md({"a": "two"})
    step = _make_step(verify, correct)
    state = _base_state([table_one, table_two], ["source one", "source two"])
    state, tok = step._execute_directly(state)

    assert state["final_answer"] == FinalAnswerEnum.Correct
    assert seen_source_tables == [["source one"], ["source two"]]  # scoped, not the whole paper
    from TabFuncFlow.utils.table_utils import markdown_to_dataframe
    assert markdown_to_dataframe(state["curated_table"])["a"].tolist() == ["one", "two"]


def test_per_table_step_skips_tables_with_no_data():
    calls = []

    def verify(sub_state):
        calls.append(sub_state["curated_table"])
        sub_state["final_answer"] = FinalAnswerEnum.Correct
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    step = _make_step(verify)
    state = _base_state([None, "table two"], ["source one", "source two"])
    step._execute_directly(state)

    assert calls == ["table two"]  # the None (failed-extraction) table was never verified


def test_per_table_step_gives_each_table_its_own_correction_loop():
    """Table 1 needs one correction round; table 2 is correct immediately - the loops must
    not share state (e.g. table 2 must not inherit table 1's previous_verification_thoughts
    or its corrected content)."""
    verify_calls = {"table_one": 0, "table_two": 0}

    def verify(sub_state):
        key = "table_one" if sub_state["curated_table"].startswith("t1") else "table_two"
        verify_calls[key] += 1
        if key == "table_one" and verify_calls[key] == 1:
            sub_state["final_answer"] = FinalAnswerEnum.Incorrect
            assert sub_state["previous_verification_thoughts"] == []
        else:
            sub_state["final_answer"] = FinalAnswerEnum.Correct
            # table two must never see table one's corrected content or thoughts
            assert sub_state["curated_table"] in ("t1 v0", "t1 v1", "t2 v0")
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    def correct(sub_state):
        sub_state["curated_table"] = "t1 v1"
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    step = _make_step(verify, correct)
    state = _base_state(["t1 v0", "t2 v0"], ["s1", "s2"])
    step._execute_directly(state)

    assert verify_calls == {"table_one": 2, "table_two": 1}


def test_per_table_step_caps_rounds_at_max_per_table_step_count():
    def always_incorrect(sub_state):
        sub_state["final_answer"] = FinalAnswerEnum.Incorrect
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    def noop_correct(sub_state):
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    step = _make_step(always_incorrect, noop_correct)
    state = _base_state(["t1"], ["s1"])
    state, _ = step._execute_directly(state)

    assert state["final_answer"] == FinalAnswerEnum.MaxStepReached


def test_per_table_step_paper_verdict_is_worst_case_across_tables():
    def verify(sub_state):
        sub_state["final_answer"] = (
            FinalAnswerEnum.Correct if sub_state["curated_table"] == "good" else FinalAnswerEnum.Incorrect
        )
        return sub_state, {**DEFAULT_TOKEN_USAGE}

    def correct(sub_state):
        return sub_state, {**DEFAULT_TOKEN_USAGE}  # never actually fixes it

    step = _make_step(verify, correct)
    state = _base_state(["good", "bad"], ["s1", "s2"])
    state, _ = step._execute_directly(state)

    assert state["final_answer"] == FinalAnswerEnum.MaxStepReached  # "bad" exhausts its rounds


def test_per_table_step_short_circuits_if_already_terminal():
    def verify(sub_state):
        raise AssertionError("must not run when final_answer is already terminal")

    step = _make_step(verify)
    state = _base_state(["t1"], ["s1"])
    state["final_answer"] = FinalAnswerEnum.NoTable  # e.g. set by an earlier terminal step
    state, tok = step._execute_directly(state)

    assert state["final_answer"] == FinalAnswerEnum.NoTable
    assert tok == {**DEFAULT_TOKEN_USAGE}


def test_per_table_step_raises_if_curated_tables_missing():
    """The graph only routes here when tool_supports_per_table(state) is True; reaching
    this step with no curated_tables is a wiring bug, not a normal fallback - it should
    fail loudly rather than silently verify nothing."""
    step = _make_step(verify_fn=lambda s: (_ for _ in ()).throw(AssertionError("unreachable")))
    state = _base_state(None, ["s1"])
    with pytest.raises(ValueError):
        step._execute_directly(state)
