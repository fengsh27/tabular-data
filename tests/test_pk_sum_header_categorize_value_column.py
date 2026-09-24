"""pk-summary header categorization: a table must not silently lose all its value columns.

pk-summary benchmark: for PMIDs 34183327 and 35465728 gpt-4o (and qwen3.6 on the second)
categorized the numeric data columns "Uncategorized"; gpt-5.4 and gemma4 called them
"Parameter value". The validator only counted keys and "Parameter type" columns, so the
mapping passed, SplitByColumnsStep built one sub-table per "Parameter value" column,
found none and returned [] without an error - every later step ran on nothing and the
paper was lost (6 whole papers, 12 of 63 runs lost at least one table).
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.common_agent.common_agent_ollama import CommonAgentOllama
from extractor.agents.pk_summary.pk_sum_header_categorize_agent import (
    HeaderCategorizeJsonSchema,
    HeaderCategorizeResult,
    find_unlabeled_value_columns,
    post_process_validate_categorized_result,
    try_fix_error_header_categories,
)
from extractor.agents.pk_summary.pk_sum_split_by_col_step import SplitByColumnsStep

# shaped like PMID 34183327's table (headers verbatim, cells synthetic)
DATA_COLS = ["Parameter", "First PK assessment (n=20)", "Second PK assessment (n=12)", "P value*"]
DATA_MD = dataframe_to_markdown(
    pd.DataFrame(
        [
            ["Cmax (ng/mL)", "12.5 (3.1)", "14.0 (2.8)", "0.04"],
            ["AUC (ng·h/mL)", "120 (30)", "135 (25)", "0.12"],
            ["Tmax (h)", "1.5 (1.0–2.0)", "2.0 (1.0–3.0)", "0.31"],
        ],
        columns=DATA_COLS,
    )
)
ALL_UNCATEGORIZED = {
    "Parameter": "Parameter type",
    "First PK assessment (n=20)": "Uncategorized",
    "Second PK assessment (n=12)": "Uncategorized",
    "P value*": "P value",
}
CORRECT = {**ALL_UNCATEGORIZED,
           "First PK assessment (n=20)": "Parameter value",
           "Second PK assessment (n=12)": "Parameter value"}

# shaped like PMID 30825333's abbreviation/equation table: no data, correctly no value column
FORMULA_MD = dataframe_to_markdown(
    pd.DataFrame(
        [
            ["Clearance", "CL", "CL = Dose / AUC"],
            ["Half-life", "t1/2", "t1/2 = 0.693 / k"],
            ["Volume", "Vd", "Vd = CL / k"],
        ],
        columns=["Parameter", "Abbreviation", "Equation"],
    )
)
FORMULA_MAPPING = {"Parameter": "Parameter type", "Abbreviation": "Uncategorized", "Equation": "Uncategorized"}


def _validate(mapping, md):
    return post_process_validate_categorized_result(HeaderCategorizeResult(categorized_headers=mapping), md)


def test_a_mapping_with_no_value_column_but_numeric_columns_is_sent_back_with_the_columns():
    with pytest.raises(RetryException) as e:
        _validate(ALL_UNCATEGORIZED, DATA_MD)
    msg = str(e.value)
    assert "First PK assessment (n=20)" in msg and "Second PK assessment (n=12)" in msg
    assert "P value*" not in msg  # already categorized, not a candidate


def test_a_correct_mapping_passes_unchanged():
    assert _validate(CORRECT, DATA_MD).categorized_headers == CORRECT


def test_a_table_of_abbreviations_and_equations_legitimately_has_no_value_column():
    # equation cells contain digits ("0.693") but are not numeric cells
    assert _validate(FORMULA_MAPPING, FORMULA_MD).categorized_headers == FORMULA_MAPPING


def test_a_subject_count_column_is_not_a_value_column():
    md = dataframe_to_markdown(pd.DataFrame([["Cohort A", "20"], ["Cohort B", "12"]], columns=["Group", "N"]))
    mapping = {"Group": "Parameter type", "N": "Uncategorized"}
    assert find_unlabeled_value_columns(mapping, md) == []
    assert _validate(mapping, md).categorized_headers == mapping


def test_cells_of_all_the_usual_numeric_shapes_count():
    md = dataframe_to_markdown(pd.DataFrame(
        [["a", "12.5 ± 3.1", "0.5–3.0", "45%", "<0.001", "1.2 [0.8-1.5]", "N/A"]],
        columns=["p", "c1", "c2", "c3", "c4", "c5", "c6"],
    ))
    mapping = {"p": "Parameter type", **{c: "Uncategorized" for c in ["c1", "c2", "c3", "c4", "c5", "c6"]}}
    # c6 has only "N/A": no evidence either way, so it is left alone
    assert find_unlabeled_value_columns(mapping, md) == ["c1", "c2", "c3", "c4", "c5"]


def test_existing_checks_are_unchanged():
    with pytest.raises(ValueError, match="Expected 4 columns"):
        _validate({"Parameter": "Parameter type"}, DATA_MD)
    with pytest.raises(ValueError, match="Expected 1 'Parameter type'"):
        _validate({**CORRECT, "Parameter": "Uncategorized"}, DATA_MD)


def test_the_dict_form_of_the_result_is_validated_too():
    with pytest.raises(RetryException):
        post_process_validate_categorized_result({"categorized_headers": json.dumps(ALL_UNCATEGORIZED)}, DATA_MD)


def test_fixer_labels_only_the_numeric_columns():
    fixed = try_fix_error_header_categories(HeaderCategorizeResult(categorized_headers=ALL_UNCATEGORIZED), DATA_MD)
    assert fixed.categorized_headers == CORRECT
    # ... and the fixed mapping now satisfies the validator
    assert _validate(fixed.categorized_headers, DATA_MD)


def test_fixer_returns_none_when_there_is_nothing_to_promote():
    assert try_fix_error_header_categories(HeaderCategorizeResult(categorized_headers=FORMULA_MAPPING), FORMULA_MD) is None


def _sub_tables(col_mapping, md=DATA_MD):
    state = {"col_mapping": dict(col_mapping)}
    SplitByColumnsStep().leave_step(state, None, processed_res=[md], token_usage=None)
    return state["md_table_list"]


def test_regression_no_value_column_used_to_empty_the_sub_tables_silently():
    assert _sub_tables(ALL_UNCATEGORIZED) == []  # the failure mode: nothing, and no error
    assert len(_sub_tables(CORRECT)) == 2  # one sub-table per value column


class _StubLLM:
    def __init__(self, content):
        self.content, self.calls = content, 0

    def bind(self, **_):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(
            content=self.content,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _go(monkeypatch, llm, **overrides):
    monkeypatch.setattr(CommonAgentOllama._invoke_agent.retry, "sleep", lambda _s: None)
    kwargs = dict(
        system_prompt="categorize", instruction_prompt="answer",
        schema=HeaderCategorizeJsonSchema, schema_basemodel=HeaderCategorizeResult,
        post_process=post_process_validate_categorized_result, md_table_aligned=DATA_MD,
    )
    kwargs.update(overrides)
    return CommonAgentOllama(llm=llm).go(**kwargs)


def test_agent_retries_then_applies_the_fixer_on_the_fifth_attempt(monkeypatch):
    llm = _StubLLM(json.dumps({"categorized_headers": ALL_UNCATEGORIZED}))
    _res, processed, *_ = _go(monkeypatch, llm, try_fix_error=try_fix_error_header_categories)
    assert llm.calls == 5  # four sent back with the column names, the fifth fixed
    assert processed.categorized_headers == CORRECT


def test_agent_accepts_a_correct_answer_first_time(monkeypatch):
    llm = _StubLLM(json.dumps({"categorized_headers": CORRECT}))
    _res, processed, *_ = _go(monkeypatch, llm, try_fix_error=try_fix_error_header_categories)
    assert llm.calls == 1 and processed.categorized_headers == CORRECT


def test_a_formula_table_is_not_retried_or_promoted(monkeypatch):
    llm = _StubLLM(json.dumps({"categorized_headers": FORMULA_MAPPING}))
    _res, processed, *_ = _go(
        monkeypatch, llm, try_fix_error=try_fix_error_header_categories, md_table_aligned=FORMULA_MD
    )
    assert llm.calls == 1 and processed.categorized_headers == FORMULA_MAPPING


def test_without_the_fixer_the_same_replies_still_fail(monkeypatch):
    with pytest.raises(Exception):
        _go(monkeypatch, _StubLLM(json.dumps({"categorized_headers": ALL_UNCATEGORIZED})))
