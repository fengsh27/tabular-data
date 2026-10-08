"""pk-individual param type/unit extraction: a reply missing whole fields must not
bypass the retry-then-fallback loop.

PMID 33253437: for a sub-table whose "Parameter type" column holds the same string
on every row, the model got stuck repeating that string ~630 times and never wrote
"parameter_units"/"parameter_values" at all - identically on every retry (temperature
0). Those two fields were required, so Pydantic raised before a result object even
existed: post_process_validate_matched_tuple and try_fix_error_param_type_unit never
ran, and the exception propagated straight out, dropping the whole table with no
retry and no fallback. The fields are now optional (default []), so a reply like that
parses into a real (if invalid) result, fails post_process's length check instead,
and reaches the fallback after the retries are exhausted - which now reads the three
lists from the sub-table itself.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.common_agent.common_agent_ollama import CommonAgentOllama
from extractor.agents.pk_individual.pk_ind_param_type_unit_extract_agent import (
    ExtractedParamTypeUnits,
    ParamTypeUnitExtractionResult,
    _unit_from_type,
    post_process_validate_matched_tuple,
    try_fix_error_param_type_unit,
)

# shaped like the 33253437 sub-table: "Parameter type" repeats the original column
# name on every row, "Parameter value" holds the real per-row number
TYPE_STR = "Mother's PL III trimester (ng/ml)"
SUB_TABLE = dataframe_to_markdown(
    pd.DataFrame(
        [["1", TYPE_STR, "19.5"], ["3", TYPE_STR, "14.4"], ["4", TYPE_STR, "25.7"],
         ["5", TYPE_STR, "NA"], ["12", TYPE_STR, "16.7"]],
        columns=["ID", "Parameter type", "Parameter value"],
    )
)
COL_MAPPING = {"ID": "Patient ID", "Parameter type": "Parameter type", "Parameter value": "Parameter value"}


def test_a_reply_missing_two_of_three_fields_still_parses():
    # the exact failure shape: only "parameter_types" in the completion
    res = ExtractedParamTypeUnits(parameter_types=[TYPE_STR] * 629)
    assert res.parameter_units == []
    assert res.parameter_values == []
    ParamTypeUnitExtractionResult(extracted_param_units=res)  # does not raise


def test_a_reply_with_no_fields_at_all_still_parses():
    ExtractedParamTypeUnits()


def test_the_now_valid_but_wrong_length_result_is_sent_back_for_a_retry():
    res = ParamTypeUnitExtractionResult(
        extracted_param_units=ExtractedParamTypeUnits(parameter_types=[TYPE_STR] * 629)
    )
    with pytest.raises(RetryException) as e:
        post_process_validate_matched_tuple(res, SUB_TABLE, COL_MAPPING)
    assert "5 rows" in str(e.value)  # expected row count named, so the model can self-correct


def test_unit_from_type_extracts_the_trailing_parenthetical():
    assert _unit_from_type("Mother's PL III trimester (ng/ml)") == "ng/ml"
    assert _unit_from_type("Cmax (ug/L)") == "ug/L"
    assert _unit_from_type("No unit here") == "N/A"


def test_fallback_grounds_all_three_lists_in_the_sub_table_not_the_model():
    # the model's own lists are exactly the degenerate 33253437 shape: 629 copies
    # of the type string, nothing else
    res = ParamTypeUnitExtractionResult(
        extracted_param_units=ExtractedParamTypeUnits(parameter_types=[TYPE_STR] * 629)
    )
    types, units, values = try_fix_error_param_type_unit(res, SUB_TABLE, COL_MAPPING)
    assert types == [TYPE_STR] * 5
    assert units == ["ng/ml"] * 5
    assert values == ["19.5", "14.4", "25.7", "NA", "16.7"]  # the real per-row values, not N/A


def test_fallback_ignores_the_models_reply_even_when_it_is_merely_mismatched():
    # a more ordinary off-by-one case: the fallback still grounds in the table,
    # not in whatever the model said
    res = ParamTypeUnitExtractionResult(
        extracted_param_units=ExtractedParamTypeUnits(
            parameter_types=["wrong"] * 4, parameter_units=["wrong"] * 6, parameter_values=["wrong"] * 4
        )
    )
    types, units, values = try_fix_error_param_type_unit(res, SUB_TABLE, COL_MAPPING)
    assert types == [TYPE_STR] * 5
    assert values == ["19.5", "14.4", "25.7", "NA", "16.7"]


class _StubLLM:
    """Always answers with the 33253437 failure shape: only parameter_types, repeated."""

    def __init__(self, n_repeats=629):
        self.n_repeats = n_repeats
        self.calls = 0

    def bind(self, **_):
        return self

    def invoke(self, _messages):
        self.calls += 1
        content = json.dumps({"extracted_param_units": {"parameter_types": [TYPE_STR] * self.n_repeats}})
        return SimpleNamespace(
            content=content,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _go(monkeypatch, llm, **overrides):
    monkeypatch.setattr(CommonAgentOllama._invoke_agent.retry, "sleep", lambda _s: None)
    kwargs = dict(
        # no schema_basemodel: the real step (pk_ind_param_type_unit_extract_step.py)
        # doesn't pass one either, so PydanticOutputParser is built on `schema` directly.
        system_prompt="extract", instruction_prompt="answer",
        schema=ParamTypeUnitExtractionResult,
        post_process=post_process_validate_matched_tuple, md_table=SUB_TABLE, col_mapping=COL_MAPPING,
        try_fix_error=try_fix_error_param_type_unit,
    )
    kwargs.update(overrides)
    return CommonAgentOllama(llm=llm).go(**kwargs)


def test_end_to_end_the_degenerate_reply_now_reaches_the_fallback_on_the_fifth_attempt(monkeypatch):
    llm = _StubLLM()
    _res, processed, *_ = _go(monkeypatch, llm)
    assert llm.calls == 5  # four sent back for a retry, the fifth triggers the fallback
    types, units, values = processed
    assert types == [TYPE_STR] * 5
    assert units == ["ng/ml"] * 5
    assert values == ["19.5", "14.4", "25.7", "NA", "16.7"]


def test_end_to_end_without_the_fallback_the_table_is_lost_as_before(monkeypatch):
    llm = _StubLLM()
    with pytest.raises(Exception):
        _go(monkeypatch, llm, try_fix_error=None)
