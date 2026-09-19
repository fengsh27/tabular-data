"""pk-individual drug matching: last-attempt fix for a wrong-length answer.

The JSON-only prompt (no row-by-row working) lets qwen3.6 lose count on long tables:
on PMID 33253437 it returned 56 indices for 55 rows on all 5 attempts, so the table
and the whole paper were lost. Retries only say "wrong length", so they repeat it.
`try_fix_error_matched_drugs` trims/pads on the fifth attempt.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent_ollama import CommonAgentOllama
from extractor.agents.pk_individual.pk_ind_drug_matching_agent import (
    MatchedDrugResult,
    post_process_validate_matched_rows,
    try_fix_error_matched_drugs,
)


def _md(n_rows: int) -> str:
    return dataframe_to_markdown(pd.DataFrame({"Patient ID": [str(i) for i in range(n_rows)]}))


SUBTABLE1 = _md(55)  # 55 rows to match
SUBTABLE2 = _md(6)  # 6 drug rows -> valid indices 0-5


def _fix(indices, expected=55, drugs=6):
    return try_fix_error_matched_drugs(
        MatchedDrugResult(matched_row_indices=indices), _md(expected), _md(drugs)
    )


def test_too_long_is_trimmed():
    out = _fix([0] * 24 + [1] * 32)
    assert len(out) == 55 and out == ([0] * 24 + [1] * 32)[:55]


def test_too_short_is_padded_with_the_error_sentinel():
    out = _fix([0] * 24 + [1] * 30)
    assert len(out) == 55 and out[:54] == [0] * 24 + [1] * 30 and out[54] == -1


def test_exact_length_is_unchanged():
    assert _fix([0, 1, 2], expected=3) == [0, 1, 2]


def test_models_own_minus_one_becomes_zero_like_post_process():
    assert _fix([0, -1, 2], expected=3) == [0, 0, 2]


def test_out_of_range_becomes_the_error_sentinel():
    assert _fix([0, 9, 2, -7], expected=4) == [0, -1, 2, -1]


def test_fixed_list_indexes_a_frame_with_the_error_row():
    # what DrugMatchingAgentStep does: Subtable 2 plus an ERROR row, then iloc[list]
    df = pd.concat(
        [pd.DataFrame({"Drug name": ["A", "B"]}), pd.DataFrame({"Drug name": ["ERROR"]})],
        ignore_index=True,
    )
    out = df.iloc[_fix([0, 1, 0, 1, 5], expected=4, drugs=2)].reset_index(drop=True)
    assert list(out["Drug name"]) == ["A", "B", "A", "B"]
    out = df.iloc[_fix([0, 1], expected=4, drugs=2)].reset_index(drop=True)
    assert list(out["Drug name"]) == ["A", "B", "ERROR", "ERROR"]


class _StubLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def bind(self, **_):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(
            content=self.content,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def _go(monkeypatch, llm, **overrides):
    # no real waiting between tenacity attempts
    monkeypatch.setattr(CommonAgentOllama._invoke_agent.retry, "sleep", lambda _s: None)
    agent = CommonAgentOllama(llm=llm)
    kwargs = dict(
        system_prompt="match rows",
        instruction_prompt="answer",
        schema=MatchedDrugResult,
        post_process=post_process_validate_matched_rows,
        md_table1=SUBTABLE1,
        md_table2=SUBTABLE2,
    )
    kwargs.update(overrides)
    return agent.go(**kwargs)


def test_agent_applies_the_fixer_on_the_fifth_attempt(monkeypatch):
    reply = json.dumps({"matched_row_indices": [0] * 24 + [1] * 2 + [2] * 30})  # 56 for 55 rows
    llm = _StubLLM(reply)
    res, processed, _usage = _go(monkeypatch, llm, try_fix_error=try_fix_error_matched_drugs)[:3]
    assert llm.calls == 5  # four wrong-length answers are retried, the fifth is fixed
    assert len(processed) == 55


def test_without_the_fixer_the_same_replies_still_fail(monkeypatch):
    reply = json.dumps({"matched_row_indices": [0] * 56})
    with pytest.raises(Exception):
        _go(monkeypatch, _StubLLM(reply))
