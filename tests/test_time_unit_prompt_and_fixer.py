"""Time/unit step: one entry per row (prompt) and a last-attempt fixer.

PMID 32153014 (arm B of the multi-agent benchmark): the first sub-table has 9 rows but
only 6 distinct (time, unit) pairs, because every patient sits on two lines. The prompt
said both "for each row ... exactly the same number of rows" and "List each *unique*
combination"; gpt-5.4 followed the second and returned the 6 distinct pairs on all five
attempts, the step raised a wrong-length error every time, and the paper's only table was
lost (gpt-4o ignores "unique", which is why it got through).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.common_agent.common_agent_ollama import CommonAgentOllama
from extractor.agents.pk_individual.pk_ind_time_unit_agent import (
    TimeAndUnitResult,
    post_process_time_and_unit,
    try_fix_error_time_and_unit,
)

AGENTS = Path(__file__).resolve().parent.parent / "extractor" / "agents"
TIME_UNIT_PROMPTS = [
    "pk_individual/pk_ind_time_unit_agent.py",
    "pk_summary/pk_sum_time_unit_agent.py",
    "pk_specimen_individual/pk_spec_ind_time_unit_agent.py",
    "pk_specimen_summary/pk_spec_sum_time_unit_agent.py",
]


@pytest.mark.parametrize("rel", TIME_UNIT_PROMPTS)
def test_prompt_asks_for_one_entry_per_row_not_unique_combinations(rel):
    text = (AGENTS / rel).read_text()
    assert "List each unique combination" not in text
    assert "exactly one entry for EVERY row" in text
    assert "NOT merged or deduplicated" in text


def _table(n_rows):
    return dataframe_to_markdown(pd.DataFrame({"Patient": [str(i) for i in range(n_rows)]}))


NINE = _table(9)
SIX_DISTINCT = [["23", "Day"], ["57", "Day"], ["31", "Day"], ["4", "Day"], ["29", "Day"], ["16", "Day"]]


def _fix(pairs, md=NINE):
    return markdown_to_dataframe(try_fix_error_time_and_unit(TimeAndUnitResult(times_and_units=pairs), md))


def test_short_answer_is_padded_with_na():
    df = _fix(SIX_DISTINCT)
    assert df.shape == (9, 2) and list(df.columns) == ["Time value", "Time unit"]
    assert df.iloc[:6].values.tolist() == SIX_DISTINCT
    assert df.iloc[6:].values.tolist() == [["N/A", "N/A"]] * 3


def test_long_answer_is_trimmed():
    pairs = [[str(i), "Day"] for i in range(12)]
    df = _fix(pairs)
    assert df.shape[0] == 9 and df.iloc[-1].tolist() == ["8", "Day"]


def test_entries_are_forced_to_two_cells():
    df = _fix([["1"], ["2", "Day", "src"]], md=_table(2))
    assert df.values.tolist() == [["1", "N/A"], ["2", "Day"]]


def test_correct_answer_matches_the_normal_post_process():
    pairs = [[str(i), "Day"] for i in range(9)]
    res = TimeAndUnitResult(times_and_units=pairs)
    assert try_fix_error_time_and_unit(res, NINE) == post_process_time_and_unit(res, NINE)


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
        system_prompt="times", instruction_prompt="answer", schema=TimeAndUnitResult,
        post_process=post_process_time_and_unit, md_table_post_processed=NINE,
    )
    kwargs.update(overrides)
    return CommonAgentOllama(llm=llm).go(**kwargs)


def test_agent_applies_the_fixer_on_the_fifth_wrong_length_answer(monkeypatch):
    llm = _StubLLM(json.dumps({"times_and_units": SIX_DISTINCT}))
    _res, processed, *_ = _go(monkeypatch, llm, try_fix_error=try_fix_error_time_and_unit)
    assert llm.calls == 5  # four wrong-length answers are retried, the fifth is fixed
    assert markdown_to_dataframe(processed).shape[0] == 9


def test_without_the_fixer_the_same_replies_still_fail(monkeypatch):
    with pytest.raises(Exception):
        _go(monkeypatch, _StubLLM(json.dumps({"times_and_units": SIX_DISTINCT})))
