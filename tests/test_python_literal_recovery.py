"""Python-cased literals in a model reply, and the pk-summary data-deletion step.

pk-summary benchmark, qwen3.6: the data-deletion prompt's example was
`{"processed": True, "row_list": [index_0, ...], ...}`, and qwen3.6 copied its casing:
`{"processed": False, "row_list": null, "col_list": null}`. That is neither JSON nor
Python, failed identically on all five temperature-0 attempts, and the step's failure
dropped the whole paper (3 papers x the pipeline and multi-agent arms = 6 of 63 runs).
"""
import json
from types import SimpleNamespace

import pytest
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from extractor.agents.common_agent.common_agent_ollama import (
    CommonAgentOllama,
    normalize_python_literals,
    recover_answer_from_prose,
)
from extractor.agents.pk_summary import pk_sum_common_step
from extractor.agents.pk_summary.pk_sum_individual_data_del_agent import (
    INDIVIDUAL_DATA_DEL_PROMPT,
    IndividualDataDelResult,
    post_process_individual_del_result,
)
from extractor.agents.pk_summary.pk_sum_individual_data_del_step import IndividualDataDelStep

# the reply logged for PMIDs 17635501, 22050870 and 30825333 (verbatim)
REAL_REPLY = '{"processed": False, "row_list": null, "col_list": null}'
MD = "| Parameter | Mean |\n| --- | --- |\n| Cmax | 12.5 |\n| AUC | 120 |"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (REAL_REPLY, '{"processed": false, "row_list": null, "col_list": null}'),
        ('{"a": True, "b": [False, None]}', '{"a": true, "b": [false, null]}'),
        ('{"note": "None of the rows", "ok": True}', '{"note": "None of the rows", "ok": true}'),
        ('{"q": "say \\"False\\" twice", "ok": False}', '{"q": "say \\"False\\" twice", "ok": false}'),
        ('{"Truest": 1, "None_key": 2, "xNone": 3}', '{"Truest": 1, "None_key": 2, "xNone": 3}'),
        ('{"already": true, "json": null}', '{"already": true, "json": null}'),
    ],
)
def test_normalize_python_literals(raw, expected):
    assert normalize_python_literals(raw) == expected
    assert len(normalize_python_literals(raw)) == len(raw)  # offsets stay valid


def test_the_real_reply_is_not_parseable_as_it_stands():
    with pytest.raises(Exception):
        PydanticOutputParser(pydantic_object=IndividualDataDelResult).parse(REAL_REPLY)


def test_recover_reads_the_real_reply():
    res = recover_answer_from_prose(REAL_REPLY, IndividualDataDelResult)
    assert res == IndividualDataDelResult(processed=False, row_list=None, col_list=None)


def test_recover_reads_a_literal_object_after_prose_and_keeps_the_last_one():
    reply = (
        'Example: {"processed": True, "row_list": [0], "col_list": ["a"]}\n'
        "The table has no individual data, so:\n"
        '{"processed": False, "row_list": null, "col_list": null}'
    )
    res = recover_answer_from_prose(reply, IndividualDataDelResult)
    assert res.processed is False and res.row_list is None


def test_recover_still_returns_none_for_a_real_error():
    assert recover_answer_from_prose('{"processed": Maybe}', IndividualDataDelResult) is None
    assert recover_answer_from_prose("no json at all", IndividualDataDelResult) is None


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


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(CommonAgentOllama._invoke_agent.retry, "sleep", lambda _s: None)


def test_agent_accepts_the_real_reply_on_the_first_attempt():
    llm = _StubLLM(REAL_REPLY)
    _res, processed, *_ = CommonAgentOllama(llm=llm).go(
        system_prompt="delete",
        instruction_prompt="answer",
        schema=IndividualDataDelResult,
        post_process=post_process_individual_del_result,
        md_table=MD,
    )
    assert llm.calls == 1  # used to burn all five attempts
    assert processed == MD  # processed=False keeps the table


def test_prompt_example_is_json_not_python():
    text = INDIVIDUAL_DATA_DEL_PROMPT.format(processed_md_table=MD)
    assert '"processed": true' in text and '"processed": false' in text
    assert "True" not in text  # no Python-cased boolean left anywhere in the prompt
    example_lines = [ln for ln in text.splitlines() if ln.startswith('{"processed"')]
    assert len(example_lines) == 2
    for line in example_lines:
        json.loads(line)  # every example line is valid JSON


def _step_state(llm):
    return {"md_table": MD, "llm": llm, "previous_errors": None}


def test_step_keeps_the_table_when_the_model_never_produces_a_parseable_reply(monkeypatch):
    llm = _StubLLM('{"processed": Maybe, "row_list": null, "col_list": null}')
    monkeypatch.setattr(
        pk_sum_common_step, "get_common_agent", lambda llm: CommonAgentOllama(llm=llm)
    )
    res, processed, token_usage = IndividualDataDelStep().execute_directly(_step_state(llm))
    assert llm.calls == 5  # every attempt was made before giving up
    assert processed == MD and res.processed is False
    assert token_usage["total_tokens"] == 0


def test_step_result_is_used_when_the_model_answers(monkeypatch):
    llm = _StubLLM(json.dumps({"processed": True, "row_list": [0], "col_list": ["Parameter", "Mean"]}))
    monkeypatch.setattr(
        pk_sum_common_step, "get_common_agent", lambda llm: CommonAgentOllama(llm=llm)
    )
    _res, processed, _tok = IndividualDataDelStep().execute_directly(_step_state(llm))
    assert "Cmax" in processed and "AUC" not in processed  # the answer was applied
