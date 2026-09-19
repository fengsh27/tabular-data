"""qwen3.6 via Ollama sometimes narrates step by step and only then writes the answer.

`format=<schema>` is not enforced in every think mode, so the reply is prose that
ends with the answer object. Neither the parser nor `handle_qwen_thinking`
(trims to the first bracket) finds it, and the identical temperature-0 retries
cannot recover. `recover_answer_from_prose` takes the LAST JSON object in the
reply that fits the schema.

`tests/data/qwen36_drug_matching_prose_reply_32153014.txt` is a real reply
(6585 chars, PMID 32153014, drug matching) that killed the paper's only table.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.prompts import ChatPromptTemplate

from extractor.agents.common_agent.common_agent_ollama import (
    CommonAgentOllama,
    recover_answer_from_prose,
)
from extractor.agents.pk_individual.pk_ind_drug_matching_agent import MatchedDrugResult
from extractor.agents.pk_individual.pk_ind_summary_data_del_agent import SummaryDataDelResult
from extractor.agents.pk_summary.pk_sum_drug_matching_agent import (
    MatchedDrugResult as SumMatchedDrugResult,
)

REAL_REPLY = (
    Path(__file__).parent / "data" / "qwen36_drug_matching_prose_reply_32153014.txt"
).read_text()


def test_real_reply_is_recovered():
    res = recover_answer_from_prose(REAL_REPLY, MatchedDrugResult)
    assert res.matched_row_indices == [0, 0, 0, 0, 0, 0, 3, 3, 3]


def test_placeholder_example_before_answer_is_ignored():
    reply = (
        'The format is {"matched_row_indices": [index_0, index_1, ..., index_n]}.\n'
        "Row 0 -> 0, row 1 -> 1.\n"
        '{"matched_row_indices": [0, 1]}'
    )
    assert recover_answer_from_prose(reply, MatchedDrugResult).matched_row_indices == [0, 1]


def test_last_valid_object_wins():
    reply = (
        'First guess: {"matched_row_indices": [0, 0]}\n'
        "Wait, row 1 is the second drug.\n"
        'Final: {"matched_row_indices": [0, 1]}'
    )
    assert recover_answer_from_prose(reply, MatchedDrugResult).matched_row_indices == [0, 1]


def test_object_that_does_not_fit_the_schema_is_skipped():
    reply = '{"matched_row_indices": [3, 4]}\nNote: {"comment": "unrelated"} done.'
    assert recover_answer_from_prose(reply, MatchedDrugResult).matched_row_indices == [3, 4]


def test_wrong_key_in_the_final_object_is_remapped():
    # the pk_summary prompt used to show `matching_row_indices`
    reply = 'Reasoning...\n{"matching_row_indices": [0, 2, 2]}'
    assert recover_answer_from_prose(reply, SumMatchedDrugResult).matched_row_indices == [0, 2, 2]


def test_exact_object_beats_a_later_wrong_key_object():
    reply = '{"matched_row_indices": [0, 1]}\nAs a note {"row_count": [2]}.'
    assert recover_answer_from_prose(reply, MatchedDrugResult).matched_row_indices == [0, 1]


def test_fenced_answer_after_prose():
    reply = 'Working it out.\n```json\n{"matched_row_indices": [1, 1]}\n```\n'
    assert recover_answer_from_prose(reply, MatchedDrugResult).matched_row_indices == [1, 1]


def test_nested_object_inside_answer_is_not_mistaken_for_it():
    reply = '{"processed": true, "row_list": [0, 1], "col_list": ["a", "b"]}'
    res = recover_answer_from_prose("Analysis first.\n" + reply, SummaryDataDelResult)
    assert res.processed is True and res.row_list == [0, 1]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "no json here at all",
        '{"matched_row_indices": ["x"]}',  # right key, wrong data: do not mask
        '{"other": ["x"]}',  # wrong key AND wrong data
        "Row 0 -> [0, 0, 0] (index) and that is all",  # a bare list in prose is not an answer
        '{"matched_row_indices": [0, 1',  # truncated before the object closed
    ],
)
def test_unrecoverable_replies_return_none(reply):
    assert recover_answer_from_prose(reply, MatchedDrugResult) is None


def test_non_model_schema_returns_none():
    assert recover_answer_from_prose('{"a": 1}', {"type": "object"}) is None


# --------------------------------------------------------------------------
# through the real get_runnable_agent, with a stub LLM (no server)
# --------------------------------------------------------------------------


class _StubLLM:
    """Stands in for ChatOllama: `.bind(format=...).invoke(msgs)` returns the reply."""

    def __init__(self, content):
        self.content = content

    def bind(self, **_):
        return self

    def invoke(self, _messages):
        return SimpleNamespace(
            content=self.content,
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


def _run(content, schema=MatchedDrugResult):
    prompt = ChatPromptTemplate.from_messages([("system", "{input}")])
    agent = CommonAgentOllama.get_runnable_agent(prompt, _StubLLM(content), schema)
    return agent.invoke({"input": "x"})


def test_runnable_agent_recovers_the_real_prose_reply():
    res, usage = _run(REAL_REPLY)
    assert res.matched_row_indices == [0, 0, 0, 0, 0, 0, 3, 3, 3]
    assert usage["total_tokens"] == 150  # token accounting is unaffected


def test_runnable_agent_still_raises_when_nothing_is_recoverable():
    with pytest.raises(Exception):
        _run("I could not determine the matching.")


def test_runnable_agent_plain_json_reply_is_unchanged():
    res, _ = _run('{"matched_row_indices": [2, 2]}')
    assert res.matched_row_indices == [2, 2]
