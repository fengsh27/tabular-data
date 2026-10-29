"""Phase 3 tests for extractor.curate_agent.comparison_judge.

Stubs `get_agent`/`agent.go` the same way tests/test_cross_pipeline_retry_and_defaults.py
does for other PKPECommonStep subclasses, so these run with no live LLM.
"""

from extractor.curate_agent.comparison_judge import (
    ComparisonFullResult,
    ComparisonJudgeStep,
    ComparisonReportsOnlyResult,
    compare,
)

REPORT = {
    "tier1": {"total_rows": 5, "dropped_rows": 0, "avg_fit_score": 10.0},
    "tier2": {"checked_numbers": 8, "unsupported": 0, "findings": []},
    "error_rate": 0.0,
    "escalate": False,
}


class _StubAgent:
    """Returns one queued response per `go()` call, recording each call's kwargs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def go(self, **kwargs):
        self.calls.append(kwargs)
        res = self.responses[len(self.calls) - 1]
        usage = {"total_tokens": 1, "completion_tokens": 1, "prompt_tokens": 0}
        return res, None, usage, "reasoning"


def _base_state(table_a="| x |\n| --- |\n| 1 |", table_b="| x |\n| --- |\n| 2 |"):
    return {
        "step_output_callback": None,
        "paper_title": "A PK study",
        "paper_abstract": "An abstract.",
        "source_tables": "Table 1: ...",
        "report_a": REPORT,
        "table_a": table_a,
        "report_b": REPORT,
        "table_b": table_b,
    }


def test_confident_from_reports_only_skips_full_tables(monkeypatch):
    stub_agent = _StubAgent([
        ComparisonReportsOnlyResult(confident=True, winner="A", explanation="A looks cleaner"),
    ])
    step = ComparisonJudgeStep(llm=None)
    monkeypatch.setattr(step, "get_agent", lambda llm: stub_agent)

    state = step.execute(_base_state())

    assert state["winner"] == "A"
    assert state["explanation"] == "A looks cleaner"
    assert state["used_full_tables"] is False
    assert len(stub_agent.calls) == 1


def test_not_confident_escalates_to_full_tables(monkeypatch):
    stub_agent = _StubAgent([
        ComparisonReportsOnlyResult(confident=False, winner=None, explanation="can't tell from reports alone"),
        ComparisonFullResult(winner="B", explanation="B's row matches the source exactly"),
    ])
    step = ComparisonJudgeStep(llm=None)
    monkeypatch.setattr(step, "get_agent", lambda llm: stub_agent)

    state = step.execute(_base_state(table_a="| x |\n| --- |\n| 1 |", table_b="| x |\n| --- |\n| 7 |"))

    assert state["winner"] == "B"
    assert state["explanation"] == "B's row matches the source exactly"
    assert state["used_full_tables"] is True
    assert len(stub_agent.calls) == 2
    # the second call's prompt must actually carry the curated tables, not
    # just the reports - that's the whole point of the second pass
    second_prompt = stub_agent.calls[1]["system_prompt"]
    assert "| 1 |" in second_prompt and "| 7 |" in second_prompt


def test_confident_winner_none_also_escalates(monkeypatch):
    # confident=True but winner missing would be a malformed response from
    # the model's own self-report - treat it the same as not confident
    # rather than crashing on a None winner.
    stub_agent = _StubAgent([
        ComparisonReportsOnlyResult(confident=True, winner=None, explanation="oops"),
        ComparisonFullResult(winner="A", explanation="fallback pick"),
    ])
    step = ComparisonJudgeStep(llm=None)
    monkeypatch.setattr(step, "get_agent", lambda llm: stub_agent)

    state = step.execute(_base_state())

    assert state["used_full_tables"] is True
    assert state["winner"] == "A"


def test_compare_wrapper_returns_plain_dict(monkeypatch):
    stub_agent = _StubAgent([
        ComparisonReportsOnlyResult(confident=True, winner="B", explanation="B wins"),
    ])
    monkeypatch.setattr(ComparisonJudgeStep, "get_agent", lambda self, llm: stub_agent)

    result = compare(
        paper_title="A PK study", paper_abstract="An abstract.",
        source_tables="Table 1: ...",
        report_a=REPORT, table_a="| x |\n| --- |\n| 1 |",
        report_b=REPORT, table_b="| x |\n| --- |\n| 2 |",
        llm="unused-placeholder",
    )

    assert result == {"winner": "B", "explanation": "B wins", "used_full_tables": False}
