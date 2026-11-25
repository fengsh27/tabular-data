"""Phase 3 tests for extractor.curate_agent.comparison_judge.

Stubs `get_agent`/`agent.go` the same way tests/test_cross_pipeline_retry_and_defaults.py
does for other PKPECommonStep subclasses, so these run with no live LLM.
"""

from extractor.curate_agent.comparison_judge import (
    ComparisonFullResult,
    ComparisonJudgeStep,
    ComparisonReportsOnlyResult,
    _format_report,
    compare,
)

REPORT = {
    "tier1": {"total_rows": 5, "dropped_rows": 0, "avg_fit_score": 10.0},
    "tier2": {"checked_numbers": 8, "unsupported": 0, "findings": []},
    "tier3": {"correct": True, "error_row_count": 0, "missed_row_count": 0,
              "explanation": "All values match."},
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


EMPTY_JUSTIFIED_REPORT = {
    "tier1": {"total_rows": 0, "dropped_rows": 0, "avg_fit_score": 0.0},
    "tier2": None,
    "tier3": {"correct": True, "error_row_count": 0, "missed_row_count": 0,
              "explanation": "rows are group means with no patient identifiers",
              "empty_table_justified": True, "expected_rows": []},
    "error_rate": 0.0,
    "escalate": False,
}

EMPTY_FAILED_REPORT = {
    "tier1": {"total_rows": 0, "dropped_rows": 0, "avg_fit_score": 0.0},
    "tier2": None,
    "tier3": {"correct": False, "error_row_count": 0, "missed_row_count": 2,
              "explanation": "one column per case",
              "empty_table_justified": False,
              "expected_rows": ["Case 1 cord blood 36.89", "Case 5 cord blood 756.5"]},
    "error_rate": 1.0,
    "escalate": True,
}


def test_format_report_describes_a_justified_empty_table_and_hides_error_rate():
    text = _format_report(EMPTY_JUSTIFIED_REPORT)
    assert "total_rows: 0" in text
    assert "NO curatable data" in text
    assert "group means with no patient identifiers" in text
    assert "error_rate" not in text
    assert "fit score" not in text


def test_format_report_lists_expected_rows_for_a_failed_empty_table():
    text = _format_report(EMPTY_FAILED_REPORT)
    assert "DOES have curatable data" in text
    assert "Case 1 cord blood 36.89" in text and "Case 5 cord blood 756.5" in text


def test_format_report_for_a_table_with_rows_has_tiers_but_no_error_rate():
    text = _format_report(REPORT)
    assert "total_rows: 5" in text
    assert "existence check: 0 unsupported out of 8" in text
    assert "LLM verification: correct=True" in text
    assert "error_rate" not in text


def test_prompts_carry_the_empty_table_rule_and_no_tie_claim(monkeypatch):
    stub_agent = _StubAgent([
        ComparisonReportsOnlyResult(confident=False, winner=None, explanation="unsure"),
        ComparisonFullResult(winner="A", explanation="A's empty table is right"),
    ])
    step = ComparisonJudgeStep(llm=None)
    monkeypatch.setattr(step, "get_agent", lambda llm: stub_agent)

    state = _base_state()
    state["report_a"] = EMPTY_JUSTIFIED_REPORT
    step.execute(state)

    for call in stub_agent.calls:
        prompt = call["system_prompt"]
        assert "tied" not in prompt
        assert "zero rows is CORRECT" in prompt.replace("Zero rows", "zero rows") or "zero rows is CORRECT" in prompt
        assert "NO curatable data" in prompt


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


def test_force_full_tables_skips_the_reports_only_pass(monkeypatch):
    stub_agent = _StubAgent([ComparisonFullResult(winner="A", explanation="A's rows are real")])
    monkeypatch.setattr(ComparisonJudgeStep, "get_agent", lambda self, llm: stub_agent)

    result = compare(
        paper_title="A PK study", paper_abstract="An abstract.", source_tables="Table 1: ...",
        report_a=REPORT, table_a="| x |\n| --- |\n| 1 |",
        report_b=EMPTY_JUSTIFIED_REPORT, table_b="| x |\n| --- |",
        llm="unused", force_full_tables=True,
    )

    assert result == {"winner": "A", "explanation": "A's rows are real", "used_full_tables": True}
    assert len(stub_agent.calls) == 1
    assert "| 1 |" in stub_agent.calls[0]["system_prompt"]
