"""Tests for extractor.curate_agent.llm_verification (ValidityChecker's Tier 3).

Stubs `get_agent`/`agent.go` the same way tests/test_cross_pipeline_retry_and_defaults.py
does for other PKPECommonStep subclasses, so these run with no live LLM.
"""

import json
import logging

import pytest

from extractor.curate_agent.llm_verification import (
    CurateAgentEmptyTableStep,
    CurateAgentVerificationResult,
    CurateAgentVerificationStep,
    EmptyTableAssessment,
    ErrorEntry,
    _count_errors,
    applicability_rules_for,
    assess_empty_table,
    completeness_rules_for,
    verify,
)

CURATED_TABLE = "| Value | SD |\n| --- | --- |\n| 91 | 30 |"


class _StubAgent:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def go(self, **kwargs):
        self.calls.append(kwargs)
        usage = {"total_tokens": 1, "completion_tokens": 1, "prompt_tokens": 0}
        return self.response, None, usage, "reasoning"


def test_count_errors_counts_distinct_idx_rows_not_cells():
    # idx 0 has two wrong columns - that's one error ROW, not two.
    res = CurateAgentVerificationResult(
        correct=False,
        error_rows=[
            ErrorEntry(idx=0, column="Value", found_value="91", expected_value="19"),
            ErrorEntry(idx=0, column="SD", found_value="30", expected_value="40"),
        ],
        explanation="placeholder",
    )
    error_rows, missed_rows, _ = _count_errors(res, CURATED_TABLE)
    assert error_rows == 1
    assert missed_rows == 0


def test_count_errors_counts_missing_rows_separately():
    res = CurateAgentVerificationResult(
        correct=False,
        error_rows=[ErrorEntry(idx=0, column="Value", found_value="91", expected_value="19")],
        missing_rows=['Drug="X", Value="5"', 'Drug="Y", Value="7"'],
        explanation="placeholder",
    )
    error_rows, missed_rows, _ = _count_errors(res, CURATED_TABLE)
    assert error_rows == 1
    assert missed_rows == 2


def test_count_errors_counts_extra_rows_as_errors():
    res = CurateAgentVerificationResult(correct=False, extra_rows=[0], explanation="placeholder")
    error_rows, missed_rows, _ = _count_errors(res, CURATED_TABLE)
    assert error_rows == 1
    assert missed_rows == 0


def test_count_errors_all_correct_is_zero():
    res = CurateAgentVerificationResult(correct=True, explanation="All values match.")
    error_rows, missed_rows, explanation = _count_errors(res, CURATED_TABLE)
    assert (error_rows, missed_rows) == (0, 0)
    assert explanation == "All values match."


def test_count_errors_drops_noop_change():
    # found_value == expected_value is not a real error, even if the model
    # (incorrectly) listed it - mirrors the production step's own
    # _remove_noop_fixes guard, as a plain field comparison instead of regex.
    res = CurateAgentVerificationResult(
        correct=False,
        error_rows=[ErrorEntry(idx=0, column="Value", found_value="91", expected_value="91")],
        explanation="placeholder",
    )
    error_rows, missed_rows, explanation = _count_errors(res, CURATED_TABLE)
    assert (error_rows, missed_rows) == (0, 0)
    assert explanation == "All values match."


def test_count_errors_drops_stale_found_value():
    # The model claims idx 0's current "Value" is "19", but the curated
    # table actually has "91" there - a fabricated/stale claim about the
    # table's current state, dropped regardless of what fix it proposes.
    # idx 0's "SD" claim is accurate ("30" really is there) and survives.
    res = CurateAgentVerificationResult(
        correct=False,
        error_rows=[
            ErrorEntry(idx=0, column="Value", found_value="19", expected_value="90"),
            ErrorEntry(idx=0, column="SD", found_value="30", expected_value="40"),
        ],
        explanation="placeholder",
    )
    error_rows, missed_rows, explanation = _count_errors(res, CURATED_TABLE)
    assert error_rows == 1
    assert missed_rows == 0
    assert "SD" in explanation
    assert '"Value"' not in explanation


def test_count_errors_drops_out_of_range_idx():
    # idx 5 doesn't exist in a 1-row table - a hallucinated row position,
    # exactly the PMID 18426260 "idx 202 on a 94-row table" failure shape.
    res = CurateAgentVerificationResult(
        correct=False,
        error_rows=[ErrorEntry(idx=5, column="Value", found_value="91", expected_value="19")],
        extra_rows=[9],
        explanation="placeholder",
    )
    error_rows, missed_rows, explanation = _count_errors(res, CURATED_TABLE)
    assert (error_rows, missed_rows) == (0, 0)
    assert explanation == "All values match."


def test_verify_wrapper_counts_from_stubbed_agent(monkeypatch):
    stub_agent = _StubAgent(CurateAgentVerificationResult(
        correct=False,
        error_rows=[ErrorEntry(idx=0, column="Value", found_value="91", expected_value="19")],
        missing_rows=['Drug="Y", Value="7"'],
        explanation="placeholder",
    ))
    monkeypatch.setattr(CurateAgentVerificationStep, "get_agent", lambda self, llm: stub_agent)

    result = verify(
        paper_title="A PK study", paper_abstract="An abstract.",
        source_tables="Table 1: ...", curated_table=CURATED_TABLE,
        llm="unused-placeholder",
    )

    assert result["correct"] is False
    assert result["error_row_count"] == 1
    assert result["missed_row_count"] == 1
    assert len(stub_agent.calls) == 1


def test_verify_wrapper_reports_zero_when_correct(monkeypatch):
    stub_agent = _StubAgent(CurateAgentVerificationResult(
        correct=True, explanation="All values match.",
    ))
    monkeypatch.setattr(CurateAgentVerificationStep, "get_agent", lambda self, llm: stub_agent)

    result = verify(
        paper_title="A PK study", paper_abstract="An abstract.",
        source_tables="Table 1: ...", curated_table=CURATED_TABLE,
        llm="unused-placeholder",
    )

    assert result["correct"] is True
    assert result["error_row_count"] == 0
    assert result["missed_row_count"] == 0


def test_completeness_rules_depend_on_schema():
    # Per-subject rules for individual-level schemas (and when unknown - the
    # original behaviour); summary-level rules otherwise. PMID 16143486: the
    # per-subject rules made Tier 3 report Table 4's 32 individual
    # measurements as missing from a pk_summary table whose gold has only
    # that table's 4 mean rows.
    assert "subject by subject" in completeness_rules_for("pk_individual")
    assert "subject by subject" in completeness_rules_for(None)
    summary = completeness_rules_for("pk_summary")
    assert "do NOT list them as missing rows" in summary
    assert "subject by subject" not in summary


def test_prompt_uses_summary_rules_for_pk_summary(monkeypatch):
    stub_agent = _StubAgent(CurateAgentVerificationResult(correct=True, explanation="All values match."))
    monkeypatch.setattr(CurateAgentVerificationStep, "get_agent", lambda self, llm: stub_agent)

    verify(paper_title="A PK study", paper_abstract="An abstract.",
           source_tables="Table 1: ...", curated_table=CURATED_TABLE,
           llm="unused-placeholder", schema="pk_summary")

    prompt = stub_agent.calls[0]["system_prompt"]
    assert "{completeness_rules}" not in prompt
    assert "summary statistics" in prompt
    assert "subject by subject" not in prompt


def test_prompt_uses_individual_rules_for_pk_individual(monkeypatch):
    stub_agent = _StubAgent(CurateAgentVerificationResult(correct=True, explanation="All values match."))
    monkeypatch.setattr(CurateAgentVerificationStep, "get_agent", lambda self, llm: stub_agent)

    verify(paper_title="A PK study", paper_abstract="An abstract.",
           source_tables="Table 1: ...", curated_table=CURATED_TABLE,
           llm="unused-placeholder", schema="pk_individual")

    assert "subject by subject" in stub_agent.calls[0]["system_prompt"]


@pytest.fixture
def propagating_extractor_logger():
    """Let records from `extractor.*` loggers reach caplog's root handler.

    `extractor/log_utils.py`'s `initialize_logger()` sets `propagate=False`
    on the "extractor" logger, and `app_script_pmids` calls it at import time -
    so once `tests/test_prepare_data_by_pmids_csv_file.py` has been collected
    in the same session, caplog is blind to every `extractor.*` record unless
    propagation is switched back on for the duration of the test.
    """
    loggers = [logging.getLogger("extractor"), logging.getLogger("extractor.curate_agent.llm_verification")]
    saved = [(lg, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.propagate = True
    try:
        yield
    finally:
        for lg, value in saved:
            lg.propagate = value


def test_verify_logs_full_structured_output_via_standard_logging(caplog, monkeypatch, propagating_extractor_logger):
    # The whole point of this log: the full error_rows/missing_rows/
    # explanation that check_validity()'s own tier3 dict throws away (see
    # _count_errors()'s docstring) must survive somewhere, not just the two
    # counts. It goes through this module's own `logger` - the same
    # "extractor.*" logging hierarchy extractor/log_utils.py's
    # initialize_logger() already wires to a file for every real entrypoint
    # (app.py/app_script.py/app_script_pmids.py/benchmark/conftest.py) - so
    # no bespoke file-writing or env var is needed here; this test only
    # has to confirm the record actually goes through `logging`.
    stub_agent = _StubAgent(CurateAgentVerificationResult(
        correct=False,
        error_rows=[ErrorEntry(idx=0, column="Value", found_value="91", expected_value="19")],
        missing_rows=['Drug="Y", Value="7"'],
        explanation="placeholder",
    ))
    monkeypatch.setattr(CurateAgentVerificationStep, "get_agent", lambda self, llm: stub_agent)

    with caplog.at_level(logging.INFO, logger="extractor.curate_agent.llm_verification"):
        result = verify(
            paper_title="A PK study", paper_abstract="An abstract.",
            source_tables="Table 1: ...", curated_table=CURATED_TABLE,
            llm="unused-placeholder",
        )

    records = [r for r in caplog.records if r.message.startswith("TIER3_LLM_CALL")]
    assert len(records) == 1
    payload = json.loads(records[0].message[len("TIER3_LLM_CALL "):])
    assert payload["paper_title"] == "A PK study"
    assert payload["curated_table"] == CURATED_TABLE
    assert payload["llm_output"]["error_rows"][0]["expected_value"] == "19"
    assert payload["llm_output"]["missing_rows"] == ['Drug="Y", Value="7"']
    assert payload["counted"] == {"error_row_count": 1, "missed_row_count": 1}
    assert result["error_row_count"] == 1  # verify()'s own return value is unaffected


# --- empty-table assessment (Tier 4 audit fix 2a) ---------------------------

def test_applicability_rules_depend_on_schema():
    ind = applicability_rules_for("pk_individual")
    summ = applicability_rules_for("pk_summary")
    assert "individual" in ind and "other published studies" in ind
    assert "group statistics" in ind
    assert "summary statistics" in summ and "mean / median / summary row" in summ
    assert applicability_rules_for(None) == ind


def test_assess_empty_table_no_data_returns_justified(monkeypatch):
    stub_agent = _StubAgent(EmptyTableAssessment(
        has_curatable_data=False, expected_rows=["should be ignored"],
        explanation="rows are group means with no patient identifiers"))
    monkeypatch.setattr(CurateAgentEmptyTableStep, "get_agent", lambda self, llm: stub_agent)

    res = assess_empty_table(
        paper_title="T", paper_abstract="A",
        source_tables="| Drug | 20 weeks |\n| --- | --- |\n| S-citalopram | 0.7 ± 0.1 |",
        schema="pk_individual", llm="unused",
    )

    assert res["has_curatable_data"] is False
    assert res["expected_rows"] == []  # not trusted when the verdict is "no data"
    assert "group means" in res["explanation"]
    prompt = stub_agent.calls[0]["system_prompt"]
    assert "individual-level pharmacokinetic table" in prompt
    assert "other published studies" in prompt
    assert "| S-citalopram | 0.7 ± 0.1 |" in prompt


def test_assess_empty_table_with_data_lists_expected_rows(monkeypatch):
    stub_agent = _StubAgent(EmptyTableAssessment(
        has_curatable_data=True, expected_rows=["Case 1 cord blood 36.89", "Case 5 cord blood 756.5"],
        explanation="one column per case"))
    monkeypatch.setattr(CurateAgentEmptyTableStep, "get_agent", lambda self, llm: stub_agent)

    res = assess_empty_table(paper_title="T", paper_abstract="A", source_tables="| x |",
                             schema="pk_summary", llm="unused")

    assert res["has_curatable_data"] is True
    assert res["expected_rows"] == ["Case 1 cord blood 36.89", "Case 5 cord blood 756.5"]
    assert "summary-level pharmacokinetic table" in stub_agent.calls[0]["system_prompt"]


def test_assess_empty_table_logs_its_output(monkeypatch, caplog, propagating_extractor_logger):
    stub_agent = _StubAgent(EmptyTableAssessment(
        has_curatable_data=False, expected_rows=[], explanation="literature cases"))
    monkeypatch.setattr(CurateAgentEmptyTableStep, "get_agent", lambda self, llm: stub_agent)
    with caplog.at_level(logging.INFO, logger="extractor.curate_agent.llm_verification"):
        assess_empty_table(paper_title="T", paper_abstract="A", source_tables="| x |",
                           schema="pk_individual", llm="unused")
    records = [r for r in caplog.records if r.getMessage().startswith("TIER3_EMPTY_TABLE_CALL ")]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage()[len("TIER3_EMPTY_TABLE_CALL "):])
    assert payload["schema"] == "pk_individual"
    assert payload["llm_output"]["has_curatable_data"] is False
    assert payload["source_tables"] == "| x |"
