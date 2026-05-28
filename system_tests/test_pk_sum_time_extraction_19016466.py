"""
System test for the Time Extraction failure observed on PMID 19016466.

Hypothesis under test (no mocks — uses the real `llm` fixture):

  When the Time Extraction step of the pk_summary pipeline is given the
  corrupted post-Row-Cleanup table seen in `output/tmp_3/logs/19016466.log`,
  the underlying Ollama call repeatedly fails with
      AttributeError: 'NoneType' object has no attribute 'strip'
  The @retry(stop_after_attempt(5)) wrapper around `_invoke_agent` retries
  5 times and finally raises `tenacity.RetryError` whose `.last_attempt`
  carries the AttributeError. Upstream callers convert this into
  Final Answer = NoTable.

To run:
    cd /users/PCON0100/feng1426/projects/github/tabular-data
    # Make sure an Ollama server is reachable as configured by
    # extractor.agents.agent_factory.get_pipeline_llm()
    pytest system_tests/test_pk_sum_time_extraction_19016466.py -s

This test exercises the real model. If a future model version stops
producing null/empty content for this input, the test will fail — that is
intentional: it documents the failure mode and detects regressions in
either direction (the bug being fixed, or the bug spreading).

Logging:
    Uses the project's `logging.getLogger(__name__)` — the logging
    configuration lives in `system_tests/conftest.py`. Both tests emit
    structured diagnostics so an operator can immediately see whether the
    failure reproduced exactly, drifted to a different failure mode, or
    whether the bug appears to be fixed (the test then fails with the
    recovered columns and sample rows logged at INFO level).
"""

import logging
import time
import traceback

import pandas as pd
import pytest
from tenacity import RetryError

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.pk_summary.pk_sum_time_unit_step import TimeExtractionStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState
from system_tests.conftest_data_19016466 import (
    data_caption_19016466,
    data_df_combined_19016466,
    data_md_table_aligned_19016466,
)


logger = logging.getLogger(__name__)

BANNER = "=" * 78
SEP = "-" * 78


@pytest.fixture(scope="module")
def caption_19016466():
    return data_caption_19016466


@pytest.fixture(scope="module")
def md_table_aligned_19016466():
    return data_md_table_aligned_19016466


@pytest.fixture(scope="module")
def df_combined_19016466():
    return markdown_to_dataframe(data_df_combined_19016466)


def _log_input_state(test_name, df_combined, md_table_aligned, caption):
    logger.info(BANNER)
    logger.info("[%s] INPUT STATE", test_name)
    logger.info(BANNER)
    logger.info("caption: %r", caption)
    logger.info(
        "md_table_aligned: %d lines, %d chars",
        len(md_table_aligned.splitlines()),
        len(md_table_aligned),
    )
    logger.info("df_combined: shape=%s", df_combined.shape)
    logger.info("df_combined columns (%d):", len(df_combined.columns))
    for col in df_combined.columns:
        logger.info("  - %r", col)
    logger.info(SEP)
    logger.info("df_combined head(5):\n%s", df_combined.head(5).to_string(index=False))
    logger.info(SEP)


def _log_exception_origin(underlying):
    """Walk the traceback and log file/line/func of every frame, plus a
    bold pointer to the innermost (deepest) frame where the exception was
    actually raised. This is the line of code that caused the failure."""
    tb = underlying.__traceback__
    if tb is None:
        logger.info("No traceback attached to exception — cannot locate origin.")
        return

    frames = traceback.extract_tb(tb)
    if not frames:
        logger.info("Traceback contained zero frames.")
        return

    logger.info("Call chain leading to %s (outermost → innermost):",
                type(underlying).__name__)
    for idx, fs in enumerate(frames):
        logger.info(
            "  [%d] %s:%d  in %s()    %s",
            idx,
            fs.filename,
            fs.lineno,
            fs.name,
            (fs.line or "").strip(),
        )

    origin = frames[-1]
    logger.info(SEP)
    logger.info(">>> ERROR ORIGIN <<<")
    logger.info("  file:     %s", origin.filename)
    logger.info("  line:     %d", origin.lineno)
    logger.info("  function: %s", origin.name)
    logger.info("  source:   %s", (origin.line or "").strip())
    logger.info(
        "  raised:   %s: %s", type(underlying).__name__, underlying
    )


def _log_retry_error_outcome(test_name, retry_error, elapsed_s):
    logger.info(BANNER)
    logger.info(
        "[%s] OUTCOME: RetryError raised (bug reproduces)", test_name
    )
    logger.info(BANNER)
    logger.info("elapsed: %.1f s", elapsed_s)
    last_attempt = retry_error.last_attempt
    logger.info("RetryError repr: %r", retry_error)
    logger.info("last_attempt.attempt_number = %s", last_attempt.attempt_number)
    underlying = last_attempt.exception()
    logger.info("last_attempt.exception() type: %s", type(underlying).__name__)
    logger.info("last_attempt.exception() repr: %r", underlying)
    logger.info("last_attempt.exception() str:  %s", underlying)
    logger.info(SEP)
    if underlying is not None:
        _log_exception_origin(underlying)
        logger.info(SEP)
        tb_text = "".join(
            traceback.format_exception(
                type(underlying), underlying, underlying.__traceback__
            )
        )
        logger.info("Full traceback of underlying exception:\n%s", tb_text)
    logger.info(SEP)


def _log_unexpected_success(test_name, state, original_columns, elapsed_s):
    logger.warning(BANNER)
    logger.warning(
        "[%s] OUTCOME: step.execute completed WITHOUT RetryError", test_name
    )
    logger.warning(BANNER)
    logger.warning("elapsed: %.1f s", elapsed_s)
    logger.warning(
        "The previously observed Time Extraction failure on PMID 19016466 "
        "no longer reproduces. Either the bug was fixed, the model changed, "
        "or the input fixture drifted."
    )
    logger.warning(SEP)
    df = state.get("df_combined")
    if isinstance(df, pd.DataFrame):
        new_cols = [c for c in df.columns if c not in original_columns]
        logger.warning("df_combined shape: %s", df.shape)
        logger.warning(
            "original columns (%d): %s", len(original_columns), original_columns
        )
        logger.warning("newly appended columns (%d): %s", len(new_cols), new_cols)
        for candidate in ("Time value", "Time unit"):
            if candidate in df.columns:
                preview = df[candidate].tolist()[:10]
                logger.warning("%r first 10 values: %s", candidate, preview)
        logger.warning(SEP)
        logger.warning(
            "df_combined head(10) after Time Extraction:\n%s",
            df.head(10).to_string(index=False),
        )
    else:
        logger.warning("df_combined is not a DataFrame: %s", type(df).__name__)
    logger.warning(SEP)


def _log_unexpected_exception(test_name, exc, elapsed_s):
    logger.error(BANNER)
    logger.error(
        "[%s] OUTCOME: step.execute raised UNEXPECTED exception", test_name
    )
    logger.error(BANNER)
    logger.error("elapsed: %.1f s", elapsed_s)
    logger.error("Exception type: %s", type(exc).__name__)
    logger.error("Exception repr: %r", exc)
    logger.error(SEP)
    _log_exception_origin(exc)
    logger.error(SEP)
    logger.error("Full traceback:", exc_info=exc)
    logger.error(SEP)


def test_TimeExtractionStep_19016466_raises_retry_error(
    llm,
    md_table_aligned_19016466,
    df_combined_19016466,
    caption_19016466,
    step_callback,
):
    """
    Run the real TimeExtractionStep on the verbatim corrupted state from
    paper 19016466 and assert the production failure mode reproduces:

      1) `step.execute(state)` raises `RetryError` (after 5 attempts).
      2) `last_attempt.exception()` is an AttributeError whose message is
         "'NoneType' object has no attribute 'strip'".
      3) The state's `df_combined` is left unchanged — no Time value /
         Time unit columns were appended.
    """
    test_name = "test_TimeExtractionStep_19016466_raises_retry_error"
    _log_input_state(
        test_name,
        df_combined_19016466,
        md_table_aligned_19016466,
        caption_19016466,
    )

    step = TimeExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["df_combined"] = df_combined_19016466
    state["md_table_aligned"] = md_table_aligned_19016466
    state["caption"] = caption_19016466
    state["step_callback"] = step_callback

    original_columns = list(df_combined_19016466.columns)

    logger.info(
        "[%s] Calling step.execute(state) — expecting RetryError after 5 "
        "retries (~20 min total).",
        test_name,
    )
    t0 = time.monotonic()

    raised = None
    try:
        step.execute(state)
    except RetryError as e:
        raised = e
    except BaseException as e:  # noqa: BLE001 — surface anything unexpected
        elapsed = time.monotonic() - t0
        _log_unexpected_exception(test_name, e, elapsed)
        raise

    elapsed = time.monotonic() - t0

    if raised is None:
        _log_unexpected_success(test_name, state, original_columns, elapsed)
        pytest.fail(
            "Expected RetryError from TimeExtractionStep on the corrupted "
            "PMID 19016466 input, but step.execute completed successfully. "
            "Inspect logs above for the recovered output — if the bug is "
            "genuinely fixed, this test should be updated or removed."
        )

    _log_retry_error_outcome(test_name, raised, elapsed)

    last_attempt = raised.last_attempt
    assert last_attempt.exception() is not None
    assert isinstance(last_attempt.exception(), AttributeError), (
        f"Expected AttributeError as the underlying cause, "
        f"got {type(last_attempt.exception()).__name__}: "
        f"{last_attempt.exception()}"
    )
    assert "'NoneType' object has no attribute 'strip'" in str(
        last_attempt.exception()
    ), (
        f"Expected the null-handling error message, "
        f"got: {last_attempt.exception()!r}"
    )

    # State is left unchanged — no Time value / Time unit columns appended.
    assert isinstance(state["df_combined"], pd.DataFrame)
    assert list(state["df_combined"].columns) == original_columns, (
        "df_combined should not have been mutated when Time Extraction "
        "failed; the new ['Time value', 'Time unit'] columns must not be "
        "present."
    )

    logger.info(
        "[%s] All assertions passed — bug reproduces as documented.",
        test_name,
    )


def test_TimeExtractionStep_19016466_retries_exactly_five_times(
    llm,
    md_table_aligned_19016466,
    df_combined_19016466,
    caption_19016466,
    step_callback,
):
    """
    Verify the @retry(stop_after_attempt(5)) policy by inspecting the
    `attempt_number` of the failing future returned in RetryError.

    Note: this test re-runs the (slow) end-to-end Ollama call. Mark the
    test as `slow` if you maintain a pytest marker layer.
    """
    test_name = "test_TimeExtractionStep_19016466_retries_exactly_five_times"
    _log_input_state(
        test_name,
        df_combined_19016466,
        md_table_aligned_19016466,
        caption_19016466,
    )

    step = TimeExtractionStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["df_combined"] = df_combined_19016466
    state["md_table_aligned"] = md_table_aligned_19016466
    state["caption"] = caption_19016466
    state["step_callback"] = step_callback

    original_columns = list(df_combined_19016466.columns)

    logger.info(
        "[%s] Calling step.execute(state) — verifying stop_after_attempt(5) "
        "policy.",
        test_name,
    )
    t0 = time.monotonic()

    raised = None
    try:
        step.execute(state)
    except RetryError as e:
        raised = e
    except BaseException as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        _log_unexpected_exception(test_name, e, elapsed)
        raise

    elapsed = time.monotonic() - t0

    if raised is None:
        _log_unexpected_success(test_name, state, original_columns, elapsed)
        pytest.fail(
            "Expected RetryError to verify retry count, but step.execute "
            "completed successfully. Bug may be fixed — see logs above for "
            "details."
        )

    _log_retry_error_outcome(test_name, raised, elapsed)

    last_attempt = raised.last_attempt
    assert last_attempt.attempt_number == 5, (
        f"Expected 5 retry attempts before RetryError; "
        f"got attempt_number={last_attempt.attempt_number}"
    )

    logger.info(
        "[%s] Confirmed 5 attempts before RetryError.", test_name
    )
