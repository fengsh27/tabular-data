"""pk-individual summary-data deletion: a table must not be cut down to a few rows.

PMID 33253437: the table had 57 rows, one per individual result. The model returned
row_list=[33] (a single row) and the table went from 57 rows to 1, so the paper was lost
with no error. Keeping far fewer rows than the table has is now sent back for a retry,
and if the retries run out the table is kept unchanged.
"""
import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_summary_data_del_agent import (
    SummaryDataDelResult,
    post_process_summary_del_result,
    try_fix_error_summary_del_result,
)

COLS = ["Drug", "ID", "Dose (mg/d)", "Cmax (ng/mL)", "Outcome"]
ROWS = [[f"Sertraline", str(i), "75", f"{10 + i}.5", "-"] for i in range(1, 21)]
# two summary lines among the individual rows, the way a PK table often ends
SUMMARY = [["Mean", "", "75", "15.0", ""], ["SD", "", "0", "3.1", ""]]
MD = dataframe_to_markdown(pd.DataFrame(ROWS + SUMMARY, columns=COLS))  # 22 rows
N = len(ROWS) + len(SUMMARY)


def _res(row_list, processed=True):
    return SummaryDataDelResult(processed=processed, row_list=row_list, col_list=None)


def test_keeping_one_row_of_many_is_sent_back_for_a_retry():
    # the 33253437 failure: row 33 of 57 kept
    with pytest.raises(RetryException) as e:
        post_process_summary_del_result(_res([3]), MD)
    assert "kept only 1 of 22 rows" in str(e.value)


def test_keeping_no_rows_is_sent_back_for_a_retry():
    with pytest.raises(RetryException):
        post_process_summary_del_result(_res([]), MD)


def test_keeping_the_individual_rows_removes_only_the_summary_rows():
    out = markdown_to_dataframe(post_process_summary_del_result(_res(list(range(len(ROWS)))), MD))
    assert out.shape[0] == len(ROWS)
    assert "Mean" not in out["Drug"].tolist()


def test_keeping_everything_leaves_the_table_as_it_is():
    out = post_process_summary_del_result(_res(list(range(N))), MD)
    assert markdown_to_dataframe(out).shape == (N, len(COLS))


def test_a_table_the_model_says_needs_no_processing_is_returned_unchanged():
    assert post_process_summary_del_result(_res(None, processed=False), MD) == MD


def test_no_row_list_keeps_all_rows():
    assert markdown_to_dataframe(post_process_summary_del_result(_res(None), MD)).shape[0] == N


def test_out_of_range_indices_do_not_count_as_kept_rows():
    # only index 0 is real; the rest are past the end of the table
    with pytest.raises(RetryException):
        post_process_summary_del_result(_res([0, 100, 101, 102]), MD)


def test_the_fallback_returns_the_table_unchanged():
    assert try_fix_error_summary_del_result(_res([3]), MD) == MD


def test_the_step_hands_the_fallback_to_the_agent():
    from extractor.agents.pk_individual.pk_ind_summary_data_del_step import SummaryDataDelStep

    assert SummaryDataDelStep().get_try_fix_error() is try_fix_error_summary_del_result
