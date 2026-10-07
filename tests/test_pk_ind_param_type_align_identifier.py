"""pk-individual param-type align: an identifier column is never a parameter-type column.

Qwen3.8 answered col_name="Unnamed: 0" (the subject index) for PMID 33253437. The step
transposed the whole table on it, collapsing 12 columns to 2 and every later step ran on
junk (multi-agent / pipeline arms). A patient-identifier column can never hold the
parameter-type labels, so the table must be returned as it is.
"""
import pandas as pd
import pytest

from TabFuncFlow.operations.f_transpose import f_transpose
from TabFuncFlow.utils.table_utils import (
    dataframe_to_markdown,
    deduplicate_headers,
    fill_empty_headers,
    markdown_to_dataframe,
    remove_empty_col_row,
)
from extractor.agents.pk_individual.pk_ind_param_type_align_agent import (
    ParameterTypeAlignResult,
    _is_identifier_column,
    post_process_parameter_type_align,
)

# shaped like PMID 33253437: a subject index, an ID, then one column per measurement
COLS = [
    "Unnamed: 0", "ID", "Dose (mg/d)", "Mother's PL III trimester (ng/ml)",
    "Mother's PL (ng/ml) delivery", "Infant's PL (ng/ml)", "Umbilical maternal ratio (%)",
    "Other drugs", "Expected phenotype (CYP2D6: major CYP metabolizer)",
    "Maternal outcomes", "Bleeding (ml)", "Neonatal outcomes",
]
MD = dataframe_to_markdown(
    pd.DataFrame([[f"r{i}", str(i)] + ["1"] * 10 for i in range(3)], columns=COLS)
)

# shaped like a long-format table: the parameter type is a column of its own
LONG_MD = dataframe_to_markdown(
    pd.DataFrame(
        [["1", "Cmax", "12.5"], ["1", "AUC", "120"], ["2", "Cmax", "14.0"], ["2", "AUC", "135"]],
        columns=["Subject", "Parameter", "Value"],
    )
)


def _align(col_name, md=MD):
    return post_process_parameter_type_align(ParameterTypeAlignResult(col_name=col_name), md)


def test_an_identifier_col_name_leaves_the_12_column_table_unchanged():
    out = _align("Unnamed: 0")
    assert out == dataframe_to_markdown(markdown_to_dataframe(MD))
    assert markdown_to_dataframe(out).shape == (3, 12)  # the failure mode was (3, 2)


@pytest.mark.parametrize(
    "col_name",
    [
        "Unnamed: 0",
        "unnamed: 0",
        "Unnamed: 0_level_0",
        "('Unnamed: 0_level_0', 'Subject')",
        "ID",
        "id",
        "Patient ID",
        "Subject",
        "No.",
        "No",
    ],
)
def test_identifier_names_are_recognized_case_insensitively(col_name):
    assert _align(col_name) == dataframe_to_markdown(markdown_to_dataframe(MD))


@pytest.mark.parametrize("col_name", ["Parameter", "Analyte", "Dose (mg/d)", "Hybrid", "Nominal", "Notes"])
def test_a_real_parameter_column_name_is_not_taken_for_an_identifier(col_name):
    # "Hybrid" and "Nominal" contain "id" / "no" inside a word; "Notes" begins with "no" but is not the word
    assert not _is_identifier_column(col_name)


def test_a_real_parameter_column_still_transposes_as_before():
    # the unchanged pre-fix path: transpose the whole table, then clean up the headers
    expected = deduplicate_headers(
        fill_empty_headers(
            fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(f_transpose(markdown_to_dataframe(LONG_MD)))))
        )
    )
    assert _align("Parameter", LONG_MD) == expected
    assert _align("Parameter", LONG_MD) != dataframe_to_markdown(markdown_to_dataframe(LONG_MD))


def test_no_col_name_is_still_a_no_op_for_the_table():
    assert _align(None) == dataframe_to_markdown(markdown_to_dataframe(MD))
