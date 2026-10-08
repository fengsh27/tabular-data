"""pk-individual header categorization: a table must not silently lose all its value columns.

Same failure shape as the pk-summary benchmark (see test_pk_sum_header_categorize_value_column.py):
with no "Parameter value" column, SplitByColumnsStep returns [] without an error and the
paper is lost. The pk-individual validator only checked the column count and "Patient ID",
so such a mapping passed. The guard sends it back with the numeric columns named.
"""
import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_header_categorize_agent import (
    HeaderCategorizeResult,
    post_process_validate_categorized_result,
)

# shaped like a PK table with a subject column, a dose column and two numeric result columns
COLS = ["ID", "Dose (mg/d)", "Cmax (ng/mL) day 1", "AUC (ng·h/mL) day 1", "Notes"]
MD = dataframe_to_markdown(
    pd.DataFrame(
        [["1", "20", "12.5 (3.1)", "120 (30)", "ok"], ["2", "40", "14.0 (2.8)", "135 (25)", "ok"]],
        columns=COLS,
    )
)
ALL_UNCATEGORIZED = {
    "ID": "Patient ID",
    "Dose (mg/d)": "Uncategorized",
    "Cmax (ng/mL) day 1": "Uncategorized",
    "AUC (ng·h/mL) day 1": "Uncategorized",
    "Notes": "Uncategorized",
}
CORRECT = {
    **ALL_UNCATEGORIZED,
    "Cmax (ng/mL) day 1": "Parameter value",
    "AUC (ng·h/mL) day 1": "Parameter value",
}

# shaped like a table of abbreviations and equations: no data, correctly no value column
FORMULA_MD = dataframe_to_markdown(
    pd.DataFrame(
        [["1", "CL", "CL = Dose / AUC"], ["2", "t1/2", "t1/2 = 0.693 / k"]],
        columns=["ID", "Abbreviation", "Equation"],
    )
)
FORMULA_MAPPING = {"ID": "Patient ID", "Abbreviation": "Uncategorized", "Equation": "Uncategorized"}


def _validate(mapping, md=MD):
    return post_process_validate_categorized_result(HeaderCategorizeResult(categorized_headers=mapping), md)


def test_a_mapping_with_no_value_column_but_numeric_columns_is_sent_back_with_the_columns():
    with pytest.raises(RetryException) as e:
        _validate(ALL_UNCATEGORIZED)
    msg = str(e.value)
    assert "Cmax (ng/mL) day 1" in msg and "AUC (ng·h/mL) day 1" in msg
    assert "Dose (mg/d)" in msg  # numeric cells, so it is a candidate too
    assert "Notes" not in msg  # text cells, stays Uncategorized
    assert "ID" not in msg  # already the patient ID


def test_a_correct_mapping_passes_unchanged():
    assert _validate(CORRECT).categorized_headers == CORRECT


def test_a_table_of_abbreviations_and_equations_legitimately_has_no_value_column():
    assert _validate(FORMULA_MAPPING, FORMULA_MD).categorized_headers == FORMULA_MAPPING


def test_the_dict_form_of_the_result_is_validated_too():
    with pytest.raises(RetryException):
        post_process_validate_categorized_result({"categorized_headers": ALL_UNCATEGORIZED}, MD)


def test_existing_checks_are_unchanged():
    with pytest.raises(ValueError, match="Expected 5 columns"):
        _validate({"ID": "Patient ID"})
    with pytest.raises(ValueError, match="at least one column that serves as the patient ID"):
        _validate({**CORRECT, "ID": "Uncategorized"})
