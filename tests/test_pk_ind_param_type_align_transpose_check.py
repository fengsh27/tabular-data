"""pk-individual param-type align: transpose when the model asks, then check the result.

Replaces the name-based guard from commit 6e8669a ("never transpose on a column called
Unnamed: N / ID / Subject / No."). That guard was built for PMID 33253437, whose blank-headed
first column holds drug names, and it silently broke PMID 34746508, whose blank-headed first
column holds the parameter labels of a transposed per-case table (cases are the columns). Both
are "Unnamed: 0"; what tells them apart is the values: parameter-type labels are distinct, a
group label repeats. See docs/TODO_QWEN38_ISSUES.md and docs/cascade_mode_results.md (round 4).

The two fixtures are the real source tables of those papers (tests/data/pk_individual_tables/).
"""
import os

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
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_param_type_align_agent import (
    MIN_DISTINCT_LABEL_RATIO,
    ParameterTypeAlignResult,
    check_transpose_result,
    post_process_parameter_type_align,
    try_fix_parameter_type_align,
)
from extractor.agents.pk_individual.pk_ind_param_type_align_step import ParametertypeAlignStep

DATA = os.path.join(os.path.dirname(__file__), "data", "pk_individual_tables")


def _read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return fh.read()


T34746508 = _read("34746508_table_2.md")  # 9 parameter rows x 5 cases, blank header over the labels
T33253437 = _read("33253437_table_2.md")  # 56 subjects x 12 columns, blank header over the drug names

# a long-format table: the parameter type is a column of its own (pre-existing behaviour kept)
LONG_MD = dataframe_to_markdown(
    pd.DataFrame(
        [["1", "Cmax", "12.5"], ["1", "AUC", "120"], ["2", "Cmax", "14.0"], ["2", "AUC", "135"]],
        columns=["Subject", "Parameter", "Value"],
    )
)


def _align(col_name, md):
    return post_process_parameter_type_align(ParameterTypeAlignResult(col_name=col_name), md)


# --- PMID 34746508: the model's answer "Unnamed: 0" is right and must be followed ---

def test_34746508_transposes_into_one_row_per_case():
    out = _align("Unnamed: 0", T34746508)
    df = markdown_to_dataframe(out)
    assert df.shape == (5, 10)
    assert df.iloc[:, 0].tolist() == ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
    for col in ("IFX/UST in mother’s blood", "IFX/UST in cord blood", "IFX/UST at 6 months"):
        assert col in df.columns
    assert df.loc[df.iloc[:, 0] == "Case 1", "IFX/UST in cord blood"].item() == "36.89 μg/ml"
    assert df.loc[df.iloc[:, 0] == "Case 5", "IFX/UST at 6 months"].item() == "Not detected"


def test_34746508_was_blocked_by_the_old_name_guard_and_is_not_any_more():
    # the pre-fix path returned the table unchanged; it must now differ (transposed)
    assert _align("Unnamed: 0", T34746508) != dataframe_to_markdown(markdown_to_dataframe(T34746508))


# --- PMID 33253437: the model's answer "Unnamed: 0" is wrong and must be sent back ---

def test_33253437_rejects_the_drug_name_column_with_a_retry():
    with pytest.raises(RetryException) as exc:
        _align("Unnamed: 0", T33253437)
    msg = str(exc.value)
    assert "'Unnamed: 0'" in msg and "distinct" in msg and "null" in msg
    assert "6 distinct" in msg  # the six drug names over 47 rows


def test_33253437_is_rejected_on_values_not_on_the_column_name():
    df = markdown_to_dataframe(T33253437)
    n = df["Unnamed: 0"].astype(str).str.strip().replace("nan", "").ne("").sum()
    distinct = df["Unnamed: 0"].astype(str).str.strip().replace("nan", "").loc[lambda s: s.ne("")].nunique()
    assert distinct / n < MIN_DISTINCT_LABEL_RATIO  # that ratio, not the header text, is the trigger
    with pytest.raises(RetryException):
        check_transpose_result("Unnamed: 0", df, f_transpose(df))


def test_33253437_fallback_after_retries_keeps_the_table_untransposed():
    res = ParameterTypeAlignResult(col_name="Unnamed: 0")
    out = try_fix_parameter_type_align(res, T33253437)
    assert markdown_to_dataframe(out).shape == markdown_to_dataframe(T33253437).shape == (56, 12)
    assert ParametertypeAlignStep().get_try_fix_error() is try_fix_parameter_type_align


# --- behaviour that must not change ---

def test_no_col_name_leaves_the_table_as_it_is():
    for md in (T34746508, T33253437):
        assert _align(None, md) == dataframe_to_markdown(markdown_to_dataframe(md))


def test_a_real_parameter_column_still_transposes_as_before():
    expected = deduplicate_headers(
        fill_empty_headers(
            fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(f_transpose(markdown_to_dataframe(LONG_MD)))))
        )
    )
    assert _align("Parameter", LONG_MD) == expected


def test_a_column_that_leaves_too_few_columns_is_rejected():
    md = dataframe_to_markdown(pd.DataFrame({"Parameter": ["Cmax"], "Subject 1": ["12.5"]}))
    with pytest.raises(RetryException, match="leaves only"):
        _align("Parameter", md)


def test_a_made_up_column_name_is_sent_back():
    with pytest.raises(RetryException, match="not a column of the table"):
        _align("Parameter type (made up)", T34746508)
