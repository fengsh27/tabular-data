"""Header categorization: keys must be the table's real column names.

The prompt lists the headers in double quotes and gpt-4o copies them into the keys about
half the time (`{'"ID"': 'Patient ID'}`). The validator only counted keys, so this passed;
SplitByColumnsStep then looked up the real names, found no "Parameter value" column and
returned an empty sub-table list with no error. Every later step ran on nothing and the
paper was silently lost (multi-agent benchmark arm A: PMIDs 33253437 and 34746508).
"""
import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_header_categorize_agent import (
    HeaderCategorizeResult,
    get_header_categorize_prompt,
    normalize_header_keys,
    post_process_validate_categorized_result,
)
from extractor.agents.pk_individual.pk_ind_split_by_col_step import SplitByColumnsStep

# headers of PMID 33253437 (apostrophes and parentheses included)
COLS = [
    "Unnamed: 0", "ID", "Dose (mg/d)", "Mother's PL III trimester (ng/ml)",
    "Mother's PL (ng/ml) delivery", "Infant's PL (ng/ml)", "Umbilical maternal ratio (%)",
    "Other drugs", "Expected phenotype (CYP2D6: major CYP metabolizer)",
    "Maternal outcomes", "Bleeding (ml)", "Neonatal outcomes",
]
CATEGORY = {
    "Unnamed: 0": "Uncategorized", "ID": "Patient ID", "Dose (mg/d)": "Uncategorized",
    "Mother's PL III trimester (ng/ml)": "Parameter value",
    "Mother's PL (ng/ml) delivery": "Parameter value",
    "Infant's PL (ng/ml)": "Parameter value", "Umbilical maternal ratio (%)": "Parameter value",
    "Other drugs": "Uncategorized",
    "Expected phenotype (CYP2D6: major CYP metabolizer)": "Uncategorized",
    "Maternal outcomes": "Uncategorized", "Bleeding (ml)": "Parameter value",
    "Neonatal outcomes": "Uncategorized",
}
MD = dataframe_to_markdown(
    pd.DataFrame([[f"r{i}", str(i)] + ["1"] * 10 for i in range(3)], columns=COLS)
)


def _wrap(mapping, q='"'):
    return {f"{q}{k}{q}": v for k, v in mapping.items()}


def test_clean_keys_are_unchanged():
    assert normalize_header_keys(dict(CATEGORY), MD) == CATEGORY


@pytest.mark.parametrize("quote", ['"', "'", "`"])
def test_wrapping_quotes_are_stripped(quote):
    assert normalize_header_keys(_wrap(CATEGORY, quote), MD) == CATEGORY


def test_inner_apostrophes_survive():
    fixed = normalize_header_keys(_wrap(CATEGORY), MD)
    assert "Mother's PL III trimester (ng/ml)" in fixed


def test_case_and_padding_differences_are_tolerated():
    got = normalize_header_keys({" id ": "Patient ID", **{k: v for k, v in CATEGORY.items() if k != "ID"}}, MD)
    assert got["ID"] == "Patient ID"


def test_unmatched_key_raises_a_retry_with_the_offending_key():
    bad = dict(CATEGORY)
    bad["Bleeding (millilitres)"] = bad.pop("Bleeding (ml)")
    with pytest.raises(RetryException) as e:
        normalize_header_keys(bad, MD)
    assert "Bleeding (millilitres)" in str(e.value) and "WITHOUT any surrounding quotes" in str(e.value)


def test_post_process_returns_real_names_for_quoted_output():
    res = post_process_validate_categorized_result(
        HeaderCategorizeResult(categorized_headers=_wrap(CATEGORY)), MD
    )
    assert res.categorized_headers == CATEGORY


def test_post_process_still_enforces_count_and_patient_id():
    no_id = {k: ("Uncategorized" if v == "Patient ID" else v) for k, v in CATEGORY.items()}
    with pytest.raises(ValueError, match="patient ID"):
        post_process_validate_categorized_result(HeaderCategorizeResult(categorized_headers=no_id), MD)
    missing_one = {k: v for k, v in CATEGORY.items() if k != "Bleeding (ml)"}
    with pytest.raises(ValueError, match="Expected 12 columns"):
        post_process_validate_categorized_result(HeaderCategorizeResult(categorized_headers=missing_one), MD)


def test_prompt_tells_the_model_not_to_copy_the_quotes():
    assert "do NOT include them in the keys" in get_header_categorize_prompt(MD)


def _sub_tables(col_mapping):
    state = {"col_mapping": dict(col_mapping)}
    SplitByColumnsStep().leave_step(state, None, processed_res=[MD], token_usage=None)
    return state["md_table_list"]


def test_regression_quoted_keys_used_to_empty_the_sub_tables_silently():
    # the failure mode: the same step, fed the raw quoted mapping, finds nothing
    assert _sub_tables(_wrap(CATEGORY)) == []
    # fed the validated mapping it builds one sub-table per Parameter value column
    fixed = post_process_validate_categorized_result(
        HeaderCategorizeResult(categorized_headers=_wrap(CATEGORY)), MD
    ).categorized_headers
    assert len(_sub_tables(fixed)) == 5
