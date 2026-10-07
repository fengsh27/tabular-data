"""Cross-pipeline hardening: apply pk_individual's bug-class fixes elsewhere.

Covers the 3 patterns applied outside pk_individual this round:
- default_factory=list on required parallel list fields (a reply that omits a
  key must still parse, instead of crashing Pydantic validation before any
  retry/fallback runs - the actual mechanism of the original production
  failure).
- bare ValueError -> RetryException on structural guards (so a plausible bad
  reply gets retry-with-feedback instead of 5 blind identical retries).
- the pe_study_out row-categorize step: one bad row-header key must be
  skipped, not abort categorization for every other key in the table.
- pk_summary's param-type-align rename: a fix_col_name miss must raise, not
  silently no-op the rename.
"""
import pandas as pd
import pytest

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.common_agent.common_agent import RetryException


# ---- default_factory=list: a partial reply must still parse ----

def test_pk_sum_param_type_unit_omitted_field_defaults_to_empty_list():
    from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_agent import (
        ExtractedParamTypeUnits,
    )

    # simulates a reply that only ever wrote "parameter_types" (e.g. a looping
    # model that never reached "parameter_units") - this used to be a hard
    # Pydantic ValidationError before post_process/retry ever ran.
    parsed = ExtractedParamTypeUnits(**{"parameter_types": ["Cmax", "Tmax"]})
    assert parsed.parameter_types == ["Cmax", "Tmax"]
    assert parsed.parameter_units == []


def test_pk_drug_ind_drug_info_omitted_field_defaults_to_empty_list():
    from extractor.agents.pk_drug_individual.pk_drug_ind_drug_info_agent import (
        DrugInfoResult,
    )

    parsed = DrugInfoResult(**{})
    assert parsed.population_combinations == []


# ---- pk_summary unit-extract mismatch: must retry with a readable message, not NameError ----

def test_pk_sum_unit_extract_mismatch_raises_retryexception_not_namererror():
    from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_agent import (
        ExtractedParamTypeUnits,
        ParamTypeUnitExtractionResult,
        post_process_validate_matched_tuple,
    )

    md_table = dataframe_to_markdown(
        pd.DataFrame({"Parameter type": ["Cmax", "Tmax", "AUC"]})
    )
    res = ParamTypeUnitExtractionResult(
        extracted_param_units=ExtractedParamTypeUnits(
            parameter_types=["Cmax", "Tmax"], parameter_units=["ng/mL"]
        )
    )
    with pytest.raises(RetryException) as e:
        post_process_validate_matched_tuple(res, md_table, col_mapping={})
    assert "Expected 3 rows" in str(e.value) and "2 (types)" in str(e.value) and "1 (units)" in str(e.value)


# ---- bare ValueError -> RetryException on a representative "empty reply" guard ----

def test_pk_drug_ind_drug_info_refine_empty_reply_raises_retryexception():
    from extractor.agents.pk_drug_individual.pk_drug_ind_drug_info_refine_agent import (
        DrugInfoRefinedResult,
        post_process_refined_drug_info,
    )

    md_table_drug = dataframe_to_markdown(pd.DataFrame({"Patient ID": ["1", "2"]}))
    res = DrugInfoRefinedResult(refined_drug_combinations=[])
    with pytest.raises(RetryException, match="No valid entries found"):
        post_process_refined_drug_info(res, md_table_drug)


# ---- pe_study_out row-categorize: one bad key must not abort the whole step ----

def test_row_categorize_step_skips_a_bad_key_instead_of_aborting_the_step(monkeypatch):
    from extractor.agents.pe_study_outcome.pe_study_out_row_categorize_step import (
        RowCategorizeStep,
    )

    md_table = dataframe_to_markdown(pd.DataFrame({"good_key": ["a", "b"]}))
    # "bad_key" does not exist as a column in md_table - mirrors the real
    # failure mode (upstream header-categorize quoting/casing drift).
    col_mapping = {"good_key": "Row headers", "bad_key": "Row headers"}

    class _StubAgent:
        def __init__(self):
            self.calls = 0

        def go(self, **kwargs):
            self.calls += 1
            return (
                {"categorized_headers": {"a": "Characteristic", "b": "Outcome"}},
                None,
                {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
                None,
            )

    stub_agent = _StubAgent()
    step = RowCategorizeStep()
    monkeypatch.setattr(step, "get_agent", lambda llm: stub_agent)
    state = {"llm": None, "md_table": md_table, "col_mapping": col_mapping}

    _res, return_dict, _token_usage = step.execute_directly(state)

    assert return_dict == {"good_key": {"a": "Characteristic", "b": "Outcome"}}
    assert "bad_key" not in return_dict
    assert stub_agent.calls == 1  # the bad key never reached agent.go


# ---- pk_summary param-type-align: fix_col_name miss must raise, not silently no-op ----

def test_param_type_align_unmatched_col_name_raises_retryexception():
    from extractor.agents.pk_summary.pk_sum_param_type_align_agent import (
        ParameterTypeAlignResult,
        post_process_parameter_type_align,
    )

    md_table_summary = dataframe_to_markdown(
        pd.DataFrame({"Cmax": ["12.5"], "Tmax": ["2.0"]})
    )
    res = ParameterTypeAlignResult(col_name="Totally Unrelated Column Name")
    with pytest.raises(RetryException) as e:
        post_process_parameter_type_align(res, md_table_summary)
    assert "Totally Unrelated Column Name" in str(e.value)


def test_param_type_align_matching_col_name_still_renames_correctly():
    from extractor.agents.pk_summary.pk_sum_param_type_align_agent import (
        ParameterTypeAlignResult,
        post_process_parameter_type_align,
    )

    md_table_summary = dataframe_to_markdown(
        pd.DataFrame({"Parameter": ["Cmax", "Tmax"], "Value": ["12.5", "2.0"]})
    )
    res = ParameterTypeAlignResult(col_name="Parameter")
    out = post_process_parameter_type_align(res, md_table_summary)
    assert "Parameter type" in out
