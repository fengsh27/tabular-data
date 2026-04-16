"""
System test for the chunked correction code step.

Tests the planning → per-task code generation → chained execution flow
using real LLM calls (no mocks).

Run with:
    poetry run pytest system_tests/test_pk_pe_correction_code_step_chunked.py -v -s
"""

import pytest
from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import (
    PKPECuratedTablesCorrectionCodeStep,
    PKPECorrectionPlanResult,
)
from extractor.agents.pk_pe_agents.pk_pe_agents_types import (
    FinalAnswerEnum,
    PKPECurationWorkflowState,
)


class TestChunkedCorrectionCodeStep:
    """Tests the two-phase (plan + per-task code) correction code step."""

    def test_chunked_correction_23200982(
        self,
        llm,
        step_callback,
        title_23200982,
        abstract_23200982,
        md_table_23200982_table_1,
        md_table_23200982_table_2,
        md_table_23200982_table_3,
        md_curated_table_pk_individual_23200982,
        verification_explanation_23200982,
        verification_final_answer_23200982,
        verification_suggested_fix_23200982,
    ):
        """
        23200982 has a large correction: the curated table only contains
        Infliximab data but should include Adalimumab, Certolizumab, and PEG.
        This is exactly the kind of multi-row correction that used to exceed
        the single-shot token limit.
        """
        pmid = "23200982"

        step = PKPECuratedTablesCorrectionCodeStep(
            llm=llm,
            pmid=pmid,
            domain="pharmacokinetics",
        )

        state: PKPECurationWorkflowState = {
            "paper_title": title_23200982,
            "paper_abstract": abstract_23200982,
            "source_tables": [
                md_table_23200982_table_1,
                md_table_23200982_table_2,
                md_table_23200982_table_3,
            ],
            "curated_table": md_curated_table_pk_individual_23200982,
            "verification_reasoning_process": verification_explanation_23200982,
            "final_answer": verification_final_answer_23200982,
            "suggested_fix": verification_suggested_fix_23200982,
            "step_output_callback": step_callback,
        }

        original_curated = state["curated_table"]

        state = step.execute(state)

        # The correction must have produced a table
        assert state["curated_table"] is not None
        assert len(state["curated_table"]) > 0

        # The table should be different from the original (corrections applied)
        assert state["curated_table"] != original_curated

        # The corrected table must be valid markdown that round-trips
        df = markdown_to_dataframe(state["curated_table"])
        assert df is not None
        assert not df.empty

        # The original only had Infliximab rows; corrected should have more rows
        df_original = markdown_to_dataframe(original_curated)
        assert len(df) > len(df_original), (
            f"Corrected table should have more rows than original "
            f"(got {len(df)} vs {len(df_original)})"
        )

        # Should NOT have ended in CorrectionError
        assert state.get("final_answer") != FinalAnswerEnum.CorrectionError
