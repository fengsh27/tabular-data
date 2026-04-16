"""
System test for pk_summary verification + correction loop on paper 19065567.

This paper (SU5416 Phase I study) has a complex source table where:
- "No. of patients" column maps to "Subject N"
- Stratum II dose=35 has only 1 patient (no mean/SD)
- The verification step previously misidentified Subject N values,
  triggering oscillating corrections.

Tests the full verification → correction → re-verification flow
with real LLM calls (no mocks).

Run with:
    poetry run pytest system_tests/test_pk_summary_verification_correction_19065567.py -v -s
"""

import pytest
from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.pk_pe_agents.pk_pe_verification_step import (
    PKPECuratedTablesVerificationStep,
)
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import (
    PKPECuratedTablesCorrectionCodeStep,
)
from extractor.agents.pk_pe_agents.pk_pe_agents_types import (
    FinalAnswerEnum,
    PKPECurationWorkflowState,
)


class TestVerificationFilters:
    """Unit tests for verification step filters (no LLM needed)."""

    def test_remove_stale_fixes(self):
        """
        If the suggested fix says change "52" to "55.1" at idx 0,
        but the actual value is already "55.1", the fix should be removed.
        """
        curated_table = (
            "| Drug name | Parameter value |\n"
            "| --- | --- |\n"
            "| SU5416 | 55.1 |\n"
            "| SU5416 | 33.1 |\n"
            "| SU5416 | 48.0 |\n"
        )

        # This fix claims idx 0 has "52" but it actually has "55.1" — stale
        text = 'idx 0, Col "Parameter value": change "52" to "55.1"'
        result = PKPECuratedTablesVerificationStep._remove_stale_fixes(text, curated_table)
        assert result.strip() == "", f"Stale fix should have been removed, got: {result}"

    def test_remove_stale_fixes_keeps_valid(self):
        """
        If the suggested fix correctly identifies the current value, it should be kept.
        """
        curated_table = (
            "| Drug name | Parameter value |\n"
            "| --- | --- |\n"
            "| SU5416 | 52 |\n"
            "| SU5416 | 33.1 |\n"
        )

        # This fix correctly says idx 0 has "52" — valid
        text = 'idx 0, Col "Parameter value": change "52" to "55.1"'
        result = PKPECuratedTablesVerificationStep._remove_stale_fixes(text, curated_table)
        assert "52" in result and "55.1" in result, f"Valid fix should have been kept, got: {result}"

    def test_remove_stale_fixes_mixed(self):
        """
        Mixed case: one stale fix and one valid fix. Only the valid one should remain.
        """
        curated_table = (
            "| Drug name | Parameter value | Subject N |\n"
            "| --- | --- | --- |\n"
            "| SU5416 | 55.1 | 1 |\n"
            "| SU5416 | 33.1 | 2 |\n"
        )

        text = (
            'idx 0, Col "Parameter value": change "52" to "55.1"\n'
            'idx 0, Col "Subject N": change "1" to "13"'
        )
        result = PKPECuratedTablesVerificationStep._remove_stale_fixes(text, curated_table)
        # Stale fix (Parameter value) should be removed, valid fix (Subject N) should remain
        assert "Parameter value" not in result, f"Stale fix should have been removed, got: {result}"
        assert 'Subject N' in result, f"Valid fix should have been kept, got: {result}"

    def test_remove_noop_fixes(self):
        """No-op fixes where from == to should be removed."""
        text = (
            'idx 0, Col "P value": change "0.001" to "0.001"\n'
            'idx 1, Col "P value": change "N/A" to "0.007"'
        )
        result = PKPECuratedTablesVerificationStep._remove_noop_fixes(text)
        assert "idx 0" not in result
        assert "idx 1" in result

    def test_remove_oscillation_fixes(self):
        """Fixes that revert a previous correction should be removed."""
        previous_thoughts = [
            'Explanation: idx 1, Col "P value": change "0.001" to "0.01"\nSuggested fix: idx 1, Col "P value": change "0.001" to "0.01"'
        ]
        # Current wants to revert: 0.01 → 0.001
        text = 'idx 1, Col "P value": change "0.01" to "0.001"'
        result = PKPECuratedTablesVerificationStep._remove_oscillation_fixes(text, previous_thoughts)
        assert result.strip() == "", f"Oscillation fix should have been removed, got: {result}"


class TestPKSummaryVerificationCorrection19065567:
    """Tests the verification and correction steps for paper 19065567 pk_summary."""

    def test_correction_uses_idx_not_value_matching(
        self,
        llm,
        step_callback,
        title_19065567,
        abstract_19065567,
        md_table_19065567_table_1,
        md_curated_table_pk_summary_19065567,
        verification_explanation_19065567,
        verification_final_answer_19065567,
        verification_suggested_fix_19065567,
    ):
        """
        The correction step should use df.at[idx, col] for targeted edits,
        not value-based matching which can match zero rows.
        """
        pmid = "19065567"

        step = PKPECuratedTablesCorrectionCodeStep(
            llm=llm,
            pmid=pmid,
            domain="pharmacokinetics",
        )

        state: PKPECurationWorkflowState = {
            "paper_title": title_19065567,
            "paper_abstract": abstract_19065567,
            "source_tables": [md_table_19065567_table_1],
            "curated_table": md_curated_table_pk_summary_19065567,
            "verification_reasoning_process": verification_explanation_19065567,
            "final_answer": verification_final_answer_19065567,
            "suggested_fix": verification_suggested_fix_19065567,
            "step_output_callback": step_callback,
        }

        original_curated = state["curated_table"]

        state = step.execute(state)

        # The correction should have produced changes (not "no changes")
        assert state["curated_table"] != original_curated, (
            "Correction step produced no changes — likely used value-based matching "
            "instead of idx-based df.at[] access"
        )

        # The corrected table must be valid markdown
        df = markdown_to_dataframe(state["curated_table"])
        assert df is not None
        assert not df.empty

        # Should NOT have ended in CorrectionError
        assert state.get("final_answer") != FinalAnswerEnum.CorrectionError

    def test_verification_noop_filter(
        self,
        llm,
        step_callback,
        title_19065567,
        abstract_19065567,
        md_table_19065567_table_1,
        md_curated_table_pk_summary_19065567,
    ):
        """
        Verification should filter out no-op fixes (change "X" to "X")
        and not pass them to the correction step.
        """
        pmid = "19065567"

        verification = PKPECuratedTablesVerificationStep(
            llm=llm,
            pmid=pmid,
            domain="pharmacokinetics",
        )

        state: PKPECurationWorkflowState = {
            "paper_title": title_19065567,
            "paper_abstract": abstract_19065567,
            "source_tables": [md_table_19065567_table_1],
            "curated_table": md_curated_table_pk_summary_19065567,
            "step_output_callback": step_callback,
        }

        state = verification.execute(state)

        # If verification says incorrect, check that suggested_fix has no no-ops
        if state["final_answer"] == FinalAnswerEnum.Incorrect:
            suggested_fix = state.get("suggested_fix", "")
            if suggested_fix and suggested_fix != "N/A":
                import re
                pattern = re.compile(r'change\s+"([^"]*)"\s+to\s+"([^"]*)"')
                for line in suggested_fix.split("\n"):
                    m = pattern.search(line)
                    if m:
                        assert m.group(1) != m.group(2), (
                            f"No-op fix should have been filtered: {line.strip()}"
                        )
    
    def test_verification_correction_steps(
        self,
        llm,
        step_callback,
        title_19065567,
        abstract_19065567,
        md_table_19065567_table_1,
        md_curated_table_pk_summary_19065567,
    ):
        """
        Test the full verification → correction → re-verification flow.
        """
        pmid = "19065567"

        verification = PKPECuratedTablesVerificationStep(
            llm=llm,
            pmid=pmid,
            domain="pharmacokinetics",
        )
        correction = PKPECuratedTablesCorrectionCodeStep(
            llm=llm,
            pmid=pmid,
            domain="pharmacokinetics",
        )

        state: PKPECurationWorkflowState = {
            "paper_title": title_19065567,
            "paper_abstract": abstract_19065567,
            "source_tables": [md_table_19065567_table_1],
            "curated_table": md_curated_table_pk_summary_19065567,
            "step_output_callback": step_callback,
        }

        original_curated = state["curated_table"]

        # state = verification.execute(state)

        max_iteration = 5
        iteration = 0
        while iteration < max_iteration:
            state = verification.execute(state)
            if state["final_answer"] == FinalAnswerEnum.Correct:
                break
            state = correction.execute(state)
            iteration += 1

        # The corrected table must be valid markdown
        df = markdown_to_dataframe(state["curated_table"])
        assert df is not None
        assert not df.empty

        # Should NOT have ended in CorrectionError
        assert state.get("final_answer") != FinalAnswerEnum.CorrectionError
