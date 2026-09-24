"""VerifyScopeEnum.PerTable: verify + correct each source table on its own, before
combining, instead of combining first and verifying the whole paper
(VerifyScopeEnum.Combined, pk_pe_verification_step.py / pk_pe_correction_code_step.py
run directly on the combined table - see pk_pe_agenttool_task.py for the graph shapes).

Confirmed by a manual spike (session 012C68HxDt9bJzFSfHvSU4AD) before this was built: on
the two worst pk-individual papers in the benchmark, this cut tokens 74-94% and matched or
beat the combined loop's score for qwen3.6 (one paper's combined run had made things worse;
scoping to one table at a time stopped that from compounding). It is not a strict
improvement though: a check that only makes sense with the whole paper in view - a row-scope
rule ("dose rows don't belong here"), duplicate rows between two tables, Patient ID
numbering consistency across tables - is out of view for a per-table pass, and one paper's
gpt-5.4 run scored lower per-table for exactly that reason (see the module docstring on
VerifyScopeEnum for the fuller picture). Every reused component (verification, correction,
their prompts) is unchanged; this module only changes what table(s) they're pointed at.
"""
from typing import Optional
import logging

import pandas as pd
from langchain_openai.chat_models.base import BaseChatOpenAI
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, FinalAnswerEnum
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import PKPECuratedTablesCorrectionCodeStep
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.constants import MAX_PER_TABLE_STEP_COUNT

logger = logging.getLogger(__name__)

# Worst-first: the paper-level answer is the worst-ranked verdict among its tables. Anything
# not listed (there is nothing else FinalAnswerEnum defines) falls back to the end (best).
_SEVERITY_ORDER = [
    FinalAnswerEnum.PipelineError,
    FinalAnswerEnum.VerificationError,
    FinalAnswerEnum.CorrectionError,
    FinalAnswerEnum.MaxStepReached,
    FinalAnswerEnum.Incorrect,
    FinalAnswerEnum.NoIndividualData,
    FinalAnswerEnum.NoTable,
    FinalAnswerEnum.Unverified,
    FinalAnswerEnum.Correct,
]


def worst_case_final_answer(answers: list[FinalAnswerEnum]) -> FinalAnswerEnum:
    """The paper's overall verdict when each table has its own. Ties within a severity
    (e.g. two tables both Incorrect) collapse to that one value; an empty list (every table
    produced nothing) is NoTable."""
    if not answers:
        return FinalAnswerEnum.NoTable
    return min(answers, key=lambda a: _SEVERITY_ORDER.index(a) if a in _SEVERITY_ORDER else len(_SEVERITY_ORDER))


def combine_markdown_tables(tables_md: list[str]) -> str:
    """Concatenate already-verified/corrected per-table markdown back into one table, the
    same shape PKPEExecutionStep would have produced in VerifyScopeEnum.Combined."""
    dfs = [markdown_to_dataframe(t) for t in tables_md if t]
    combined = pd.concat(dfs, axis=0).reset_index(drop=True) if dfs else pd.DataFrame()
    return dataframe_to_markdown(combined)


class PKPEPerTableVerifyCorrectStep(PKPECommonStep):
    """One graph node standing in for the whole verification_step <-> correction_step loop
    of VerifyScopeEnum.Combined. Internally it re-runs that same loop once per table, scoped
    to just that table (curated_table=this table's markdown, source_tables=[this table's
    source] only), then combines the results - so from the graph's point of view it behaves
    like a single, larger step that leaves state["curated_table"] and state["final_answer"]
    set exactly as VerifyScopeEnum.Combined would, and every other consumer (the CSV writer,
    PKPEAgentToolTask.run()) doesn't need to know which scope produced them.
    """

    def __init__(self, llm: BaseChatOpenAI, pmid: str, domain: str):
        super().__init__(llm)
        self.step_name = "PK PE Per-Table Verify+Correct Step"
        self.pmid = pmid
        self.domain = domain
        self.verification_step = PKPECuratedTablesVerificationStep(llm=llm, pmid=pmid, domain=domain)
        self.correction_step = PKPECuratedTablesCorrectionCodeStep(llm=llm, pmid=pmid, domain=domain)

    def _run_one_table(
        self, state: PKPECurationWorkflowState, table_md: str, source_md: str, table_label: str
    ) -> tuple[str, FinalAnswerEnum, dict]:
        sub_state: PKPECurationWorkflowState = {
            **state,
            "curated_table": table_md,
            "source_tables": [source_md],
            "previous_verification_thoughts": [],
            "previous_errors": None,
            "final_answer": None,
        }
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        self._print_step(state, step_name=table_label)
        answer: Optional[FinalAnswerEnum] = None
        for _round in range(MAX_PER_TABLE_STEP_COUNT):
            sub_state, tok = self.verification_step._execute_directly(sub_state)
            total_token_usage = increase_token_usage(total_token_usage, tok)
            answer = sub_state.get("final_answer")
            if answer is not None and answer.is_terminal:
                break
            sub_state, tok = self.correction_step._execute_directly(sub_state)
            total_token_usage = increase_token_usage(total_token_usage, tok)
        else:
            answer = FinalAnswerEnum.MaxStepReached
            sub_state["final_answer"] = answer
        return sub_state["curated_table"], answer, total_token_usage

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        answer = state.get("final_answer")
        if answer is not None and answer.is_terminal:
            return state, {**DEFAULT_TOKEN_USAGE}

        curated_tables = state.get("curated_tables")
        source_tables = state.get("source_tables")
        if not curated_tables:
            # The tool that ran didn't populate curated_tables - nothing to scope to. The
            # graph is only built this way for tools that do (see pk_pe_agenttool_task.py),
            # so reaching this is a bug elsewhere, not a normal fallback path; fail loudly
            # rather than silently verifying nothing.
            raise ValueError(
                "PKPEPerTableVerifyCorrectStep requires state['curated_tables']; "
                "the configured tool did not populate it."
            )

        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        per_table_answers: list[FinalAnswerEnum] = []
        corrected_tables: list[str] = []
        n = len(curated_tables)
        for i, (table_md, source_md) in enumerate(zip(curated_tables, source_tables)):
            if not table_md:
                continue  # this source table produced no data; nothing to verify/correct
            corrected_md, table_answer, tok = self._run_one_table(
                state, table_md, source_md, table_label=f"Table {i + 1}/{n}"
            )
            total_token_usage = increase_token_usage(total_token_usage, tok)
            per_table_answers.append(table_answer)
            corrected_tables.append(corrected_md)
            logger.info(f"[{self.pmid}] table {i + 1}/{n}: {table_answer}")

        state["curated_table"] = combine_markdown_tables(corrected_tables)
        state["final_answer"] = worst_case_final_answer(per_table_answers)
        return state, total_token_usage

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        return super().leave_step(state, token_usage)
