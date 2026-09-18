from abc import abstractmethod, ABC
from typing import Callable, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
import json
from langgraph.graph import END, START, StateGraph
import pandas as pd
import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_pe_agents.pk_pe_execution_step import PKPEExecutionStep
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.constants import MAX_AGENTTOOL_TASK_STEP_COUNT
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import (
    PKPECurationWorkflowState,
    PaperTypeEnum,
    FinalAnswerEnum,
)
# from extractor.agents.pk_pe_agents.pk_pe_correction_step import PKPECuratedTablesCorrectionStep
from extractor.agents.pk_pe_agents.pk_pe_correction_code_step import PKPECuratedTablesCorrectionCodeStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, FinalAnswerEnum

logger = logging.getLogger(__name__)

class PKPEAgentToolTask(ABC):
    def __init__(
        self,
        pipeline_llm: BaseChatOpenAI,
        agent_llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
        enable_verification: bool = True,
    ):
        self.pipeline_llm = pipeline_llm
        self.agent_llm = agent_llm
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        self.output_callback = output_callback
        self.task_name = "Agent Tool Task"
        # "pipeline mode": when False, the graph stops after execution_step -
        # no verification_step, no correction_step, no retry loop at all.
        self.enable_verification = enable_verification

    def print_step(
        self,
        step_name: str | None = None,
        step_description: str | None = None,
        step_output: str | None = None,
        step_reasoning_process: str | list[str] | None = None,
        token_usage: dict | object | None = None,
    ):
        if self.output_callback is None:
            return
        self.output_callback(
            step_name=step_name,
            step_description=step_description,
            step_reasoning_process=step_reasoning_process,
            step_output=step_output,
            token_usage=token_usage,
        )

    @abstractmethod
    def _create_tool(self, pmid: str):
        pass

    @abstractmethod
    def _get_domain(self) -> str:
        pass

    @abstractmethod
    def _get_paper_type(self) -> PaperTypeEnum:
        pass

    def _build_workflow(self, pmid: str):
        execution_step = PKPEExecutionStep(
            llm=self.agent_llm,
            tool=self._create_tool(pmid),
        )
        graph = StateGraph(PKPECurationWorkflowState)
        graph.add_node("execution_step", execution_step.execute)
        graph.add_edge(START, "execution_step")

        if not self.enable_verification:
            # Pipeline mode: take execution_step's output as-is. No
            # verification_step, no correction_step - neither node is even
            # constructed, so this mode makes zero extra LLM calls beyond
            # execution_step's own.
            graph.add_edge("execution_step", END)
            return graph.compile()

        def check_verification_step(state: PKPECurationWorkflowState):
            answer = state["final_answer"]
            if answer is not None and answer.is_terminal:
                self.print_step(step_name="Final Answer")
                self.print_step(step_output=state["final_answer"].value)
                return END
            if "step_count" in state and state["step_count"] >= MAX_AGENTTOOL_TASK_STEP_COUNT:
                self.print_step(step_name="Max Step Count Reached")
                state["final_answer"] = FinalAnswerEnum.MaxStepReached
                return END
            if not "curated_table" in state or (state["curated_table"] is None or len(state["curated_table"]) == 0):
                self.print_step(step_name="No Curated Table")
                return END
            return "correction_step"
        verification_step = PKPECuratedTablesVerificationStep(
            llm=self.agent_llm, # FIXME: use agent_llm
            pmid=pmid,
            domain=self._get_domain(),
        )
        correction_step = PKPECuratedTablesCorrectionCodeStep(
            llm=self.agent_llm,
            pmid=pmid,
            domain=self._get_domain(),
        )
        graph.add_node("verification_step", verification_step.execute)
        graph.add_node("correction_step", correction_step.execute)
        graph.add_edge("execution_step", "verification_step")
        graph.add_conditional_edges(
            "verification_step",
            check_verification_step,
            {"correction_step", END},
        )
        graph.add_edge("correction_step", "verification_step")
        return graph.compile()

    def _run_workflow(self, pmid: str):
        graph = self._build_workflow(pmid)
        pmid_info = self.pmid_db.select_pmid_info(pmid)
        for s in graph.stream(
            input={
                "pmid": pmid,
                "paper_type": self._get_paper_type(),
                "paper_title": pmid_info[1],
                "paper_abstract": pmid_info[2],
                "full_text": pmid_info[3],
                "step_output_callback": self.print_step,
                "step_count": 0,
            },
            config={"max_recursion_limit": MAX_AGENTTOOL_TASK_STEP_COUNT},
            stream_mode="values",
        ):
            continue
        return s

    def run(self, pmid: str) -> tuple[bool, str | None, str | None, str | None]:
        self.print_step(step_name=f"Running {self.task_name} for pmid-{pmid}")
        state = self._run_workflow(pmid)
        final_answer = state.get("final_answer")
        curated_table = state.get("curated_table")
        if final_answer is not None:
            correct = final_answer
        elif not self.enable_verification and curated_table:
            # Pipeline mode with no early exit from execution_step (e.g. NoTable):
            # a table was curated but never checked - that's the expected,
            # intentional outcome here, not an error.
            correct = FinalAnswerEnum.Unverified
        else:
            correct = FinalAnswerEnum.PipelineError
        explanation = state.get("explanation")
        suggested_fix = state.get("suggested_fix")
        return correct, curated_table, explanation, suggested_fix

        
