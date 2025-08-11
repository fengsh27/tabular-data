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
from extractor.constants import MAX_STEP_COUNT
from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState, PaperTypeEnum
from extractor.agents.pk_pe_agents.pk_pe_correction_step import PKPECuratedTablesCorrectionStep

logger = logging.getLogger(__name__)

class PKPEAgentToolTask(ABC):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        pmid_db: PMIDDB | None = None,
        output_callback: Callable | None = None,
    ):
        self.llm = llm
        self.pmid_db = pmid_db if pmid_db is not None else PMIDDB()
        self.output_callback = output_callback
        self.task_name = "Agent Tool Task"

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
        def check_verification_step(state: PKPECurationWorkflowState):
            if state["final_answer"] is not None and state["final_answer"]:
                self.print_step(step_name="Final Answer")
                self.print_step(step_output=state["final_answer"])
                return END
            if "step_count" in state and state["step_count"] >= MAX_STEP_COUNT:
                self.print_step(step_name="Max Step Count Reached")
                return END
            return "correction_step"
        execution_step = PKPEExecutionStep(
            llm=self.llm,
            tool=self._create_tool(pmid),
        )
        verification_step = PKPECuratedTablesVerificationStep(
            llm=self.llm,
            pmid=pmid,
            domain=self._get_domain(),
        )
        correction_step = PKPECuratedTablesCorrectionStep(
            llm=self.llm,
            pmid=pmid,
            domain=self._get_domain(),
        )
        graph = StateGraph(PKPECurationWorkflowState)
        graph.add_node("execution_step", execution_step.execute)
        graph.add_node("verification_step", verification_step.execute)
        graph.add_node("correction_step", correction_step.execute)
        graph.add_edge(START, "execution_step")
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
            config={"max_recursion_limit": MAX_STEP_COUNT},
            stream_mode="values",
        ):
            continue
        return s

    def run(self, pmid: str) -> tuple[bool, str | None, str | None, str | None]:
        self.print_step(step_name=f"Running {self.task_name} for pmid-{pmid}")
        state = self._run_workflow(pmid)
        correct = state["final_answer"] if state["final_answer"] is not None else False
        curated_table = state["curated_table"] if "curated_table" in state else None
        explanation = state["explanation"] if "explanation" in state else None
        suggested_fix = state["suggested_fix"] if "suggested_fix" in state else None
        return correct, curated_table, explanation, suggested_fix

        
