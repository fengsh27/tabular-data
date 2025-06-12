import time
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)

from extractor.agents.pe_study_info.pe_study_info_workflow_utils import PEStudyInfoWorkflowState
from extractor.agents.pe_study_info.pe_study_info_design_info_step import DesignInfoExtractionStep
from extractor.agents.pe_study_info.pe_study_info_design_info_refine_step import DesignInfoRefinementStep
from extractor.agents.pe_study_info.pe_study_info_assembly_step import AssemblyStep
from extractor.agents.pe_study_info.pe_study_info_row_cleanup_step import RowCleanupStep

logger = logging.getLogger(__name__)


class PEStudyInfoWorkflow:
    """pk summary workflow"""

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        design_info_step = DesignInfoExtractionStep()
        design_info_refined_step = DesignInfoRefinementStep()
        assembly_step = AssemblyStep()
        row_cleanup_step = RowCleanupStep()
        #
        graph = StateGraph(PEStudyInfoWorkflowState)
        graph.add_node("design_info_step", design_info_step.execute)
        graph.add_node("design_info_refined_step", design_info_refined_step.execute)
        graph.add_node("assembly_step", assembly_step.execute)
        graph.add_node("row_cleanup_step", row_cleanup_step.execute)
        #
        graph.add_edge(START, "design_info_step")
        graph.add_edge("design_info_step", "design_info_refined_step")
        graph.add_edge("design_info_refined_step", "assembly_step")
        graph.add_edge("assembly_step", "row_cleanup_step")

        self.graph = graph.compile()
        # display(Image(self.graph.get_graph().draw_mermaid_png()))
        # logger.info(self.graph.get_graph().draw_ascii())
        # print(self.graph.get_graph().draw_ascii())

    def go_full_text(
        self,
        title: str,
        full_text: str,
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
    ):
        config = {"recursion_limit": 500}

        for s in self.graph.stream(
            input={
                "title": title,
                "full_text": full_text,
                "llm": self.llm,
                "step_callback": step_callback,
            },
            config=config,
            stream_mode="values",
        ):
            if sleep_time is not None:
                time.sleep(sleep_time)
            print(s)

        df_combined = s["df_combined"]
        column_mapping = {
            "Source text": "Note",
        }
        df_combined = df_combined.rename(columns=column_mapping)
        # df_combined = markdown_to_dataframe(s["md_table_design_refined"])
        return df_combined

    def go(
        self,
        title: str,
        full_text: str,
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
    ):
        return self.go_full_text(
            title=title,
            full_text=full_text,
            step_callback=step_callback,
            sleep_time=sleep_time,
        )
