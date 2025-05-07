import time
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)
from extractor.agents.pk_specimen_summary.pk_spec_sum_assembly_step import AssemblyStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_patient_info_refine_step import PatientInfoRefinementStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_specimen_info_step import SpecimenInfoExtractionStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_row_cleanup_step import RowCleanupStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_time_unit_step import TimeExtractionStep
from extractor.agents.pk_specimen_summary.pk_spec_sum_workflow_utils import PKSpecSumWorkflowState

logger = logging.getLogger(__name__)


class PKSpecSumWorkflow:
    """pk summary workflow"""

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        specimen_info_step = SpecimenInfoExtractionStep()
        patient_info_refined_step = PatientInfoRefinementStep()
        time_unit_step = TimeExtractionStep()
        assembly_step = AssemblyStep()
        row_cleanup_step = RowCleanupStep()
        #
        graph = StateGraph(PKSpecSumWorkflowState)
        graph.add_node("specimen_info_step", specimen_info_step.execute)
        graph.add_node("patient_info_refined_step", patient_info_refined_step.execute)
        graph.add_node("time_unit_step", time_unit_step.execute)
        graph.add_node("assembly_step", assembly_step.execute)
        graph.add_node("row_cleanup_step", row_cleanup_step.execute)
        #
        graph.add_edge(START, "specimen_info_step")
        graph.add_edge("specimen_info_step", "patient_info_refined_step")
        graph.add_edge("patient_info_refined_step", "time_unit_step")
        graph.add_edge("time_unit_step", "assembly_step")
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
            "Population N": "Subject N",
            "Source text": "Time note",
        }
        df_combined = df_combined.rename(columns=column_mapping)
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
