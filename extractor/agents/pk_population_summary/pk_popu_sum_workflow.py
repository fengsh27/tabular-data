import time
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)
from extractor.agents.pk_population_summary.pk_popu_sum_assembly_step import AssemblyStep
from extractor.agents.pk_population_summary.pk_popu_sum_patient_info_refine_step import PatientInfoRefinementStep
from extractor.agents.pk_population_summary.pk_popu_sum_characteristic_info_step import CharacteristicInfoExtractionStep
from extractor.agents.pk_population_summary.pk_popu_sum_row_cleanup_step import RowCleanupStep
from extractor.agents.pk_population_summary.pk_popu_sum_characteristic_info_refine_step import CharacteristicInfoRefinementStep
from extractor.agents.pk_population_summary.pk_popu_sum_workflow_utils import PKPopuSumWorkflowState

logger = logging.getLogger(__name__)


class PKPopuSumWorkflow:
    """pk summary workflow"""

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        characteristic_info_step = CharacteristicInfoExtractionStep()
        patient_info_refined_step = PatientInfoRefinementStep()
        characteristic_info_refined_step = CharacteristicInfoRefinementStep()
        assembly_step = AssemblyStep()
        row_cleanup_step = RowCleanupStep()
        #
        graph = StateGraph(PKPopuSumWorkflowState)
        graph.add_node("characteristic_info_step", characteristic_info_step.execute)
        graph.add_node("patient_info_refined_step", patient_info_refined_step.execute)
        graph.add_node("characteristic_info_refined_step", characteristic_info_refined_step.execute)
        graph.add_node("assembly_step", assembly_step.execute)
        graph.add_node("row_cleanup_step", row_cleanup_step.execute)
        #
        graph.add_edge(START, "characteristic_info_step")
        graph.add_edge("characteristic_info_step", "patient_info_refined_step")
        graph.add_edge("patient_info_refined_step", "characteristic_info_refined_step")
        graph.add_edge("characteristic_info_refined_step", "assembly_step")
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
        previous_errors: str | None = None,
    ):
        previous_errors = previous_errors if previous_errors is not None else "N/A"
        config = {"recursion_limit": 500}

        for s in self.graph.stream(
            input={
                "title": title,
                "full_text": full_text,
                "llm": self.llm,
                "step_callback": step_callback,
                "previous_errors": previous_errors,
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
            "Source text": "Note",
            "Population characteristic": "Characteristic",
            "Characteristic sub-category": "Characteristic subcategory",
            "Unit": "Characteristic unit",
            "Main value": "Characteristic value",
        }
        df_combined = df_combined.rename(columns=column_mapping)

        # df_combined = markdown_to_dataframe(s["md_table_characteristic_refined"])
        return df_combined

    def go(
        self,
        title: str,
        full_text: str,
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
        previous_errors: str | None = None,
    ):
        return self.go_full_text(
            title=title,
            full_text=full_text,
            step_callback=step_callback,
            sleep_time=sleep_time,
            previous_errors=previous_errors,
        )
