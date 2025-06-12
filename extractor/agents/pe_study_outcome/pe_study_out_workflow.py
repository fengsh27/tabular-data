import time
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)
# from extractor.agents.pe_study_outcome.pe_study_out_assembly_step import AssemblyStep
# from extractor.agents.pe_study_outcome.pe_study_out_drug_matching_step import (
#     DrugMatchingAgentStep,
#     DrugMatchingAutomaticStep,
# )
from extractor.agents.pe_study_outcome.pe_study_out_header_categorize_step import HeaderCategorizeStep
from extractor.agents.pe_study_outcome.pe_study_out_row_categorize_step import RowCategorizeStep
from extractor.agents.pe_study_outcome.pe_study_out_mapping_step import MappingStep
from extractor.agents.pe_study_outcome.pe_study_out_param_value_step import ParameterValueExtractionStep
from extractor.agents.pe_study_outcome.pe_study_out_row_cleanup_step import RowCleanupStep

from extractor.agents.pe_study_outcome.pe_study_out_workflow_utils import PEStudyOutWorkflowState

logger = logging.getLogger(__name__)


class PEStudyOutWorkflow:
    """pk summary workflow"""

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):
        # def select_drug_matching_step(state: PEStudyOutWorkflowState):
        #     md_table_drug = state["md_table_drug"]
        #     df = markdown_to_dataframe(md_table_drug)
        #     need_drug_matching = not df.shape[0] == 1
        #     return (
        #         "drug_matching_agent_step"
        #         if need_drug_matching
        #         else "drug_matching_automatic_step"
        #     )
        #
        # def select_patient_matching_step(state: PEStudyOutWorkflowState):
        #     md_table_patient = state["md_table_patient"]
        #     df = markdown_to_dataframe(md_table_patient)
        #     need_patient_matching = not df.shape[0] == 1
        #     return (
        #         "patient_matching_agent_step"
        #         if need_patient_matching
        #         else "patient_matching_automatic_step"
        #     )

        # patient_info_step = PatientInfoExtractionStep()
        header_categorize_step = HeaderCategorizeStep()
        row_categorize_step = RowCategorizeStep()
        mapping_step = MappingStep()
        param_value_step = ParameterValueExtractionStep()
        row_clean_up_step = RowCleanupStep()

        graph = StateGraph(PEStudyOutWorkflowState)
        # graph.add_node("patient_info_step", patient_info_step.execute)
        graph.add_node("header_categorize_step", header_categorize_step.execute)
        graph.add_node("row_categorize_step", row_categorize_step.execute)
        graph.add_node("mapping_step", mapping_step.execute)
        graph.add_node("param_value_step", param_value_step.execute)
        graph.add_node("row_clean_up_step", row_clean_up_step.execute)

        # graph.add_edge(START, "patient_info_step")
        graph.add_edge(START, "header_categorize_step")
        graph.add_edge("header_categorize_step", "row_categorize_step")
        graph.add_edge("row_categorize_step", "mapping_step")
        graph.add_edge("mapping_step", "param_value_step")
        graph.add_edge("param_value_step", "row_clean_up_step")

        self.graph = graph.compile()
        # display(Image(self.graph.get_graph().draw_mermaid_png()))
        # logger.info(self.graph.get_graph().draw_ascii())
        # print(self.graph.get_graph().draw_ascii())

    def go_md_table(
        self,
        md_table: str,
        caption_and_footnote: str,
        title: str | None = None,
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
    ):
        config = {"recursion_limit": 500}

        for s in self.graph.stream(
            input={
                "md_table": md_table,
                "caption": caption_and_footnote,
                "llm": self.llm,
                "step_callback": step_callback,
                "title": title if title is not None else "",
            },
            config=config,
            stream_mode="values",
        ):
            if sleep_time is not None:
                time.sleep(sleep_time)
            print(s)

        df_combined = s["df_combined"]
        # import pandas as pd
        # df_combined = pd.DataFrame([["a", 1], ["b", 2]], columns=["col1", "col2"])
        return df_combined

    def go(
        self,
        html_content: str,
        caption_and_footnote: str,
        title: str | None = None,
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
    ):
        md_table = single_html_table_to_markdown(html_content)
        return self.go_md_table(
            md_table=md_table, 
            caption_and_footnote=caption_and_footnote,
            title=title,
            step_callback=step_callback,
            sleep_time=sleep_time,
        )
