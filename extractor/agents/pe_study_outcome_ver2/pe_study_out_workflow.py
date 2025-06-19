import time
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)

from extractor.agents.pe_study_outcome_ver2.pe_study_out_numeric_retain_step import NumericRetainStep
from extractor.agents.pe_study_outcome_ver2.pe_study_out_param_value_step import ParameterValueExtractionStep
from extractor.agents.pe_study_outcome_ver2.pe_study_out_study_info_step import StudyInfoExtractionStep
from extractor.agents.pe_study_outcome_ver2.pe_study_out_row_cleanup_step import RowCleanupStep

from extractor.agents.pe_study_outcome_ver2.pe_study_out_workflow_utils import PEStudyOutWorkflowState

logger = logging.getLogger(__name__)


class PEStudyOutWorkflow:
    """pk summary workflow"""

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):

        numeric_retain_step = NumericRetainStep()
        param_value_step = ParameterValueExtractionStep()
        study_info_step = StudyInfoExtractionStep()
        row_cleanup_step = RowCleanupStep()

        graph = StateGraph(PEStudyOutWorkflowState)
        graph.add_node("numeric_retain_step", numeric_retain_step.execute)
        graph.add_node("param_value_step", param_value_step.execute)
        graph.add_node("study_info_step", study_info_step.execute)
        graph.add_node("row_cleanup_step", row_cleanup_step.execute)

        # graph.add_edge(START, "patient_info_step")
        graph.add_edge(START, "numeric_retain_step")
        graph.add_edge("numeric_retain_step", "param_value_step")
        graph.add_edge("param_value_step", "study_info_step")
        graph.add_edge("study_info_step", "row_cleanup_step")

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
