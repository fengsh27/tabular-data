
import time
from typing import Callable, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe, single_html_table_to_markdown
from extractor.agents.pk_summary.pk_sum_assembly_step import AssemblyStep
from extractor.agents.pk_summary.pk_sum_drug_matching_step import DrugMatchingAgentStep, DrugMatchingAutomaticStep
from extractor.agents.pk_summary.pk_sum_header_categorize_step import HeaderCategorizeStep
from extractor.agents.pk_summary.pk_sum_individual_data_del_step import IndividualDataDelStep
from extractor.agents.pk_summary.pk_sum_param_type_align_step import ParametertypeAlignStep
from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_step import ExtractParamTypeAndUnitStep
from extractor.agents.pk_summary.pk_sum_param_value_step import ParameterValueExtractionStep
from extractor.agents.pk_summary.pk_sum_patient_info_refine_step import PatientInfoRefinementStep
from extractor.agents.pk_summary.pk_sum_patient_info_step import PatientInfoExtractionStep
from extractor.agents.pk_summary.pk_sum_drug_info_step import DrugInfoExtractionStep

from extractor.agents.pk_summary.pk_sum_patient_matching_step import PatientMatchingAgentStep, PatientMatchingAutomaticStep
from extractor.agents.pk_summary.pk_sum_row_cleanup_step import RowCleanupStep
from extractor.agents.pk_summary.pk_sum_split_by_col_step import SplitByColumnsStep
from extractor.agents.pk_summary.pk_sum_time_unit_step import TimeExtractionStep
from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState

logger = logging.getLogger(__name__)

class PKSumWorkflow:
    """ pk summary workflow """
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def build(self):

        def select_drug_matching_step(state: PKSumWorkflowState):
            md_table_drug = state['md_table_drug']
            df = markdown_to_dataframe(md_table_drug)
            need_drug_matching = not df.shape[0] == 1
            return 'drug_matching_agent_step' if need_drug_matching \
                else 'drug_matching_automatic_step'
        
        def select_patient_matching_step(state: PKSumWorkflowState):
            md_table_patient = state['md_table_patient']
            df = markdown_to_dataframe(md_table_patient)
            need_patient_matching = not df.shape[0] == 1
            return 'patient_matching_agent_step' if need_patient_matching \
                else 'patient_matching_automatic_step'

        drug_info_step = DrugInfoExtractionStep()
        patient_info_step = PatientInfoExtractionStep()
        patient_info_refined_step = PatientInfoRefinementStep()
        individual_data_del_step = IndividualDataDelStep()
        param_type_align_step = ParametertypeAlignStep()
        header_categorize_step = HeaderCategorizeStep()
        split_by_col_step = SplitByColumnsStep()
        type_unit_extract_step = ExtractParamTypeAndUnitStep()
        drug_matching_automatic_step = DrugMatchingAutomaticStep()
        drug_matching_agent_step = DrugMatchingAgentStep()
        patient_matching_automatic_step = PatientMatchingAutomaticStep()
        patient_matching_agent_step = PatientMatchingAgentStep()
        param_value_step = ParameterValueExtractionStep()
        assembly_step = AssemblyStep()
        row_cleanup_step = RowCleanupStep()
        time_unit_step = TimeExtractionStep()

        graph = StateGraph(PKSumWorkflowState)
        graph.add_node('drug_info_step', drug_info_step.execute)
        graph.add_node('patient_info_step', patient_info_step.execute)
        graph.add_node('patient_info_refined_step', patient_info_refined_step.execute)
        graph.add_node('individual_data_del_step', individual_data_del_step.execute)
        graph.add_node('param_type_align_step', param_type_align_step.execute)
        graph.add_node('header_categorize_step', header_categorize_step.execute)
        graph.add_node('split_by_col_step', split_by_col_step.execute)
        graph.add_node('type_unit_extract_step', type_unit_extract_step.execute)
        graph.add_node('drug_matching_automatic_step', drug_matching_automatic_step.execute)
        graph.add_node('drug_matching_agent_step', drug_matching_agent_step.execute)
        graph.add_node('patient_matching_automatic_step', patient_matching_automatic_step.execute)
        graph.add_node('patient_matching_agent_step', patient_matching_agent_step.execute)
        graph.add_node('param_value_step', param_value_step.execute)
        graph.add_node('assembly_step', assembly_step.execute)
        graph.add_node('row_cleanup_step', row_cleanup_step.execute)
        graph.add_node('time_unit_step', time_unit_step.execute)

        graph.add_edge(START, 'drug_info_step')
        graph.add_edge('drug_info_step', 'patient_info_step')
        graph.add_edge('patient_info_step', 'patient_info_refined_step')
        graph.add_edge('patient_info_refined_step', 'individual_data_del_step')
        graph.add_edge('individual_data_del_step', 'param_type_align_step')
        graph.add_edge('param_type_align_step', 'header_categorize_step')
        graph.add_edge('header_categorize_step', 'split_by_col_step')
        graph.add_edge('split_by_col_step', 'type_unit_extract_step')
        graph.add_conditional_edges(
            'type_unit_extract_step', 
            select_drug_matching_step, 
            {'drug_matching_automatic_step', 'drug_matching_agent_step'}
        )
        
        graph.add_conditional_edges(
            'drug_matching_automatic_step', 
            select_patient_matching_step, 
            {'patient_matching_automatic_step', 'patient_matching_agent_step'}
        )
        graph.add_conditional_edges(
            'drug_matching_agent_step', 
            select_patient_matching_step, 
            {'patient_matching_automatic_step', 'patient_matching_agent_step'}
        )        
        graph.add_edge('patient_matching_automatic_step', 'param_value_step')
        graph.add_edge('patient_matching_agent_step', 'param_value_step')
        graph.add_edge('param_value_step', 'assembly_step')
        graph.add_edge('assembly_step', 'row_cleanup_step')
        graph.add_edge('row_cleanup_step', 'time_unit_step')

        self.graph = graph.compile()
        # display(Image(self.graph.get_graph().draw_mermaid_png()))            
        # logger.info(self.graph.get_graph().draw_ascii())
        # print(self.graph.get_graph().draw_ascii())

    def go_md_table(
        self, 
        md_table: str, 
        caption_and_footnote: str, 
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
            }, 
            config=config,
            stream_mode="values",
        ):
            if sleep_time is not None:
                time.sleep(sleep_time)
            print(s)

        df_combined = s['df_combined']
        column_mapping = {
            "Main value": "Parameter value",
            "Statistics type": "Parameter statistic",
            "Lower bound": "Lower limit",
            "Upper bound": "High limit",
        }
        df_combined = df_combined.rename(columns=column_mapping)
        return df_combined

    def go(
        self, 
        html_content: str, 
        caption_and_footnote: str, 
        step_callback: Callable | None = None,
        sleep_time: float | None = None,
    ):
        md_table = single_html_table_to_markdown(html_content)
        return self.go_md_table(md_table, caption_and_footnote, step_callback, sleep_time)

        

        

            



