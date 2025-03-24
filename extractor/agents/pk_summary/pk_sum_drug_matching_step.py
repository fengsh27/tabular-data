
from typing import List
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_summary.pk_sum_common_agent import PKSumCommonAgent
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_drug_matching_agent import (
    get_matching_drug_prompt,
    MatchedDrugResult,
    post_process_validate_matched_rows,
)

class DrugMatchingStep(PKSumCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Drug Matching"
        self.end_title = "Completed Drug Matching"

    def get_system_prompt(self, state):
        """ we won't use this method in this step """
        return ""
    
    def get_schema(self):
        return MatchedDrugResult
    
    def get_post_processor_and_kwargs(self, state):
        """ we won't use this method in this step """
        return post_process_validate_matched_rows, None

    def execute_directly(self, state):
        drug_list = []
        round = 0
        md_table_drug = state['md_table_drug']
        md_table_list = state['md_table_list']
        need_match_drug = not markdown_to_dataframe(
            md_table_drug
        ).shape[0] == 1
        total_token_usage = DEFAULT_TOKEN_USAGE
        if not need_match_drug:
            for md in md_table_list:
                df = markdown_to_dataframe(md)
                row_num = df.shape[0]
                df_expanded = pd.concat([markdown_to_dataframe(md_table_drug)] * row_num, ignore_index=True)
                drug_list.append(dataframe_to_markdown(df_expanded))
        else:
            llm = state['llm']
            caption = state['caption']
            md_table_aligned = state['md_table_aligned']
            for md in md_table_list:
                round += 1
                system_prompt = get_matching_drug_prompt(
                    md_table_aligned, md, md_table_drug, caption
                )
                instruction_prompt = self.get_instruction_prompt(state)
                agent = PKSumCommonAgent(llm=llm)
                res, processed_res, token_usage = agent.go(
                    system_prompt=system_prompt,
                    instruction_prompt=instruction_prompt,
                    schema=MatchedDrugResult,
                    post_process=post_process_validate_matched_rows,
                    md_table=md,
                )
                drug_match_list: List[int] = processed_res
                df_table_drug = markdown_to_dataframe(md_table_drug)
                df_table_drug = pd.concat([
                    df_table_drug, pd.DataFrame([{
                        'Drug name': 'ERROR', 'Analyte': 'ERROR', 'Specimen': 'ERROR'
                    }])
                ], ignore_index=True)
                df_table_drug_reordered = df_table_drug.iloc[drug_match_list].reset_index(drop=True)
                drug_list.append(dataframe_to_markdown(df_table_drug_reordered))
                total_token_usage = increase_token_usage(total_token_usage, token_usage)
        
        return MatchedDrugResult(
            reasoning_process="",
            matched_row_indices=[]
        ), drug_list, total_token_usage
    
    def leave_step(self, state, res, processed_res = None, token_usage = None):
        if processed_res is not None:
            state["drug_list"] = processed_res
        return super().leave_step(state, res, processed_res, token_usage)
            
                
                


