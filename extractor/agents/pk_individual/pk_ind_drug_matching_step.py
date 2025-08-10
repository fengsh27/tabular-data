import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgent
from extractor.agents.pk_individual.pk_ind_common_step import (
    PKIndCommonStep,
)
from extractor.agents.pk_individual.pk_ind_drug_matching_agent import (
    get_matching_drug_prompt,
    MatchedDrugResult,
    post_process_validate_matched_rows,
)


class DrugMatchingAutomaticStep(PKIndCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Drug Matching (Automatic)"
        self.end_title = "Completed Drug Matching"

    def execute_directly(self, state):
        drug_list = []
        md_table_list = state["md_table_list"]
        md_table_drug = state["md_table_drug"]
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            df_expanded = pd.concat(
                [markdown_to_dataframe(md_table_drug)] * row_num, ignore_index=True
            )
            drug_list.append(dataframe_to_markdown(df_expanded))

        return None, drug_list, {**DEFAULT_TOKEN_USAGE}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["drug_list"] = processed_res
            self._step_output(state, step_output="Result (drug_list):")
            self._step_output(state, step_output=str(processed_res))

        return super().leave_step(state, res, processed_res, token_usage)


class DrugMatchingAgentStep(PKIndCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Drug Matching (Agent)"
        self.end_title = "Completed Drug Matching"

    def execute_directly(self, state):
        drug_list = []
        round = 0
        md_table_drug = state["md_table_drug"]
        md_table_list = state["md_table_list"]
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        llm = state["llm"]
        caption = state["caption"]
        md_table_aligned = state["md_table_aligned"]
        for md in md_table_list:
            round += 1
            self._step_output(state, step_output="=" * 64)
            self._step_output(state, f"Trial {round}")
            system_prompt = get_matching_drug_prompt(
                md_table_aligned, md, md_table_drug, caption
            )
            previous_errors_prompt = self._get_previous_errors_prompt(state)
            system_prompt = system_prompt + previous_errors_prompt
            agent = PKIndCommonAgent(llm=llm)
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=INSTRUCTION_PROMPT,
                schema=MatchedDrugResult,
                post_process=post_process_validate_matched_rows,
                md_table1=md,
                md_table2=md_table_drug,
            )
            self._step_output(
                state,
                step_reasoning_process=res.reasoning_process if res is not None else "",
            )
            drug_match_list: list[int] = processed_res
            df_table_drug = markdown_to_dataframe(md_table_drug)
            df_table_drug = pd.concat(
                [
                    df_table_drug,
                    pd.DataFrame(
                        [
                            {
                                "Drug name": "ERROR",
                                "Analyte": "ERROR",
                                "Specimen": "ERROR",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            df_table_drug_reordered = df_table_drug.iloc[drug_match_list].reset_index(
                drop=True
            )
            drug_list.append(dataframe_to_markdown(df_table_drug_reordered))
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return None, drug_list, total_token_usage

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["drug_list"] = processed_res
            self._step_output(state, step_output="Result (drug_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
