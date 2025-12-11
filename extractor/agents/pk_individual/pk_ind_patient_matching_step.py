import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, get_reasoning_process, increase_token_usage
from extractor.agents.pk_individual.pk_ind_common_step import (
    PKIndCommonStep,
)
from extractor.agents.pk_individual.pk_ind_patient_matching_agent import (
    get_matching_patient_prompt,
    MatchedPatientResult,
    post_process_validate_matched_patients,
    try_fix_error_matched_patients,
)


class PatientMatchingAutomaticStep(PKIndCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Patient Matching (Automatic)"
        self.end_title = "Completed Patient Matching"

    def execute_directly(self, state):
        patient_list = []
        md_table_list = state["md_table_list"]
        md_table_patient_refined = state["md_table_patient_refined"]

        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            df_expanded = pd.concat(
                [markdown_to_dataframe(md_table_patient_refined)] * row_num,
                ignore_index=True,
            )
            patient_list.append(dataframe_to_markdown(df_expanded))

        return None, patient_list, {**DEFAULT_TOKEN_USAGE}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["patient_list"] = processed_res
            self._step_output(state, step_output="Result (patient_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)


class PatientMatchingAgentStep(PKIndCommonStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Patient Matching (Agent)"
        self.end_title = "Completed Patient Matching"

    def _validate_matched_patients(self, matched_row_indices: list[int], df_table_patient: pd.DataFrame):
        expected_rows = df_table_patient.shape[0]
        if len(matched_row_indices) <= expected_rows:
            return matched_row_indices
        
        return [ix for ix in matched_row_indices if ix < expected_rows]

    def execute_directly(self, state):
        patient_list = []
        md_table_list = state["md_table_list"]
        md_table_patient = state["md_table_patient"]
        md_table_patient_refined = state["md_table_patient_refined"]
        df_table_patient_refined = markdown_to_dataframe(md_table_patient_refined)
        md_table_aligned = state["md_table_aligned"]
        llm = state["llm"]
        llm2 = state["llm2"]
        caption = state["caption"]
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        round = 0
        for md in md_table_list:
            round += 1
            self._step_output(f"Trial {round}")
            system_prompt = get_matching_patient_prompt(
                md_table_aligned, md, md_table_patient, caption
            )
            previous_errors_prompt = self._get_previous_errors_prompt(state)
            system_prompt = system_prompt + previous_errors_prompt
            instruction_prompt = INSTRUCTION_PROMPT
            agent = self.get_agent(state) # PKIndCommonAgent(llm=llm, llm2=llm2)
            result = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=MatchedPatientResult,
                try_fix_error=try_fix_error_matched_patients,
                post_process=post_process_validate_matched_patients,
                md_table=md,
            )
            res: MatchedPatientResult = result[0]
            processed_res = result[1]
            token_usage = result[2]
            reasoning_process = get_reasoning_process(result)
            self._step_output(
                state,
                step_reasoning_process=reasoning_process,
            )
            patient_match_list: list[int] = processed_res
            df_table_patient = df_table_patient_refined.copy()
            df_table_patient = pd.concat(
                [
                    df_table_patient,
                    pd.DataFrame(
                        [
                            {
                                "Patient ID": "ERROR",
                                "Population": "ERROR",
                                "Pregnancy stage": "ERROR",
                                "Pediatric/Gestational age": "ERROR",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            patient_match_list = self._validate_matched_patients(patient_match_list, df_table_patient)
            df_table_patient_reordered = df_table_patient.iloc[
                patient_match_list
            ].reset_index(drop=True)
            patient_list.append(dataframe_to_markdown(df_table_patient_reordered))
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return (
            MatchedPatientResult(
                reasoning_process="",
                matched_row_indices=[],
            ),
            patient_list,
            total_token_usage,
        )

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["patient_list"] = processed_res
            self._step_output(state, step_output="Result (patient_list):")
            self._step_output(state, step_output=str(processed_res))
        return super().leave_step(state, res, processed_res, token_usage)
