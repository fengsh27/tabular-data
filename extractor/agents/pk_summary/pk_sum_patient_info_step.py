
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import (
    display_md_table, 
    extract_integers,
)
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_patient_info_agent import (
    PATIENT_INFO_PROMPT,
    PatientInfoResult,
    post_process_convert_patient_info_to_md_table,
)

class PatientInfoExtractionStep(PKSumCommonAgentStep):
    """ Step to Extract Patient Information """
    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Population Information"
        self.end_title = "Completed to Extract Population Information"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        caption = state["caption"]
        int_list = extract_integers(md_table + caption)
        return PATIENT_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
            int_list=int_list,
        )
    
    def leave_step(self, state, res, processed_res = None, token_usage = None):
        if processed_res is not None:
            state["md_table_patient"] = processed_res
            self._step_output(state, step_output="Result (md_table_patient):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)
        
    def get_post_processor_and_kwargs(self, state):
        return post_process_convert_patient_info_to_md_table, None
    
    def get_schema(self):
        return PatientInfoResult


