from typing import Any
from extractor.agents.agent_utils import (
    display_md_table,
    extract_integers,
)
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_patient_info_agent import (
    PATIENT_INFO_PROMPT,
    PatientInfoResult,
    post_process_convert_patient_info_to_md_table,
)
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState


class PatientInfoExtractionStep(PKIndCommonAgentStep):
    """Step to Extract Patient Information"""

    def __init__(self):
        super().__init__()
        self.start_title = "Extracting Population Information"
        self.end_title = "Completed to Extract Population Information"
        self.infer_from_fulltext = False

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        caption = state["caption"]
        full_text = state["full_text"]
        full_text = full_text if full_text is not None else "N/A"
        int_list = extract_integers(md_table + caption)
        system_prompt = PATIENT_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
            int_list=int_list,
            full_text=full_text if self.infer_from_fulltext else "N/A",
        )
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt
    
    def get_instruction_prompt(self, state: PKIndWorkflowState):
        if not self.infer_from_fulltext:
            return super().get_instruction_prompt(state)
        return """### **Important Instructions:**
 * If you don't find any individual patient information (Patient ID) from the table, try to infer it from the full text.
   e.g., The study focuses on a single patient, so the Patient ID can be condsidered as "1".
 * If you can't find any individual patient information (Patient ID) from the full text, just return empty list, **do not** make up anything.
 * Before returning the final answer, please explain your thought process in detail.
 """

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["md_table_patient"] = processed_res
            self._step_output(state, step_output="Result (md_table_patient):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)

    def get_post_processor_and_kwargs(self, state):
        return post_process_convert_patient_info_to_md_table, None

    def get_schema(self):
        return PatientInfoResult

    def execute_directly(
        self,
        state: PKIndWorkflowState,
    ) -> tuple[PKIndCommonAgentResult, Any | None, dict | None]:
        res, processed_res, token_usage = super().execute_directly(state)
        if (processed_res is not None and len(processed_res) > 0) or self.infer_from_fulltext:
            return res, processed_res, token_usage

        self.infer_from_fulltext = True
        res, processed_res, token_usage = super().execute_directly(state)
        self.infer_from_fulltext = False
        return res, processed_res, token_usage
        
        
