from pydantic import Field
import logging

from extractor.agents.common_agent.common_agent import RetryException
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table, get_reasoning_process, increase_token_usage
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult
from extractor.agents.pk_individual.pk_ind_common_step import PKIndCommonAgentStep
from extractor.agents.pk_individual.pk_ind_workflow_utils import PKIndWorkflowState

logger = logging.getLogger(__name__)

CHECK_PATIENT_ID_SYSTEM_PROMPT = """
You are a pharmacokinetics (PK) expert.
You are given a table containing pharmacokinetics (PK) data:
{processed_md_table}
Here is the table caption:
{caption}

---

### **TASK:**
Carefully analyze the table, **row by row and column by column**, and follow these steps:
(1) Check if the table contains patient ID.
(2) If the table contains patient ID, return "True".
(3) If the table does not contain patient ID, try to infer the individual patient ID for each row, return "True" if you can infer the patient ID for each row, otherwise return "False".

---

### **OUTPUT FORMAT:**
The output must **exactly match** the following format:
{{
    "patient_id": <True or False>
}}
"""

INFER_PATIENT_ID_SYSTEM_PROMPT = """


You are a pharmacokinetics (PK) expert.  
You are given a table containing pharmacokinetics (PK) data:

{processed_md_table}

Here is the table caption:  
{caption}

Here is the full text:  
{full_text}

---

### **TASK**
Infer the **Patient ID** for each row in the given table by carefully analyzing the full text and caption.  

- If the study describes a **single patient**, assign `1` to all rows.  
- If the study describes **multiple patients/cases**, assign IDs consistently based on the text (e.g., Case 1 → `1`, Case 2 → `2`, etc.).  
- If the patient ID cannot be determined for a row, return `"N/A"` for that row.  

---

### **OUTPUT FORMAT**
The output **must exactly match** the following format:
{{
    "reasoning_process": <reasoning_process>,
    "patient_ids": [patient_id_0, patient_id_1, ..., patient_id_N]
}}

#### Examples:

{{
    "reasoning_process": "balahbalah",
    "patient_ids": [1, 1, 1, 1, 1]
}}


{{
    "reasoning_process": "balahbalah",
    "patient_ids": [1, 2, 3, 4, 5]
}}


{{
    "reasoning_process": "balahbalah",
    "patient_ids": [1, "N/A", 2]
}}

---

### **Important Rules**
- The number of IDs must match the number of rows in the input table.  


"""

class CheckPatientIDResult(PKIndCommonAgentResult):
    patient_id: bool = Field(
        description="Bool value, representing whether the table contains patient ID."
    )

class InferPatientIDResult(PKIndCommonAgentResult):
    patient_ids: list[str] = Field(
        description="List of patient IDs for each row in the table."
    )

class PKIndPreprocessStep(PKIndCommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Preprocessing"
        self.end_title = "Completed to Preprocess"

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        caption = state["caption"]
        full_text = state["full_text"]
        return INFER_PATIENT_ID_SYSTEM_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
            full_text=full_text,
        )

    def get_schema(self):
        return InferPatientIDResult

    def get_instruction_prompt(self, state):
        return INSTRUCTION_PROMPT

    def get_post_processor_and_kwargs(self, state):
        def post_process(res: InferPatientIDResult):
            md_table = state["md_table"]
            df_table = markdown_to_dataframe(md_table)
            patient_ids = res.patient_ids
            if len(patient_ids) != df_table.shape[0]:
                error_msg = f"""Wrong answer example:
{patient_ids}
Why it's wrong:
Mismatch: Expected {df_table.shape[0]} rows, but got {len(patient_ids)} extracted patient IDs."""
                logger.error(error_msg)
                raise RetryException(error_msg)
            return patient_ids
        return post_process, None

    def execute_directly(self, state: PKIndWorkflowState):
        # First check if the table contains patient ID
        md_table = state["md_table"]
        caption = state["caption"]
        self._step_output(state, step_output=f"Caption: \n{caption}")
        self._step_output(state, step_output=f"md_able: \n{md_table}")
        system_prompt = CHECK_PATIENT_ID_SYSTEM_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
        )
        agent = self.get_agent(state['llm']) # PKIndCommonAgent(llm=state["llm"])
        result = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=INSTRUCTION_PROMPT,
            schema=CheckPatientIDResult,
        )
        res: CheckPatientIDResult = result[0]
        cur_token_usage = result[2]
        reasoning_process = get_reasoning_process(result)
        self._step_output(state, step_output=f"Reasoning:\n{reasoning_process}")
        self._step_output(state, step_output="Result (patient_id):")
        self._step_output(state, step_output=res.patient_id)
        if res.patient_id:
            # As it has already included patient ID, we don't need to infer the patient ID for each row
            # So we return None as the processed_res (processed_res is the patient IDs for each row)
            return res, None, cur_token_usage

        # We need to infer the patient ID for each row
        result = super().execute_directly(state)
        res: InferPatientIDResult = result[0]
        processed_res = result[1]
        token_usage = {**cur_token_usage} if cur_token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        cur_token_usage = result[2]
        reasoning_process = get_reasoning_process(result)
        self._step_output(state, step_output=f"Reasoning:\n{reasoning_process}")
        self._step_output(state, step_output="Result (patient_ids):")
        self._step_output(state, step_output=processed_res)
        token_usage = increase_token_usage(token_usage, cur_token_usage)

        return res, processed_res, token_usage

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is None: # The table does contain patient ID, just return
            return super().leave_step(state, res, processed_res, token_usage)
        patient_ids: list[str] = processed_res
        md_table = state["md_table"]
        df_table = markdown_to_dataframe(md_table)
        df_table.insert(0, "Patient ID", patient_ids)
        state["md_table"] = dataframe_to_markdown(df_table)
        return super().leave_step(state, res, processed_res, token_usage)