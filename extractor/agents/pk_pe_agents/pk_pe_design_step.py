from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import logging

from extractor.agents.common_agent.common_agent import CommonAgent, RetryException
from extractor.agents.common_agent.common_step import CommonStep,CommonState
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import (
    PKIndividualTablesCurationTool, 
    PKSummaryTablesCurationTool,
    PKPopulationIndividualCurationTool,
    PKPopulationSummaryCurationTool,
    PEStudyOutcomeCurationTool,
)
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.constants import COT_USER_INSTRUCTION, PipelineTypeEnum

logger = logging.getLogger(__name__)

## -------------------- Helper Functions --------------------
def get_tools_descriptions() -> str:
    return f"""
{PipelineTypeEnum.PK_SUMMARY.value}: {PKSummaryTablesCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_INDIVIDUAL.value}: {PKIndividualTablesCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_SPEC_SUMMARY.value}: This tool is used to curate the PK specimen summary data from the source paper.
{PipelineTypeEnum.PK_DRUG_SUMMARY.value}: This tool is used to curate the PK drug summary data from the source paper.
{PipelineTypeEnum.PK_POPU_SUMMARY.value}: {PKPopulationSummaryCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_SPEC_INDIVIDUAL.value}: This tool is used to curate the PK specimen individual data from the source paper.
{PipelineTypeEnum.PK_DRUG_INDIVIDUAL.value}: This tool is used to curate the PK drug individual data from the source paper.
{PipelineTypeEnum.PK_POPU_INDIVIDUAL.value}: {PKPopulationIndividualCurationTool.get_tool_description()}
{PipelineTypeEnum.PE_STUDY_INFO.value}: This tool is used to curate the PE study info data from the source paper.
{PipelineTypeEnum.PE_STUDY_OUTCOME.value}: {PEStudyOutcomeCurationTool.get_tool_description()}
"""

class PKPEDesignStepResult(BaseModel):
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")
    pipeline_tools: list[str] = Field(description="A list of pipeline tool names")

PKPE_DESIGN_SYSTEM_PROMPT = """
You are a biomedical research assistant with expertise in pharmacology, specifically pharmacokinetics (PK) and pharmacoepidemiology (PE).  
Your goal is to curate data from the given paper using the available pipeline tools.

---

### **Reference Definitions**  
- **Pharmacokinetics (PK):** Study of how a drug is absorbed, distributed, metabolized, and excreted. PK parameters may include clearance, half-life, AUC, Cmax, volume of distribution, and bioavailability. Typical designs measure drug concentrations in plasma/tissues over time.  
- **Pharmacoepidemiology (PE):** Study of drug use and effects in large populations, often using observational data (e.g., insurance claims, EHRs). Focus areas include safety, utilization, adherence, effectiveness, risk–benefit, and post-marketing surveillance.  
- **Clinical Trials:** A type of study in which participants are randomly assigned to one or more treatment groups to evaluate the safety and effectiveness of a medical intervention.

---

### **Pipeline Tools**  
You can only choose from the following tools:  
{tools_descriptions}  

---

### **Task**
You will be given paper title and full text, ant the paper type (PK, PE/Clinical Trial, Both, Neither), your task is:
1. Choose the appropriate pipeline tool(s) to use for the given paper.  

---

### **Workflow Reminder**
- **Design (current stage):** Choose the appropriate pipeline tool(s) to use for the given paper.  
- **Execution:** Run the steps to curate data.  
- **Verification:** Check correctness of curated data.  
- **Correction:** Fix any detected errors.  

---

### **Output Format (must follow exactly)**  
**Pipeline Tools**: [list of pipeline tool names]

---

### **Input**  

- **Title**: 
{paper_title}

- **Paper Type**: 
{paper_type}

- **Full Text**: 
{full_text}  

---

"""

def post_process_design(res: PKPEDesignStepResult) -> PKPEDesignStepResult:
    all_tools = [member.value for member in PipelineTypeEnum]
    for tool in res.pipeline_tools:
        if tool not in all_tools:
            raise RetryException(f"Invalid tool: {tool}")
    return res

class PKPEDesignStep(CommonStep):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)
        self.step_name = "PK PE Design Step"
        self.tools_descriptions = get_tools_descriptions()

    def _execute_directly(self, state: PKPECurationWorkflowState) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        system_prompt = PKPE_DESIGN_SYSTEM_PROMPT.format(
            paper_title=state["paper_title"],
            full_text=state["full_text"],
            paper_type=state["paper_type"].value,
            tools_descriptions=self.tools_descriptions,
        )
        instruction_prompt = COT_USER_INSTRUCTION

        agent = CommonAgent(llm=self.llm)

        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            schema=PKPEDesignStepResult,
            post_process=post_process_design,
        )
        if reasoning_process is None:
            reasoning_process = res.reasoning_process if hasattr(res, "reasoning_process") else "N / A"
        self._print_step(state, step_output=reasoning_process)
        res: PKPEDesignStepResult = res
        state["pipeline_tools"] = res.pipeline_tools

        return state, token_usage