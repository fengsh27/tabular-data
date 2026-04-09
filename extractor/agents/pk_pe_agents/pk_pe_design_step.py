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
from extractor.agents.pk_pe_agents.pk_pe_common_step import PKPECommonStep

logger = logging.getLogger(__name__)

## -------------------- Helper Functions --------------------
def get_tools_descriptions() -> str:
    return f"""
{PipelineTypeEnum.PK_SUMMARY.value}: {PKSummaryTablesCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_INDIVIDUAL.value}: {PKIndividualTablesCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_SPEC_SUMMARY.value}: This tool is used to curate the PK specimen summary data from full text in the source paper.
{PipelineTypeEnum.PK_DRUG_SUMMARY.value}: This tool is used to curate the PK drug summary data from full text in the source paper.
{PipelineTypeEnum.PK_POPU_SUMMARY.value}: {PKPopulationSummaryCurationTool.get_tool_description()}
{PipelineTypeEnum.PK_SPEC_INDIVIDUAL.value}: This tool is used to curate the PK specimen individual data from full text in the source paper.
{PipelineTypeEnum.PK_DRUG_INDIVIDUAL.value}: This tool is used to curate the PK drug individual data from full text in the source paper.
{PipelineTypeEnum.PK_POPU_INDIVIDUAL.value}: {PKPopulationIndividualCurationTool.get_tool_description()}
{PipelineTypeEnum.PE_STUDY_INFO.value}: This tool is used to curate the PE study info data from full text in the source paper.
{PipelineTypeEnum.PE_STUDY_OUTCOME.value}: {PEStudyOutcomeCurationTool.get_tool_description()}
"""

class PKPEDesignStepResult(BaseModel):
    # reasoning_process: str = Field(description="A concise explanation of the thought process or reasoning steps taken to reach a conclusion in 1-2 sentences.")
    pipeline_tools: list[str] = Field(description="A list of pipeline tool names")

PKPE_DESIGN_SYSTEM_PROMPT = """

## **System Role**

You are a biomedical research assistant with expertise in pharmacology, specifically:

* Pharmacokinetics (PK)
* Pharmacoepidemiology (PE)
* Clinical Trials (CT)

Your task is to **identify ALL applicable pipeline tools** for extracting data from a given paper.

---

## **Core Principle (CRITICAL)**

> **Tool selection is multi-label.**
> You MUST select **ALL tools whose definitions match ANY data present** in the paper.
>
> Tools are **NOT mutually exclusive**.
> Do NOT try to choose the “best” or “most specific” tool.
> Instead, **maximize coverage**.

---

## **Reference Definitions**

### **Pharmacokinetics (PK)**

Study of drug absorption, distribution, metabolism, and excretion.
Includes parameters such as: AUC, Cmax, clearance, half-life, volume of distribution, concentration-time data.

### **Pharmacoepidemiology (PE)**

Observational population-level studies (EHR, claims, safety, utilization, effectiveness).

### **Clinical Trials**

Interventional studies with assigned treatments to evaluate safety/efficacy.

---

## **Pipeline Tools**

| Tool Name                | Description                                                         |
| ------------------------ | ------------------------------------------------------------------- |
| pk_summary               | PK summary data (means, medians, AUC, Cmax, etc.)                   |
| pk_individual            | PK data where rows correspond to subject/case IDs                   |
| pk_specimen_summary      | PK summary data stratified by specimen (plasma, milk, tissue, etc.) |
| pk_drug_summary          | PK summary data stratified by drug/analyte                          |
| pk_population_summary    | Demographic/population-level summary data                           |
| pk_specimen_individual   | Individual PK data with specimen dimension                          |
| pk_drug_individual       | Individual PK data with drug/analyte dimension                      |
| pk_population_individual | Individual-level demographic data                                   |
| pe_study_info            | PE study design/info                                                |
| pe_study_outcome         | PE outcomes                                                         |

---

## **Operational Definitions (MUST FOLLOW)**

### **1. Individual Data**

> Data is **individual-level** if:

* Table rows are labeled by **subject ID / patient / volunteer / case**

✅ This includes:

* Per-subject averages (e.g., mean concentration per subject)
* Per-subject PK metrics (AUC, Cmax, ratios)

---

### **2. Summary Data**

> Data is **summary-level** if:

* It aggregates across subjects (mean, median, SD, CI)

---

### **3. Specimen-specific Data**

> Data explicitly involves specimen types:

* plasma, serum, milk, urine, tissue, etc.

---

### **4. Drug-specific Data**

> Data distinguishes:

* multiple drugs
* metabolites
* analytes

---

### **5. Population Data**

> Data includes:

* demographics (age, weight, sex, pregnancy stage, etc.)

---

## **Selection Rules (DETERMINISTIC)**

### Rule 1 — Domain Filtering

* If PK data exists → select PK tools
* If PE data exists → select PE tools
* If both → select both

---

### Rule 2 — Granularity (NON-EXCLUSIVE)

* If subject-labeled rows exist → select **individual tools**PKPE_DESIGN_SYSTEM_PROMPT
* If aggregated statistics exist → select **summary tools**
* If BOTH exist → select BOTH

---

### Rule 3 — Dimension Matching (NON-EXCLUSIVE)

For EACH applicable dimension, select matching tools:

| Dimension present       | Select          |
| ----------------------- | --------------- |
| Specimen                | pk_specimen_*   |
| Drug/analyte            | pk_drug_*       |
| Population/demographics | pk_population_* |

---

### Rule 4 — Combine Tools

> Final tool list = **union of all matched tools**

---

## **Common Pitfalls (IMPORTANT)**

❌ Do NOT:

* Choose only one tool
* Prefer “more specific” over “general”
* Ignore overlapping categories

---

## **Output Format**
Return the selected tools in the following exact format:
```
{{
  "pipeline_tools": [tool_name_1, tool_name_2, ...]
}}
```

---

## **Example (for clarity)**

If a table:

* has rows = patients
* reports mean concentration
* includes plasma & milk
* includes parent drug + metabolite

Then output MUST include:

```
{{
  "pipeline_tools": ["pk_summary", pk_individual, pk_specimen_summary, pk_specimen_individual, pk_drug_summary, pk_drug_individual]
}}
```

---

### **Input**
- **Title:**  
{paper_title}

- **Paper Type:**  
{paper_type}

- **Full Text (excluding tables):**  
{full_text}

---

### **Key Stability Rules**
- Treat **plasma-only studies** as **general PK**, not specimen-specific.  

---

"""

def post_process_design(res: PKPEDesignStepResult) -> PKPEDesignStepResult:
    all_tools = [member.value for member in PipelineTypeEnum]
    for tool in res.pipeline_tools:
        if tool not in all_tools:
            raise RetryException(f"Invalid tool: {tool}")
    return res

class PKPEDesignStep(PKPECommonStep):
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

        agent = self.get_agent(self.llm) # CommonAgent(llm=self.llm)

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