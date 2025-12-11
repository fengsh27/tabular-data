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
    reasoning_process: str = Field(description="A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.")
    pipeline_tools: list[str] = Field(description="A list of pipeline tool names")

PKPE_DESIGN_SYSTEM_PROMPT = """
You are a biomedical research assistant specializing in pharmacology, pharmacokinetics (PK), pharmacoepidemiology (PE), and clinical trials (CT).  
Your goal is to select the **most appropriate and stable set of pipeline tools** to curate data from the given paper.

---

### **Reference Definitions**

- **Pharmacokinetics (PK):** Study of how a drug is absorbed, distributed, metabolized, and excreted.  
  Typical PK data include AUC, Cmax, Tmax, CL, Vd, t½, bioavailability, and measured concentrations in biological matrices such as plasma or tissues.

- **Pharmacoepidemiology (PE):** Study of drug use and effects in large populations (e.g., EHR, insurance claims).  
  Focuses on safety, utilization, adherence, effectiveness, risk–benefit, and post-marketing surveillance.

- **Clinical Trials (CT):** Randomized or controlled experiments evaluating treatment efficacy or safety.

---

### **Pipeline Tools**

| Tool Name | Description | Data Type | Scope |
|:--|:--|:--|:--|
| **pk_summary** | Curate PK summary data from tables | Summary | General PK |
| **pk_individual** | Curate PK individual data from tables | Individual | General PK |
| **pk_specimen_summary** | Curate PK specimen summary data (compare across specimen types) from full text | Summary | Specimen-specific |
| **pk_specimen_individual** | Curate PK specimen individual data (specimen-based sampling per subject) from full text | Individual | Specimen-specific |
| **pk_drug_summary** | Curate PK drug summary data (drug-specific parameters) from full text | Summary | Drug-specific |
| **pk_drug_individual** | Curate PK drug individual data from full text | Individual | Drug-specific |
| **pk_population_summary** | Curate PK population summary data (demographics) from tables | Summary | Population/Demographic |
| **pk_population_individual** | Curate PK population individual data from tables | Individual | Population/Demographic |
| **pe_study_info** | Curate PE study information from full text | — | PE study info |
| **pe_study_outcome** | Curate PE outcome data from tables | — | PE outcome tables |

---

### **Stable Selection Rules**

Follow these steps **in order** to ensure deterministic and context-sensitive tool selection.

#### 1. Determine Study Domain
- **PK only:** Choose from `pk_*` tools.  
- **PE only:** Choose from `pe_*` tools.  
- **Both:** Include relevant PK and PE pipelines.  
- **Neither:** Return an empty list.

#### 2. Determine Data Granularity
- **Summary** → mean, SD, median, range, IQR, N=, aggregated group data.  
- **Individual** → rows labeled by subject/case ID.  
If both appear in distinct tables, include both granularity levels.

#### 3. Determine Data Scope (Revised Hierarchy)
- pk_summary, pk_individual, pk_population_summary, pk_population_individual pipelines are always curating data from tables.
- pk_specimen_summary, pk_specimen_individual, pk_drug_summary, pk_drug_individual pipelines are always curating data from full text (including tables).
- pe_study_info pipeline is always curating data from full text (including tables) while pe_study_outcome pipeline is always curating data from tables.
- Tables have higher priority than full text, that is, if a table contains information that can be curated by a pipeline, use the pipeline to curate the table instead of using the full text.

**Clarifications to ensure stability:**

- **True Specimen-specific:**  
  Select *specimen* tools **only if** the paper explicitly compares or curates data from **two or more different specimen types** (e.g., plasma vs. milk, maternal vs. cord blood, serum vs. amniotic fluid).  
  - Examples:  
    - “Drug concentrations in maternal plasma and breast milk” → specimen-specific.  
    - “Drug levels measured in plasma only” → **general PK**, not specimen-specific.

- **Drug-specific:**  
  Choose when tables aggregate PK parameters *by drug or metabolite*, not by subject or specimen (e.g., “mean AUC of Drug A vs. Drug B”).

- **Population/Demographic:**  
  Choose when data summarize or stratify by population variables (e.g., age, genotype, BMI, maternal vs. fetal group averages).

- **General:**  
  Default category for standard PK tables (e.g., plasma concentrations, level-to-dose ratios, AUC tables) **when only one specimen type is present**.

> **Default rule:** “Plasma-only data” is **general PK**, not specimen-specific.

#### 4. Handle Multi-Type Papers
- Include **all relevant** pipeline tools following the above rules.  
- **Do not include redundant tools** of the same granularity and overlapping scope.  
  Example: If `pk_summary` already fits, do not also include `pk_specimen_summary`.

#### 5. Tie-Breaking Rules
- Prefer **the most specific valid match** that does **not conflict** with the default rules.  
- If data could fit both *specimen* and *drug* scopes, use **drug-specific** unless multiple specimen types are clearly compared.  
- Never classify a paper as *specimen-specific* solely because it mentions plasma concentrations or sampling.

---

### **Output Format**
Return the selected tools in the following exact format:
```

Pipeline Tools: [tool_name_1, tool_name_2, ...]

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

### **Example Cases**

#### Example 1 – Single specimen (plasma), individual + summary data  
> Tables show individual plasma levels and summary L/D ratios.  
```

Pipeline Tools: [pk_individual, pk_summary, pk_drug_summary]

```

#### Example 2 – Multiple specimens (plasma + milk)  
```

Pipeline Tools: [pk_specimen_individual, pk_specimen_summary]

```

#### Example 3 – Drug-level comparison only  
```

Pipeline Tools: [pk_drug_summary]

```

#### Example 4 – PK + PE mixed study  
```

Pipeline Tools: [pk_summary, pe_study_outcome]

```

---

### **Key Stability Rules**
- Treat **plasma-only studies** as **general PK**, not specimen-specific.  
- Apply **Rules 1–5 strictly** and never switch hierarchy interpretations between runs.  
```

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

        agent = self.get_agent(state) # CommonAgent(llm=self.llm)

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