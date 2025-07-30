from typing import Literal
from altair.utils import parse_shorthand
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import logging

from extractor.agents.common_agent.common_agent_2steps import CommonAgentTwoSteps
from extractor.agents.common_agent.common_step import CommonStep,CommonState
from extractor.constants import COT_USER_INSTRUCTION

logger = logging.getLogger(__name__)

class PKPEIdentificationState(CommonState):
    title: str
    abstract: str

class PKPEIdentificationStepResult(BaseModel):
    pkpe_type: Literal["PK", "PE", "Both", "Neither"] = Field(description="The type of the paper")

PKPE_IDENTIFICATION_SYSTEM_PROMPT = """
You are a biomedical research assistant with expertise in pharmacology, specifically in pharmacokinetics (PK) and pharmacoepidemiology (PE). Your task is to determine whether a given published paper is **PK-related**, **PE-related**, **both**, or **neither**, based on its title, abstract, and other available content.

---

### **Definitions for Reference**:

* **Pharmacokinetics (PK)**: The study of how a drug is absorbed, distributed, metabolized, and excreted by the body. PK studies often involve parameters such as clearance, half-life, AUC (area under the curve), Cmax, volume of distribution, and bioavailability. Experimental designs may include drug concentration measurements in plasma/tissues over time.

* **Pharmacoepidemiology (PE)**: The study of the use and effects of drugs in large populations. PE studies often involve observational or real-world data (e.g., insurance claims, electronic health records) and focus on drug safety, utilization, adherence, effectiveness, risk-benefit, and post-marketing surveillance.

---

### **Input**:

You will be given the following fields:

* **Title**: The title of the paper.
* **Abstract**: The abstract text of the paper.

---

### **Your Task**:

Determine whether the paper is:

* `"PK"`: Related to pharmacokinetics
* `"PE"`: Related to pharmacoepidemiology
* `"Both"`: Related to both PK and PE
* `"Neither"`: Not related to PK or PE

---

### **Output Format**:

Respond in the following exact format (no additional text):

```
**FinalAnswer**: [PK / PE / Both / Neither]
```

---

### **Input**:

**Title**: 
{title}

**Abstract**: 
{abstract}

---

"""
class PKPEIdentificationStep(CommonStep):
    def __init__(self):
        super().__init__()
        self.step_name = "PKPEIdentificationStep"

    def _execute_directly(self, state: CommonState):
        state: PKPEIdentificationState = state
        system_prompt = PKPE_IDENTIFICATION_SYSTEM_PROMPT.format(
            title=state["title"],
            abstract=state["abstract"],
        )
        instruction_prompt = COT_USER_INSTRUCTION
        llm = state["llm"]
        agent = CommonAgentTwoSteps(
            llm=llm,
        )
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            schema=PKPEIdentificationStepResult,
        )
        res: PKPEIdentificationStepResult = res
        state["final_answer"] = res.pkpe_type
        self._print_step(state, step_output=reasoning_process)

        return state, token_usage


