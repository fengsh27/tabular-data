from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI

from extractor.agents.common_agent.common_step import CommonStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.constants import COT_USER_INSTRUCTION

PKPE_VERIFICATION_SYSTEM_PROMPT = """
You are a biomedical data verification assistant with expertise in {domain} and data accuracy validation. 
Your task is to carefully examine the **source paper title and tables**, and determine whether the **curated {domain} data table** is an accurate and faithful representation of the information provided in the source.

---

### **Your Responsibilities**

* Verify that all values in the curated table exactly match or are correctly derived from the source table(s) in the paper.
* Check that the table structure (rows, columns, units, and headers) is consistent with the source.
* Identify any discrepancies in numerical values, missing data, wrong units, or incorrect associations (e.g., a value placed in the wrong row or column).
* Consider the context from the paper title if needed (e.g., study type, drug, population) to interpret ambiguous values.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Source Table(s)**: Table(s) extracted directly from the publication, preserving structure and labels.
* **Curated Table**: The data table that has been curated from the above source for downstream use.

---

### **Your Output**

You must respond using the **exact format** below:

```
**FinalAnswer**: [Correct / Incorrect]
**Explanation**: [Brief explanation of whether the curated table is accurate. If incorrect, explain what is wrong, including specific mismatched values or structure issues.]
**SuggestedFix**: [If incorrect, provide a corrected version of the curated table or the corrected values/rows/columns.]
```

---

### **Important Notes**

* Be exact and detail-oriented. Even a small mismatch in value, label, or unit is considered incorrect.
* If the curated table is correct in content but uses slightly different formatting (e.g., reordering of columns), that is acceptable as long as it does not alter the meaning or value.
* If you're unsure, err on the side of flagging the item as **Incorrect** with justification.

---

### **Input**

#### **Paper Title**

{paper_title}

#### **Source Table(s)**

{source_tables}

#### **Curated Table**

{curated_table}


"""

class PKPEVerificationStep(CommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        output_callback: Callable[[dict], None],
        pmid: str,
        domain: str,
    ):
        super().__init__(llm, output_callback)
        self.pmid = pmid
        self.domain = domain

    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        state: PKPECurationWorkflowState = state
        system_prompt = PKPE_VERIFICATION_SYSTEM_PROMPT.format(
            paper_title=state["paper_title"],
            source_tables=state["source_tables"],
            curated_table=state["curated_table"],
            domain=self.domain,
        )
        instruction_prompt = COT_USER_INSTRUCTION

