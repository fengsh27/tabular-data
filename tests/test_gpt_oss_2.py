import pytest
import logging
import tiktoken
import os
import requests
from langchain_core.messages import BaseMessage

from extractor.request_gpt_oss import get_gpt_oss, get_gpt_qwen_30b
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.agents.pk_pe_agents.pk_pe_correction_step import PKPECuratedTablesCorrectionStep
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState

logger = logging.getLogger(__name__)

msg = """You are a biomedical data verification assistant with expertise in pharmacokinetic population and individual and data accuracy validation. 
Your task is to carefully examine the **source paper title and tables**, and determine whether the **curated pharmacokinetic population and individual data table** is an accurate and faithful representation of the information provided in the source.

---

### **Your Responsibilities**

* Verify that all values in the curated table exactly match or are correctly derived from the source table(s) in the paper.
* Check that the table structure (rows, columns, units, and headers) is curated correctly from the source.
* Identify any discrepancies in numerical values, missing data, wrong units, or incorrect associations (e.g., a value placed in the wrong row or column).
* Consider the context from the paper title if needed (e.g., study type, drug, population) to interpret ambiguous values.

---

### **Input**

You will be given:

* **Paper Title**: The title of the publication.
* **Paper Abstract**: The abstract of the publication.
* **Source Table(s) or full text**: Table(s) extracted directly from the publication, preserving structure and labels, or the full text of the publication.
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

* The columns in the curated table are fixed, so you **should not doubt** the columns in the curated table.
* Focus on **substantial mismatches** in values or structure that could affect the meaning or interpretation. Minor typos, slight wording differences, or small formatting variations are acceptable.
* If the curated table is correct in content but uses slightly different formatting (e.g., reordering of columns), that is acceptable as long as it does not alter the meaning or value.
* In the **Explanation** section, you should try your best to **list all the mismatched values or structure issues**, and provide a brief explanation of why you think the curated table is incorrect.
* Your response will be used to correct the curated table, so you should be **very specific and detailed** in your explanation. **Do not give any general explanation.**
* when values in text and table disagree, treat the table values as the ground truth (even if the text mentions slightly different ones).
---

### **Input**

#### **Paper Title**

Presence of benzophenones commonly used as UV filters and absorbers in paired maternal and fetal samples

#### **Paper Abstract**


Background: Previous studies have demonstrated widespread exposure of humans to certain benzophenones commonly used as UV filters or UV absorbers; some of which have been demonstrated to have endocrine disrupting abilities.

Objectives: To examine whether benzophenones present in pregnant women pass through the placental barrier to amniotic fluid and further to the fetal blood circulation.

Methods: A prospective study of 200 pregnant women with simultaneously collected paired samples of amniotic fluid and maternal serum and urine. In addition, unique samples of human fetal blood (n=4) obtained during cordocentesis: and cord blood (n=23) obtained at delivery, both with paired maternal samples of serum and urine collected simultaneously, were used. All biological samples were analyzed by TurboFlow-liquid chromatography - tandem mass spectrometry for seven different benzophenones.

Results: Benzophenone-1 (BP-1), benzophenone-3 (BP-3), 4-methyl-benzophenone (4-MBP), and 4-hydroxy-benzophenone (4-HBP) were all detectable in amniotic fluid and cord blood samples and except 4-HBP also in fetal blood; albeit at a low frequency. BP-1 and BP-3 were measured at ~10-times lower concentrations in fetal and cord blood compared to maternal serum and 1000-times lower concentration compared to maternal urine levels. Therefore BP-1 and BP-3 were only detectable in the fetal circulation in cases of high maternal exposure indicating some protection by the placental barrier. 4-MBP seems to pass into fetal and cord blood more freely with a median 1:3 ratio between cord blood and maternal serum levels. Only for BP-3, which the women seemed to be most exposed to, did the measured concentrations in maternal urine and serum correlate to concentrations measured in amniotic fluid. Thus, for BP-3, but not for the other tested benzophenones, maternal urinary levels seem to be a valid proxy for fetal exposure.

Conclusions: Detectable levels of several of the investigated benzophenones in human amniotic fluid as well as in fetal and cord blood calls for further investigations of the toxicokinetic and potential endocrine disrupting properties of these compounds in order for better assessment of the risk to the developing fetus.

Keywords: 4-Hydroxy-benzophenone (4-HBP); 4-Methyl-benzophenone (4-MBP); Benzophenone-1 (BP-1); Benzophenone-3 (BP-3); Endocrine disruptors; Fetal exposure.


#### **Source Table(s) or full text**


| Empty Cell | ID | Cordocentesis_0 | Cordocentesis_1 | Cordocentesis_2 | Cordocentesis_3 | Delivery_0 | Delivery_1 | Delivery_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Empty Cell | ID | Urine | Amnion | Serum | Fetal serum | Urine | Serum | Cord blood |
| BP-1 | 1 | – | < LOD | < LOD | < LOD | – | – | – |
| BP-1 | 2 | – | < LOD | 2.28 | 0.36 | – | < LOD | < LOD |
| BP-1 | 3 | 6.47 | < LOD | < LOD | < LOD | – | – | – |
| BP-1 | 4 | 3.68 | – | < LOD | < LOD | 4.13 | < LOD | < LOD |
| BP-3 | 1 | – | < LOD | 0.34 | < LOD | – | – | – |
| BP-3 | 2 | – | 0.33 | 37 | 10.1 | – | 1 | < LOD |
| BP-3 | 3 | 106.3 | < LOD | 0.77 | < LOD | – | – | – |
| BP-3 | 4 | 17.9 | – | 0.55 | < LOD | 32 | 0.77 | < LOD |
| 4-MBP | 1 | – | < LOD | 0.59 | 0.31 | – | – | – |
| 4-MBP | 2 | – | < LOD | 1.62 | < LOD | – | 1.12 | < LOD |
| 4-MBP | 3 | < LOD | < LOD | 1.04 | 1.3 | – | – | – |
| 4-MBP | 4 | 0.61 | – | 4.98 | 1.19 | < LOD | 6.31 | < LOD |


#### **Curated Table**


| Patient ID | Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Parameter type | Parameter unit | Parameter value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | BP-1 | BP-1 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 6.47 | N/A | N/A |
| 4 | BP-1 | BP-1 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 3.68 | N/A | N/A |
| 3 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 106.3 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 17.9 | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.61 | N/A | N/A |
| 1 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LODa | N/A | N/A |
| 2 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.33 | N/A | N/A |
| 3 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 2.28 | N/A | N/A |
| 3 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.34 | N/A | N/A |
| 2 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 37 | N/A | N/A |
| 3 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.77 | N/A | N/A |
| 4 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.55 | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.59 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.62 | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.04 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 4.98 | N/A | N/A |
| 1 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.36 | N/A | N/A |
| 3 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 10.1 | N/A | N/A |
| 3 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.31 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.3 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.19 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 4.13 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 32 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1 | N/A | N/A |
| 4 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.77 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.12 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 6.31 | N/A | N/A |
| 1 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |


---

### **Output Format Instructions**
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"reasoning_process": {"description": "A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.", "title": "Reasoning Process", "type": "string"}, "correct": {"description": "Whether the curated table is accurate and faithful to the source table(s).", "title": "Correct", "type": "boolean"}, "explanation": {"description": "Brief explanation of whether the curated table is accurate. If incorrect, explain what is wrong, including specific mismatched values or structure issues.", "title": "Explanation", "type": "string"}, "suggested_fix": {"description": "If incorrect, provide a corrected version of the curated table or the corrected values/rows/columns.", "title": "Suggested Fix", "type": "string"}}, "required": ["reasoning_process", "correct", "explanation", "suggested_fix"]}
```

"""


def approx_token_count(text: str | list[BaseMessage]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    if isinstance(text, str):
        return len(enc.encode(text))
    return sum(len(enc.encode(m.content or "")) for m in text)

@pytest.mark.skip()
def test_gpt_oss_with_29100749():
    long_msg = msg+msg+msg
    logger.info(f"Token count: {approx_token_count(long_msg)}")
    llm = get_gpt_oss()
    res = llm.invoke(long_msg)
    logger.info(res)
    logger.info(res.response_metadata)
    logger.info(res.usage_metadata)

@pytest.mark.skip()
def test_request_gpt_oss():
    long_msg = msg+msg+msg
    
    payload = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": long_msg}],
        "stream": False,
        "think": "low",
        "options": {"num_ctx": 16384}
    }
    base_url = os.getenv("OLLAMA_BASE_URL")
    r = requests.post(f"{base_url}/api/chat", json=payload)
    res = r.json()
    logger.info(res.get("prompt_eval_count"))