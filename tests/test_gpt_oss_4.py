import pytest
import requests
import json
import dotenv
import os
import logging

msg = """Human: 
The following main table contains pharmacokinetics (PK) data:  
col: | "Pharmacokinetic parameter." | "Naso- or orogastric tube administration,n= 14/19." | "Oral administration,n= 5/19." | "Pvalue." |
| --- | --- | --- | --- |
row 0: | AUC0–12(h*μg/mL) | 220 (157.5–355.4) | 213.8 (154.0–348.8) | 0.90 |
row 1: | Cmax(μg/mL) | 23.8 (18.8–41.3) | 26.4 (19.3–34.2) | 0.84 |
row 2: | Tmax(h) | 1.5 (1.5–2.5) | 2.5 (1.5–2.9) | 0.46 |
Here is the table caption:  

Open in new tab Table 3\u2002Comparison of pharmacokinetic parameters between participants receiving levetiracetam via naso- or orogastric tube vs. participants receiving the drug orally
All values expressed as medians and interquartile ranges. Maximum plasma concentration (Cmax), time to maximum plasma concentration (Tmax) and area under the curve 0–12 h (AUC0–12).aThe study was not powered to specifically compare naso- or orogastric administration vs. oral administration.

From the main table above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population" - "Pregnancy stage" - "Subject N," as follows:
col: | "Population" | "Pregnancy stage" | "Subject N" |
| --- | --- | --- |
row 0: | N/A | N/A | "14" |
row 1: | N/A | N/A | "5" |
row 2: | N/A | N/A | "19" |

Carefully analyze the tables and follow these steps to refine Subtable 1 into a more detailed Subtable 2:  

(1) Identify all unique combinations of **[Population, Pregnancy stage, Pediatric/Gestational age, Subject N]** from the table.
    - **Population**: The age group of the subjects.  
    **Common categories include:**  
    - "Maternal" (pregnant individuals)
    - "Preterm" or "Premature" (typically ≤ 37 weeks of gestation)  
    - "Neonates" or "Newborns" (generally birth to ~1 month)  
    - "Infants" (~1 month to ~1 year)  
    - "Children" (~1 year to ~12 years)  
    - "Adolescents" or "Teenagers" (~13 years to ~17 years)  
    - "Adults" (typically 18 years or older)  
    
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
    **Common categories include:**  
    - "Trimester 1" (usually up to 14 weeks of pregnancy)  
    - "Trimester 2" (~15–28 weeks of pregnancy)  
    - "Trimester 3" (~≥ 28 weeks of pregnancy)  
    - "Fetus" or "Fetal Stage" (referring to the developing baby during pregnancy)  
    - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
    - "Postpartum" (~6–8 weeks after birth)  
    - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
    
    - **Pediatric/Gestational age**: The child\'s age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
    Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
    For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
    
    - **Subject N**: The number of subjects corresponding to the specific population.

(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["N/A", "N/A", "N/A", "15"], ...]

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
   (6) Strictly ensure that you process only rows 0 to 2 from the Subtable 1 (which has 3 rows in total).   
   - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
   - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Subject N for each row in Subtable 2 must be the same as in Subtable 1.
   
   
   ---
   
   ### **Previous Errors to Avoid**
   
   You must pay close attention to the following corrections from previous runs. Do not repeat these specific errors.
   
   
   N/A


---

"""

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

@pytest.mark.skip()
def test_gpt_oss_3():
    url = os.getenv("OLLAMA_BASE_URL")
    payload = {
        # "model": "gpt-oss:20b",     # or any model installed in your local ollama
        "model": "qwen3:30b",
        "prompt": msg,
        "think": False,
        "stream": False,
        "format": {
            "type": "object",
            "description": "Refined Patient Info Result", 
            "properties": {
                "reasoning_process": {
                    "description": "A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.", 
                    "title": "Reasoning Process", 
                    "type": "string"
                }, 
                "refined_patient_combinations": {
                    "description": "a list of lists of unique combinations [Population, Pregnancy stage, Subject N]", 
                    "items": {
                        "items": {
                            "type": "string"
                        }, 
                        "type": "array"
                    }, 
                    "title": "Refined Patient Combinations", 
                    "type": "array"
                }
            }, 
            "required": ["reasoning_process", "refined_patient_combinations"],
            "additionalProperties": False
        },
        "options": {
            "num_ctx": 16384,
            "num_predict": 4096,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        }
    }
    response = requests.post(f"{url}/api/generate", json=payload)
    response.raise_for_status()

    res = response.json()
    logger.info("done=%s done_reason=%s response_len=%d thinking_len=%d error=%s",
            res.get("done"),
            res.get("done_reason"),
            len(res.get("response") or ""),
            len(res.get("thinking") or ""),
            res.get("error"))
    logger.info(res.get("response"))