import pytest
import os
import logging
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

msg = [(
    "system",
    """You are a PK expert. You must output valid JSON only. Do not use <think> tags.
"""
),(
    "user",
    """
You are given a pharmacokinetics (PK) data table and its caption.

Your goal is to identify unique combinations of:
[Patient ID, Population, Pregnancy stage]

--------------------------------------------------
TABLE (verbatim, row by row):
col: | "Unnamed: 0" | "ID" | "Dose (mg/d)" | "Mother\'s PL III trimester (ng/ml)" | "Mother\'s PL (ng/ml) delivery" | "Infant\'s PL (ng/ml)" | "Umbilical maternal ratio (%)" | "Other drugs" | "Expected phenotype (CYP2D6: major CYP metabolizer)" | "Maternal outcomes" | "Bleeding (ml)" | "Neonatal outcomes" |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
row 0: | Sertraline | 1 | 75 | 19.5 | NA | NA | NA | - | UM | - | 250 | Apnea with desaturation and mild bradycardia, superficial breathing, polypnea at 48 h of life up to 5 day |
row 1: | Sertraline | 3 | 75 | 14.4 | 6.1 | <5 | 40.1 | - | IM | PROM | 400 | - |
row 2: | Sertraline | 4 | 150 | 25.7 | NA | 9 | NA | - | NA | - | 300 | Transient respiratory difficulty, mild irritability up to 48 h of life |
row 3: | Sertraline | 5 | 100 | 42.7 | 16.7 | 9.7 | 58.1 | - | PM | - | 100 | Tachypnea up to 48 h of life, SGA |
row 4: | Sertraline | 12 | 150 | 16.7 | 14.1 | <5 | 17.7 | - | EM | - | 400 | - |
row 5: | Sertraline | 13 | 75 | 26 | 12 | <5 | 20.8 | Phenobarbital | RM | - | 300 | SGA |
row 6: | Sertraline | 14 | 50 | 14.1 | 11.6 | <5 | 21.5 | - | IM | - | 500 | Left clubfoot |
row 7: | Sertraline | 15 | 100 | 27.4 | 42.6 | 19 | 44.6 | - | EM | - | 100 | Low apgar score 1′, RDS, hypotonus, SGA |
row 8: | Sertraline | 18 | 100 | 12.1 | NA | NA | NA | Elvitegravir, Cobicistat, Emtricitabine, Tenofovir, Alafenamide; Zidovudine chemoprophylaxis during labour | UM | - | 500 | Low apgar score 1′, RDS (CPAP, transfer in NICU), SGA |
row 9: | Sertraline | 22 | 50 | NA | 24.5 | 11.1 | 45.3 | - | PM | - | 100 | RDS, neurologic symptom |
row 10: | Sertraline | 23 | 100 | NA | 9.2 | 6.5 | 70.6 | - | EM | - | 150 | - |
row 11: | Sertraline | 25 | 125 | NA | 89 | 35.5 | 39.9 | - | EM | - | 200 | - |
row 12: | Sertraline | 26 | 100 | NA | 32.7 | 7.8 | 23.8 | - | IM | PPH | 500 | - |
row 13: | Sertraline | 32 | 50 | NA | 12.7 | 13 | 102.4 | - | EM | - | 300 | - |
row 14: | Sertraline | 33 | 25 | NA | 9.5 | <5 | 26.3 | - | EM | - | 300 | - |
row 15: | Sertraline | 34 | 50 | NA | 7.4 | <5 | 33.8 | - | EM | - | 100 | - |
row 16: | Sertraline | 35 | 50 | NA | 7.8 | <5 | 32.0 | - | NA | - | 200 | - |
row 17: |  | 37 | 50 | NA | 16.9 | 8.2 | 48.5 | Levothyroxine | IM | - | 100 | - |
row 18: |  | 38 | 50 | NA | 9.5 | <5 | 26.6 | - | UM | - | 300 | - |
row 19: |  | 43 | 50 | 13.2 | 7.7 | 12.3 | 159.7 | - | NA | - | 100 | - |
row 20: |  | 46 | 50 | NA | 20 | 6 | 30.0 | Nifedipine | NA | GH | 800 | - |
row 21: |  | 48 | 50 | NA | 8.7 | 5 | 57.5 | - | NA | PPH | 1600 | - |
row 22: |  | 52 | 25 | <5 | <5 | <5 | NA | - | EM | PPH | 1500 | - |
row 23: |  | 53 | 50 | 13.1 | <5 | NA | NA | - | EM | PROM | 250 | Low apgar score 1′ and 5’; cyanosis, no spontaneous breathing (ventilation up to 24 h of life) |
row 24: | Citalopram | 36 | 20 | 90.2 | NA | NA | NA | - | EM | PROM, PPH | 1500 | - |
row 25: | Citalopram | 42 | 20 | 14.9 | 9.3 | 5.5 | 59.1 | Ursodeoxycholic acid | IM | Intrahepatic cholestasis | 250 | Neurologic symptom |
row 26: | Escitalopram | 9 | 5 | <5 | NA | NA | NA | - | IM | - | 250 | - |
row 27: | Escitalopram | 10 | 10 | 21.8 | 32.4 | 15.9 | 49.1 | - | NA | - | 600 | - |
row 28: | Escitalopram | 28 | 10 | NA | 10.1 | NA | NA | - | EM | - | 600 | - |
row 29: | Escitalopram | 29 | 10 | NA | <5 | NA | NA | - | IM | - | 900 | Neurologic symptom |
row 30: | Escitalopram | 30 | 10 | 68.8 | 36.5 | NA | NA | Ranitidine | IM | - | 100 | Low apgar score 1′, RDS, hypotonus SGA |
row 31: | Escitalopram | 47 | 5 | NA | 8.9 | 7.7 | 86.5 | - | NA | PROM | 500 | Bilateral hydronephrosis |
row 32: | Escitalopram | 56 | 10 | <5 | <5 | <5 | NA | - | EM | PROM | 150 | - |
row 33: | Escitalopram |  |  |  |  |  |  |  | Expected phenotype (CYP2D6: major CYP metabolizer) |  |  |  |
row 34: | Venlafaxine | 2 | 75 | 245 | 76.8 | 184.8 | 240.6 | - | NA | - | 400 | SGA |
row 35: | Venlafaxine | 6 | 375 | 37.6 | NA | NA | NA | Levothyroxine, Insulin | EM | GDM, PE, GH | 500 | Prematurity, left thumb bifid, low apgar score 1′ |
row 36: | Venlafaxine | 7 | 150 | NA | 179.9 | 234.7 | 13.5 | - | EM | PROM | 100 | - |
row 37: | Venlafaxine | 16 | 75 | 86.1 | NA | NA | NA | Levothyroxine | EM | PPH | 700 | Prematurity, macroglossia, transient hypoglycemia |
row 38: | Venlafaxine | 17 | 75 | 71.5 | NA | NA | NA | Insulin | EM | GDM | NA | - |
row 39: |  | 20 | 150 | NA | 199.8 | 78.9 | 39.5 | - | EM | - | 100 | Prematurity, low apgar score 1′ |
row 40: |  | 24 | 75 | 154.6 | NA | NA | NA | - | I/EM | PPH | 1100 | - |
row 41: | Paroxetine | 8 | 20 | 10.3 | NA | NA | NA | - | EM | PROM | 200 | SGA |
row 42: | Paroxetine | 19 | 20 | NA | 10 | NA | NA | Levothyroxine | EM | PROM | 600 | Risk of fetal distress (deep bradycardia), postpartum anemia, low apgar score 1′ |
row 43: | Paroxetine | 21 | 25 | NA | 23.4 | 9.1 | 38.9 | - | IM | - | 100 | Acrocyanosis at 8 h of life |
row 44: | Paroxetine | 31 | 20 | NA | 48.6 | 10.3 | 21.2 | - | I/EM | - | 300 | SGA, prematurity, low apgar score 1′, RDS, neurologic symptoms |
row 45: | Paroxetine | 45 | 20 | NA | <5 | <5 | NA | Levothroxine | NA | PPH | 1100 | BIlateral hydronephrosis |
row 46: | Paroxetine | 49 | 20 | <5 | <5 | NA | NA | - | EM | - | 300 | - |
row 47: | Paroxetine | 50 | 10 | 41.9 | 12.9 | 2.5 | 19.4 | - | PM | - | 100 | - |
row 48: | Paroxetine | 51 | 20 | <5 | <5 | <5 | NA | Levothiroxine | EM | PPH | 900 | - |
row 49: | Paroxetine | 54 | 10 | NA | <5 | <5 | NA | - | EM | - | 150 | - |
row 50: | Paroxetine | 55 | 20 | NA | <5 | <5 | NA | - | NA | - | 350 | SGA, Bicuspid aorta, hyper-excitability, bradycardia, desaturation, hypertonus, afinalistic sucking. Spontaneous resolution at 24 h of life |
row 51: | Paroxetine | 57 | 20 | NA | <5 | <5 | NA | - | NA | GDM, PROM, PPH | 1000 | - |
row 52: | Fluoxetine | 39 | 30 | 346.8 | NA | NA | NA | - | IM | PROM | NA | - |
row 53: | Fluoxetine | 40 | NA | NA | 147 | 152 | 103.4 | - | NA | Anhydramnios | 200 | - |
row 54: | Fluoxetine | 41 | NA | NA | <5 | <5 | NA | Insuline | NA | PROM | NA | Prematurity |
row 55: | Fluoxetine | 44 | 20 | NA | 216.8 | 108.6 | 50.1 | - | UM | - | 100 | SGA, Neonatal infection |
--------------------------------------------------

TABLE CAPTION:
Individual pharmacokinetic, pharmacogenetic data and maternal/neonatal outcomes for women
Abbreviations: APGAR, Appearance, Pulse, Grimace, Activity, Respiration; CPAP, Continuous Positive Airway Pressure; EM, Extensive Metabolizer; GDM, Gestational Diabetes Mellitus; GH, gestational Hypertension; IM, Intermediate Metabolizer; NA, not available; NICU, Neonatal Intensive Care Unit; PL, Plasma Level; PROM, Premature Rupture of Membranes; PM, Poor Metabolizer; PPH, postpartum hemorrhage; RDS, Respiratory Distress Syndrome; RP, Rapid Metabolizer; SGA, Small for Gestational Age; UM, Ultrarapid Metabolizer.
[Correction added on December 5, 2020 after first online publication: The 8th column heading has been corrected. “CYP2C19” has been revised to “CYP2D6”.]
--------------------------------------------------

DEFINITIONS (IMPORTANT):

1. Patient ID
- Patient ID refers to the identifier assigned to a unique individual patient.
- Use the exact value as written in the table.
- If no explicit Patient ID exists, infer a unique unit for each row and use that as the Patient ID.
- If inference is required, prefer row-level identifiers such as row number or drug-specific unit labels.
- Do NOT invent new identifiers.

2. Population
- Population refers to the patient age group.
- If not explicitly stated, use "N/A".

3. Pregnancy stage
- Pregnancy stage refers to the stage of pregnancy mentioned in the study.
- If not explicitly stated, use "N/A".

--------------------------------------------------
TASK INSTRUCTIONS:

1. Examine the table row by row.
2. Identify all unique [Patient ID, Population, Pregnancy stage] combinations.
3. Deduplicate combinations while preserving correctness.
4. Every element in every combination MUST be a string.
5. If information is missing:
   - First try to infer from context or nearby rows.
   - Use "N/A" only if inference is not possible.
6. If and ONLY if no valid Patient ID or unique unit can be identified for the entire table,
   return an empty list.

--------------------------------------------------
OUTPUT REQUIREMENTS (STRICT):

- You MUST return a valid JSON object.
- Do NOT include markdown, comments, explanations, or extra text.
- Do NOT include angle brackets << >>.
- Do NOT include Python code.
- The output MUST conform EXACTLY to the schema below.

SCHEMA:
{
  "patient_combinations": [
    ["Patient ID", "Population", "Pregnancy stage"]
  ]
}

- If no combinations are found, return:
{
  "patient_combinations": []
}

You MUST output a JSON object. Do not output anything else.

"""
)]

def test_chat_ollama():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        base_url=base_url,
        model="gpt-oss:20b",
        # model="qwen3:30b",
        reasoning=False,
        streaming=False,
        num_ctx=16384,
        num_predict=512,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
    )

    raw = llm.invoke(msg)
    logger.info("content repr:", repr(raw.content))
    logger.info("additional_kwargs:", raw.additional_kwargs)
    assert raw.content.strip() != "" or raw.additional_kwargs.get("reasoning")