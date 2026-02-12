import pytest
import os
import logging
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

msg = """

You are a biomedical data correction engineer with expertise in pharmacokinetics and robust Python data wrangling.

You are given:
1) Paper Title
2) Paper Abstract
3) Source Table(s) or full text
4) Curated Table (a markdown table string; this is the input to your code)
5) Reasoning Process (explains what is wrong and what to fix)
6) A PROVIDED Python function: markdown_to_dataframe(md_table: str) -> pandas.DataFrame

CRITICAL: markdown_to_dataframe() is already implemented and available at runtime.
- You MUST NOT define it, re-implement it, or include any parsing logic that duplicates it.
- Assume it exists in the global scope and call it directly as: df = markdown_to_dataframe(curated_md).
- If you think the **Reasoning Process** is incorrect, please return code "pass".
- Do not use <think> tags or hidden reasoning. All output must be visible plain text.

FORBIDDEN (must not appear anywhere in your output code):
- Any function definition or assignment for markdown_to_dataframe, including:
  - "def markdown_to_dataframe"
  - "markdown_to_dataframe ="
  - any custom markdown parsing implementation intended to replace it

If you violate the FORBIDDEN rule, your output is considered invalid.

Your task:
Write Python code ONLY that corrects the curated table according to the Reasoning Process and the source table(s).

Hard requirements:

- The code inside "code" must be runnable as-is (no placeholders).
- Do NOT make network calls.
- Do NOT read/write files.
- Use only standard libraries plus pandas (assume pandas is installed).

Input/Output contract:
- curated_md (str) will be provided at runtime.
- You must produce df_corrected (pandas.DataFrame).
- df_corrected must preserve:
  - the same columns (names and order) as the curated table header
  - the same number of rows as input, unless the Reasoning Process explicitly requires insertion/deletion
- All cell values must remain strings unless the Reasoning Process explicitly requires type conversion.

Patch-only correction rule (highest priority):
- Treat the curated table as the base.
- Apply ONLY the explicit corrections described in the Reasoning Process (no additional cleanup or normalization).
- Preserve all other rows/columns unchanged.
- If a correction targets a duplicate row, disambiguate by matching as many columns as needed; if still ambiguous, use row position (0-based index in the body) and add a brief code comment explaining the choice.

Required structure of the code (enforced order):
1) import pandas as pd
2) df = markdown_to_dataframe(curated_md)
3) Apply the minimal set of edits specified by the Reasoning Process
4) df_corrected = df (or a modified copy), ensuring column order unchanged
5) Lightweight validation assertions:
   - row count unchanged unless explicitly required
   - edits applied only to intended rows (use boolean masks; assert mask.sum() matches expectation)

DO NOT:
- Define markdown_to_dataframe (see FORBIDDEN)
- Re-parse markdown manually
- Change column names
- Reorder rows unless explicitly required
- Convert types unless explicitly required

--------------------
Paper Title:

Placental transfer of anti-tumor necrosis factor agents in pregnant patients with inflammatory bowel disease


Paper Abstract:

Background & aims: Some women with inflammatory bowel disease require therapy with tumor necrosis factor (TNF) antagonists during pregnancy. It is not clear whether these drugs are transferred to the fetus via the placenta and then cleared, or whether structurally different TNF antagonists have different rates of transfer.
Methods: We studied 31 pregnant women with inflammatory bowel disease receiving infliximab (IFX, n = 11), adalimumab (ADA, n = 10), or certolizumab (CZP, n = 10). Serum concentrations of the drugs were measured at birth in the mother, infant, and in cord blood, and then monthly in the infant until the drugs were undetectable. Drug concentrations in the cord and the infant at birth were compared with those of the mother.
Results: Concentrations of IFX and ADA, but not CZP, were higher in infants at birth and their cords than in their mothers. The levels of CZP in infants and their cords were less than 2 μg/mL. The median level of IFX in the cord was 160% that of the mother, the median level of ADA in the cord was 153% that of the mother, and the median level of CZP in the cord was 3.9% that of the mother. IFX and ADA could be detected in the infants for as long as 6 months. No congenital anomalies or serious complications were reported.
Conclusions: The TNF antagonists IFX and ADA are transferred across the placenta and can be detected in infants at birth; the drugs were detected in infants up to 6 months after birth. CZP has the lowest level of placental transfer, based on levels measured in cords and infants at birth, of the drugs tested.


Source Table(s) or full text:

| ('Pt #', 'Pt #') | ('IFX dose (mg/kg)', 'IFX dose (mg/kg)') | ('IFX interval (wks)', 'IFX interval (wks)') | ('Time dose to birth (days)', 'Time dose to birth (days)') | ('IFX (μg/ml) at Birth', 'Mom:') | ('IFX (μg/ml) at Birth', 'Cord:') | ('IFX (μg/ml) at Birth', 'Infant') | ('Ratio cord/Mother (%)', 'Ratio cord/Mother (%)') | ('Month IFX undetectable', 'Month IFX undetectable') | ('Newborn Complications', 'Newborn Complications') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1^ | 10 | 6 | 14 | 40 (6 wks) | -- | 39.5* (6 wks) | -- | 7 | None |
| 2 | 5 | 6 | 30 | 15.1 | -- | 25.3 | -- | 5 | Meconium |
| 3#^ | 5 | 6 | 2 | 1.4 | 2.0 | 2.9* (2 wks) | 143% | 2 | Hand-foot-mouth (9mos); respiratory distress (11 mos) |
| 4#^ | 5 | 6 | 14 | 19.2 | 26.5 | 23.6 | 138% | 7 | Oral candida (10 wks); GERD (4 mos) |
| 5 | 5 | 8 | 91 | 3.8 | 3.3 | 4.2 | 87% | 2 | Jaundice |
| 6 | 5 | 8 | 15 | 4.8 | 8.8 | 8.7 | 183% | 3 | None |
| 7 | 5 | 8 | 55 | 14.5 | 20.5 | 28.2 | 141% | 4 | URI 2 weeks |
| 8 | 5 | 6 | 46 | 16.5 | 26.5 | 27.5 | 160% | 5 | None |
| 9 | 5 | 8 | 35 | 2.2 | 8.4 | 10.6 | 381% | 4 | None |
| 10 | 5 | 6 | 77 | 4.1 | 13.6 | 4.7* (4 wks) | 332% | -- | None |
| 11 | 10 | 8 | 74 | 5.1 | 20.4 | 8.4* (4 wks) | 400% | 4 | None |


| ('Pt #', 'Pt #') | ('ADA dose', 'ADA dose') | ('ADA interval', 'ADA interval') | ('Time dose to birth (days)', 'Time dose to birth (days)') | ('ADA(μg/ml) at Birth', 'Mom:') | ('ADA(μg/ml) at Birth', 'Cord:') | ('ADA(μg/ml) at Birth', 'Infant') | ('Ratio Cord/Mother', 'Ratio Cord/Mother') | ('Follow ADA Levels (time)', 'Follow ADA Levels (time)') | ('Newborn Complications', 'Newborn Complications') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1#^ | 40 mg | EOW | 7 | 6.05 | 9.29 | 6.17 | 153% | -- | None |
| 2^ | 40 mg | EOW | 56 | 1.8399999999999999 | 5.39 | 6.01 | 293% | 1.94 (6 wks) | Pulmonary edema, brief at birth |
| 3#^ | 40 mg | EOW | 7 | 3.84 | 4.57 | -- | 119% | -- | None |
| 4# | 40 mg | EOW | 42 | 0.0 | 0.16 | -- | -- | -- | None |
| 5 | 40 mg | EOW | 35 | 2.2 | 4.18 | 4.28 | 190% | .934 (8 wks) | None |
| 6 | 40 mg | EOW | 42 | 3.21 | 4.74 | 4.87 | 148% | 1.31 (7 wks) | None |
| 7^ | 40 mg | EOW | 42 | 3.36 | 8.94 | 8.09 | 266% | 0.529 (11 wks) | None |
| 8 | 40 mg | Weekly | 1 | 16.1 | 19.7 | 17.7 | 122% | -- | None |
| 9 | 40 mg | EOW | 49 | 2.24 | 4.95 | 4.64 | 220% | -- | None |
| 10 | 40 mg | EOW | 7 | 8.48 | 8.29 | 9.17 | 98% | -- | None |


| ('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'Mother #') | ('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Last Dose (days)') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Mother') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Cord') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Infant') | ('Ratio Cord/Mother', 'Unnamed: 5_level_1', 'Unnamed: 5_level_2') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Mother') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Cord') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Infant') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 14 | 18.83 | 1.65 | - | 8.8% | 33.4 | * | * |
| 02 | 7 | 59.57 | 0.94 | 1.02 | 1.6% | 51.3 | * | * |
| 03 | 28 | 4.87 | 1.19 | 1.22 | 24% | * | * | * |
| 04 | 17 | 20.13 | 0.57 | 0.44 | 2.8% | 34.7 | * | * |
| 05 | 21 | 16.49 | <0.41 | <0.41 | 2.5% | 27.7 | * | No sample |
| 06 | 24 | 34.65 | 1.66 | 1.58 | 4.8% | 34.4 | * | * |
| 07 | 28 | 1.87 | <0.41 | <0.41 | 22% | * | * | * |
| 08-A | 42 | 6.32 | <0.41 | 0.58 | 6.4% | 11.1 | * | * |
| B |  |  | <0.41 | <0.41 | 6.4% |  | * | * |
| 09-A | 6 | 42.7 | 1.28 | 1.34 | 3.0% | 62.1 | * | * |
| B |  |  | 1.16 | 1.18 | 2.7% |  | * | * |
| 10 | 5 | 37.83 | 0.55 | 0.6 | 1.5% | 74.9 | * | * |


Curated Table:
(curated markdown table string will be assigned to curated_md at runtime)

| Patient ID | Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Parameter type | Parameter unit | Parameter value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1^ | Infliximab | Infliximab | Mom | Maternal | Trimester 3 | 7 | IFX (μg/ml) at Birth-Mom: | μg/ml | 40 | 14 | days |
| 2 | Infliximab | Infliximab | Mom | Maternal | N/A | 5 | IFX (μg/ml) at Birth-Mom: | μg/ml | 15.1 | 30 | days |
| 3#^ | Infliximab | Infliximab | Mom | Maternal | Postpartum | 2 | IFX (μg/ml) at Birth-Mom: | μg/ml | 1.4 | 2 | days |
| 4#^ | Infliximab | Infliximab | Mom | Maternal | Postpartum | 7 | IFX (μg/ml) at Birth-Mom: | μg/ml | 19.2 | 14 | days |
| 5 | Infliximab | Infliximab | Mom | Maternal | N/A | 2 | IFX (μg/ml) at Birth-Mom: | μg/ml | 3.8 | 91 | days |
| 6 | Infliximab | Infliximab | Mom | Maternal | N/A | 3 | IFX (μg/ml) at Birth-Mom: | μg/ml | 4.8 | 15 | days |
| 7 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Mom: | μg/ml | 14.5 | 55 | days |
| 8 | Infliximab | Infliximab | Mom | Maternal | N/A | 5 | IFX (μg/ml) at Birth-Mom: | μg/ml | 16.5 | 46 | days |
| 9 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Mom: | μg/ml | 2.2 | 35 | days |
| 10 | Infliximab | Infliximab | Mom | Maternal | N/A | N/A | IFX (μg/ml) at Birth-Mom: | μg/ml | 4.1 | 77 | days |
| 11 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Mom: | μg/ml | 5.1 | 74 | days |
| 3#^ | Infliximab | Infliximab | Cord | Maternal | Postpartum | 2 | IFX (μg/ml) at Birth-Cord: | μg/ml | 2.0 | 2 | days |
| 4#^ | Infliximab | Infliximab | Cord | Maternal | Postpartum | 7 | IFX (μg/ml) at Birth-Cord: | μg/ml | 26.5 | 14 | days |
| 5 | Infliximab | Infliximab | Cord | Maternal | N/A | 2 | IFX (μg/ml) at Birth-Cord: | μg/ml | 3.3 | 91 | days |
| 6 | Infliximab | Infliximab | Cord | Maternal | N/A | 3 | IFX (μg/ml) at Birth-Cord: | μg/ml | 8.8 | 15 | days |
| 7 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Cord: | μg/ml | 20.5 | 55 | days |
| 8 | Infliximab | Infliximab | Cord | Maternal | N/A | 5 | IFX (μg/ml) at Birth-Cord: | μg/ml | 26.5 | 46 | days |
| 9 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Cord: | μg/ml | 8.4 | 35 | days |
| 10 | Infliximab | Infliximab | Cord | Maternal | N/A | N/A | IFX (μg/ml) at Birth-Cord: | μg/ml | 13.6 | 77 | days |
| 11 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Cord: | μg/ml | 20.4 | 74 | days |
| 1^ | Infliximab | Infliximab | Infant | Maternal | Trimester 3 | 7 | IFX (μg/ml) at Birth-Infant | μg/ml | 39.5* (6 wks) | 14 | days |
| 2 | Infliximab | Infliximab | Infant | Maternal | N/A | 5 | IFX (μg/ml) at Birth-Infant | μg/ml | 25.3 | 30 | days |
| 3#^ | Infliximab | Infliximab | Infant | Maternal | Postpartum | 2 | IFX (μg/ml) at Birth-Infant | μg/ml | 2.9* (2 wks) | 2 | days |
| 4#^ | Infliximab | Infliximab | Infant | Maternal | Postpartum | 7 | IFX (μg/ml) at Birth-Infant | μg/ml | 23.6 | 14 | days |
| 5 | Infliximab | Infliximab | Infant | Maternal | N/A | 2 | IFX (μg/ml) at Birth-Infant | μg/ml | 4.2 | 91 | days |
| 6 | Infliximab | Infliximab | Infant | Maternal | N/A | 3 | IFX (μg/ml) at Birth-Infant | μg/ml | 8.7 | 15 | days |
| 7 | Infliximab | Infliximab | Infant | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Infant | μg/ml | 28.2 | 55 | days |
| 8 | Infliximab | Infliximab | Infant | Maternal | N/A | 5 | IFX (μg/ml) at Birth-Infant | μg/ml | 27.5 | 46 | days |
| 9 | Infliximab | Infliximab | Infant | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Infant | μg/ml | 10.6 | 35 | days |
| 10 | Infliximab | Infliximab | Infant | Maternal | N/A | N/A | IFX (μg/ml) at Birth-Infant | μg/ml | 4.7* (4 wks) | 77 | days |
| 11 | Infliximab | Infliximab | Infant | Maternal | N/A | 4 | IFX (μg/ml) at Birth-Infant | μg/ml | 8.4* (4 wks) | 74 | days |
| 3#^ | Infliximab | Infliximab | Cord | Maternal | Postpartum | 2 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 143% | 2 | days |
| 4#^ | Infliximab | Infliximab | Cord | Maternal | Postpartum | 7 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 138% | 14 | days |
| 5 | Infliximab | Infliximab | Cord | Maternal | N/A | 2 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 87% | 91 | days |
| 6 | Infliximab | Infliximab | Cord | Maternal | N/A | 3 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 183% | 15 | days |
| 7 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 141% | 55 | days |
| 8 | Infliximab | Infliximab | Cord | Maternal | N/A | 5 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 160% | 46 | days |
| 9 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 381% | 35 | days |
| 10 | Infliximab | Infliximab | Cord | Maternal | N/A | N/A | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 332% | 77 | days |
| 11 | Infliximab | Infliximab | Cord | Maternal | N/A | 4 | Ratio cord/Mother (%)-Ratio cord/Mother (%) | % | 400% | 74 | days |
| 1^ | Infliximab | Infliximab | Mom | Maternal | Trimester 3 | 7 | Month IFX undetectable-Month IFX undetectable | months | 7 | N/A | N/A |
| 2 | Infliximab | Infliximab | Mom | Maternal | N/A | 5 | Month IFX undetectable-Month IFX undetectable | months | 5 | N/A | N/A |
| 3#^ | Infliximab | Infliximab | Mom | Maternal | Postpartum | 2 | Month IFX undetectable-Month IFX undetectable | months | 2 | N/A | N/A |
| 4#^ | Infliximab | Infliximab | Mom | Maternal | Postpartum | 7 | Month IFX undetectable-Month IFX undetectable | months | 7 | N/A | N/A |
| 5 | Infliximab | Infliximab | Mom | Maternal | N/A | 2 | Month IFX undetectable-Month IFX undetectable | months | 2 | N/A | N/A |
| 6 | Infliximab | Infliximab | Mom | Maternal | N/A | 3 | Month IFX undetectable-Month IFX undetectable | months | 3 | N/A | N/A |
| 7 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | Month IFX undetectable-Month IFX undetectable | months | 4 | N/A | N/A |
| 8 | Infliximab | Infliximab | Mom | Maternal | N/A | 5 | Month IFX undetectable-Month IFX undetectable | months | 5 | N/A | N/A |
| 9 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | Month IFX undetectable-Month IFX undetectable | months | 4 | N/A | N/A |
| 11 | Infliximab | Infliximab | Mom | Maternal | N/A | 4 | Month IFX undetectable-Month IFX undetectable | months | 4 | N/A | N/A |
| 1#^ | Adalimumab | Adalimumab | Maternal blood | Maternal | Trimester 3, Postpartum | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 6.05 | 7 | Day |
| 2^ | Adalimumab | Adalimumab | Maternal blood | Maternal | Postpartum | 6 wks | ADA(μg/ml) at Birth-Mom: | μg/ml | 1.8399999999999999 | 56 | Day |
| 3#^ | Adalimumab | Adalimumab | Maternal blood | Maternal | Trimester 3, Postpartum | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 3.84 | 7 | Day |
| 4# | Adalimumab | Adalimumab | Maternal blood | Maternal | Trimester 3 | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 0.0 | 42 | Day |
| 5 | Adalimumab | Adalimumab | Maternal blood | Maternal | N/A | 8 wks | ADA(μg/ml) at Birth-Mom: | μg/ml | 2.2 | 35 | Day |
| 6 | Adalimumab | Adalimumab | Maternal blood | Maternal | N/A | 7 wks | ADA(μg/ml) at Birth-Mom: | μg/ml | 3.21 | 42 | Day |
| 7^ | Adalimumab | Adalimumab | Maternal blood | Maternal | Postpartum | 11 wks | ADA(μg/ml) at Birth-Mom: | μg/ml | 3.36 | 42 | Day |
| 8 | Adalimumab | Adalimumab | Maternal blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 16.1 | 1 | Day |
| 9 | Adalimumab | Adalimumab | Maternal blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 2.24 | 49 | Day |
| 10 | Adalimumab | Adalimumab | Maternal blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Mom: | μg/ml | 8.48 | 7 | Day |
| 1#^ | Adalimumab | Adalimumab | Cord blood | Maternal | Trimester 3, Postpartum | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 9.29 | 7 | days |
| 2^ | Adalimumab | Adalimumab | Cord blood | Maternal | Postpartum | 6 wks | ADA(μg/ml) at Birth-Cord: | μg/ml | 5.39 | 56 | days |
| 3#^ | Adalimumab | Adalimumab | Cord blood | Maternal | Trimester 3, Postpartum | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 4.57 | 7 | days |
| 4# | Adalimumab | Adalimumab | Cord blood | Maternal | Trimester 3 | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 0.16 | 42 | days |
| 5 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | 8 wks | ADA(μg/ml) at Birth-Cord: | μg/ml | 4.18 | 35 | days |
| 6 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | 7 wks | ADA(μg/ml) at Birth-Cord: | μg/ml | 4.74 | 42 | days |
| 7^ | Adalimumab | Adalimumab | Cord blood | Maternal | Postpartum | 11 wks | ADA(μg/ml) at Birth-Cord: | μg/ml | 8.94 | 42 | days |
| 8 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 19.7 | 1 | days |
| 9 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 4.95 | 49 | days |
| 10 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Cord: | μg/ml | 8.29 | 7 | days |
| 1#^ | Adalimumab | Adalimumab | Infant blood | Maternal | Trimester 3, Postpartum | N/A | ADA(μg/ml) at Birth-Infant | μg/ml | 6.17 | 7 | days |
| 2^ | Adalimumab | Adalimumab | Infant blood | Maternal | Postpartum | 6 wks | ADA(μg/ml) at Birth-Infant | μg/ml | 6.01 | 56 | days |
| 5 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | 8 wks | ADA(μg/ml) at Birth-Infant | μg/ml | 4.28 | 35 | days |
| 6 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | 7 wks | ADA(μg/ml) at Birth-Infant | μg/ml | 4.87 | 42 | days |
| 7^ | Adalimumab | Adalimumab | Infant blood | Maternal | Postpartum | 11 wks | ADA(μg/ml) at Birth-Infant | μg/ml | 8.09 | 42 | days |
| 8 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Infant | μg/ml | 17.7 | 1 | days |
| 9 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Infant | μg/ml | 4.64 | 49 | days |
| 10 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | N/A | ADA(μg/ml) at Birth-Infant | μg/ml | 9.17 | 7 | days |
| 1#^ | Adalimumab | Adalimumab | Cord blood | Maternal | Trimester 3, Postpartum | N/A | Ratio Cord/Mother-Ratio Cord/Mother | % | 153% | 7 | days |
| 2^ | Adalimumab | Adalimumab | Cord blood | Maternal | Postpartum | 6 wks | Ratio Cord/Mother-Ratio Cord/Mother | % | 293% | 56 | days |
| 3#^ | Adalimumab | Adalimumab | Cord blood | Maternal | Trimester 3, Postpartum | N/A | Ratio Cord/Mother-Ratio Cord/Mother | % | 119% | 7 | days |
| 5 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | 8 wks | Ratio Cord/Mother-Ratio Cord/Mother | % | 190% | 35 | days |
| 6 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | 7 wks | Ratio Cord/Mother-Ratio Cord/Mother | % | 148% | 42 | days |
| 7^ | Adalimumab | Adalimumab | Cord blood | Maternal | Postpartum | 11 wks | Ratio Cord/Mother-Ratio Cord/Mother | % | 266% | 42 | days |
| 8 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | Ratio Cord/Mother-Ratio Cord/Mother | % | 122% | 1 | days |
| 9 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | Ratio Cord/Mother-Ratio Cord/Mother | % | 220% | 49 | days |
| 10 | Adalimumab | Adalimumab | Cord blood | Maternal | N/A | N/A | Ratio Cord/Mother-Ratio Cord/Mother | % | 98% | 7 | days |
| 2^ | Adalimumab | Adalimumab | Infant blood | Maternal | Postpartum | 6 wks | Follow ADA Levels (time)-Follow ADA Levels (time) | μg/ml | 1.94 (6 wks) | N/A | N/A |
| 5 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | 8 wks | Follow ADA Levels (time)-Follow ADA Levels (time) | μg/ml | .934 (8 wks) | N/A | N/A |
| 6 | Adalimumab | Adalimumab | Infant blood | Maternal | N/A | 7 wks | Follow ADA Levels (time)-Follow ADA Levels (time) | μg/ml | 1.31 (7 wks) | N/A | N/A |
| 7^ | Adalimumab | Adalimumab | Infant blood | Maternal | Postpartum | 11 wks | Follow ADA Levels (time)-Follow ADA Levels (time) | μg/ml | 0.529 (11 wks) | N/A | N/A |
| 01 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 18.83 | 14 | Day |
| 02 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 59.57 | 7 | Day |
| 03 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 4.87 | 28 | Day |
| 04 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 20.13 | 17 | Day |
| 05 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 16.49 | 21 | Day |
| 06 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 34.65 | 24 | Day |
| 07 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 1.87 | 28 | Day |
| 08-A | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 6.32 | 42 | Day |
| 09-A | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 42.7 | 6 | Day |
| 10 | Certolizumab pegol | CZP | Mother | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Mother | μg/ml | 37.83 | 5 | Day |
| 01 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 1.65 | 14 | Day |
| 02 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 0.94 | 7 | Day |
| 03 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 1.19 | 28 | Day |
| 04 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 0.57 | 17 | Day |
| 05 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | <0.41 | 21 | Day |
| 06 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 1.66 | 24 | Day |
| 07 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | <0.41 | 28 | Day |
| 08-A | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | <0.41 | 42 | Day |
| B | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | <0.41 | N/A | N/A |
| 09-A | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 1.28 | 6 | Day |
| B | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 1.16 | N/A | N/A |
| 10 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Cord | μg/ml | 0.55 | 5 | Day |
| 01 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | - | 14 | Day |
| 02 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 1.02 | 7 | Day |
| 03 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 1.22 | 28 | Day |
| 04 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 0.44 | 17 | Day |
| 05 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | <0.41 | 21 | Day |
| 06 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 1.58 | 24 | Day |
| 07 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | <0.41 | 28 | Day |
| 08-A | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 0.58 | 42 | Day |
| B | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | <0.41 | N/A | N/A |
| 09-A | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 1.34 | 6 | Day |
| B | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 1.18 | N/A | N/A |
| 10 | Certolizumab pegol | CZP | Infant | Maternal | Delivery | N/A | Day of Birth CZP concentrations (μg/ml)-Lower limit of Quantification <0.41-Infant | μg/ml | 0.6 | 5 | Day |
| 01 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 8.8% | 14 | Day |
| 02 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 1.6% | 7 | Day |
| 03 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 24% | 28 | Day |
| 04 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 2.8% | 17 | Day |
| 05 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 2.5% | 21 | Day |
| 06 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 4.8% | 24 | Day |
| 07 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 22% | 28 | Day |
| 08-A | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 6.4% | 42 | Day |
| B | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 6.4% | N/A | N/A |
| 09-A | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 3.0% | 6 | Day |
| B | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 2.7% | N/A | N/A |
| 10 | Certolizumab pegol | CZP | Cord | Maternal | Delivery | N/A | Ratio Cord/Mother-Unnamed: 5_level_1-Unnamed: 5_level_2 | % | 1.5% | 5 | Day |
| 01 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 33.4 | 14 | days |
| 02 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 51.3 | 7 | days |
| 04 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 34.7 | 17 | days |
| 05 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 27.7 | 21 | days |
| 06 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 34.4 | 24 | days |
| 08-A | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 11.1 | 42 | days |
| 09-A | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 62.1 | 6 | days |
| 10 | Certolizumab pegol | PEG | Mother | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Mother | μg/ml | 74.9 | 5 | days |
| 01 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 14 | Day |
| 02 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 7 | Day |
| 03 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 28 | Day |
| 04 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 17 | Day |
| 05 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 21 | Day |
| 06 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 24 | Day |
| 07 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 28 | Day |
| 08-A | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 42 | Day |
| B | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | N/A | N/A |
| 09-A | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 6 | Day |
| 10 | Certolizumab pegol | PEG | Cord | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Cord | μg/ml | * | 5 | Day |
| 01 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 14 | Day |
| 02 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 7 | Day |
| 03 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 28 | Day |
| 04 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 17 | Day |
| 05 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | No sample | 21 | Day |
| 06 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 24 | Day |
| 07 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 28 | Day |
| 08-A | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 42 | Day |
| B | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | N/A | N/A |
| 09-A | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 6 | Day |
| 10 | Certolizumab pegol | PEG | Infant | Maternal | Delivery | N/A | Day of Birth PEG concentrations (μg/ml)-Lower limit of Quantification <9-Infant | μg/ml | * | 5 | Day |


Reasoning Process:

The curated table is incomplete: it includes only Infliximab data and excludes all Adalimumab, Certolizumab pegol, and PEG data that appear in the source tables. Therefore it does not accurately reflect the full pharmacokinetic information provided.

--------------------


/no_think
"""


def test_generate_code():
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
    logger.info("content: " + raw.content)
    logger.info("additional_kwargs:", raw.additional_kwargs)
    assert raw.content.strip() != "" or raw.additional_kwargs.get("reasoning")