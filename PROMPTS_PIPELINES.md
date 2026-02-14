# Prompt Catalog (Workflow-Organized)

Organized by the PK/PE curation workflow, followed by pipeline-specific prompt groups.

## Workflow Overview

1. PKPE Identification Step — `extractor/agents/pk_pe_agents/pk_pe_identification_step.py`: Determine whether the paper is PK, PE, Both, or Neither using title/abstract.
2. PKPE Design Step — `extractor/agents/pk_pe_agents/pk_pe_design_step.py`: Select the most appropriate pipeline tools based on paper type and content.
3. Run Selected Pipelines: Execute the chosen pipeline prompts to curate tables from the paper.
4. PKPE Curated Tables Verification Step — `extractor/agents/pk_pe_agents/pk_pe_verification_step.py`: Validate curated tables; if incorrect, produce reasoning and fix suggestions.
5. PKPE Curated Tables Correction Code Step — `extractor/agents/pk_pe_agents/pk_pe_correction_code_step.py`: Generate and run Python code to apply the suggested corrections.

## Shared Instruction / Error Prompt Fragments

### COT User Instruction
Source: `extractor/constants.py`

```text
Now, let's start.
```

### Previous Errors Appendix
Source: `extractor/prompts_utils.py` (used by common step classes to append prior errors)

```text
---
### **Previous Errors to Avoid**

You must pay close attention to the following corrections from previous runs. Do not repeat these specific errors.

{previous_errors}
```

## Step 1 — PKPE Identification

### extractor/agents/pk_pe_agents/pk_pe_identification_step.py

#### PKPE_IDENTIFICATION_SYSTEM_PROMPT

```text
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
```

## Step 2 — PKPE Design

### extractor/agents/pk_pe_agents/pk_pe_design_step.py

#### PKPE_DESIGN_SYSTEM_PROMPT

```text
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
```

## Step 3 — Pipeline Prompts

Prompts used by the pipelines selected in Step 2.

### PK Summary

#### extractor/agents/pk_summary/pk_sum_drug_info_agent.py

##### DRUG_INFO_PROMPT

```text
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
This table is from a paper titled: **{paper_title}**
Carefully analyze the paper title and the table and follow these steps:  
(1) Identify how many unique [Drug name, Analyte, Specimen] combinations are present in the table.  
Drug name is the name of the drug mentioned in the study.
Analyte is the substance measured in the study, which can be the primary drug, its metabolite, or another drug it affects, etc. When filling in "Analyte," only enter the name of the substance.
Specimen is the type of sample.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   [["Lorazepam", "Lorazepam", "Plasma"], ["Lorazepam", "Lorazepam", "Urine"]] (example)  
(3) Verify the source of each [Drug Name, Analyte, Specimen] combination before including it in your answer.  
(4) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
(5) If none of the elements are explicitly stated in the table or caption, infer them from the **paper title**.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_summary/pk_sum_drug_matching_agent.py

##### MATCHING_DRUG_PROMPT

```text
Analyze the following pharmacokinetics (PK) data tables and perform the specified matching operation:

MAIN TABLE (PK Data):
{processed_md_table_aligned}

Caption: {caption}

SUBTABLE 1 (Extracted from Main Table):
{processed_md_table_aligned_with_1_param_type_and_value}

SUBTABLE 2 (Drug-Analyte-Specimen Combinations):
{processed_drug_md_table}

---
### **TASK:**
1. For each of rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} in Subtable 1, find the BEST matching row in Subtable 2 based on:
   - For each row in Subtable 1, find **the best matching one** row in Subtable 2
   - Context from the table caption about the drug (e.g. lorazepam)

2. Processing Rules:
   - Only process rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} from Subtable 1 (exactly {md_table_aligned_with_1_param_type_and_value_row_num} rows total)
   - Return indices of matching Subtable 2 rows as a Python list of integers
   - If no clear best match is identified for a given row, default to using -1. Important: This default should only be applied when no legitimate match exists after thorough evaluation of all available data.
   - Example output format: [0, 1, 2, 3, 4, 5, 6, ...]

### ** Important Instructions:**
   - You **must follow** the following steps to match the row in Subtable 1 to the row in Subtable 2:
     For each row in Subtable 1, 
      * First find the corresponding row in **main table** for the row in Subtable 1 according to row index (row index in main table is the same as the row index in Subtable 1), 
      * The row in main table provide more context,then find the best matching row in **Subtable 2** according to the row in main table.
   - As SUBTABLE 1 is extracted from MAIN TABLE in row order, if you cannot determine the best matching row in Subtable 2 for a given row in Subtable 1, 
     you can infer the best matching by referring to the row before it or after it.
     For example, main table is like this:
     | Drug Name      | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- | --------------- |
     | B1             | 1               | urine           | -               |
     | B1             | 2               | urine           | -               |
     | B1             | 3               | urine           | -               |
     | B2             | 1               | urine           | -               |
     | B2             | 2               | urine           | -               |
     | B2             | 3               | urine           | -               |
     | B3             | 1               | urine           | -               |
     | B3             | 2               | urine           | -               |
     | B3             | 3               | urine           | -               |
     then Subtable 1 is like this:
     | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -             |
     then Subtable 2 is like this:
     | Drug Name       | Analyte       | Specimen       |
     | -------------- | --------------- | --------------- |
     | B1             | B1              | urine           |
     | B2             | B2              | urine           |
     | B3             | B3              | urine           |
     
     1. For the row 0, 1 and 2 in Subtable 1, the best matching row in Subtable 2 is 0 (index).
     As Subtable 1 is extracted from main table in row order, the corresponding rows in main table for row 0, 1 and 2 in Subtable 1 are 0, 1 and 2 (index).
     Then, we can determin their drug name from main table are B1, so the best matching rows in Subtable 2 for row 0, 1 and 2 in main table are [0, 0, 0] (index)
     Thus, the best matching rows in Subtable 2 for the row 0, 1 and 2 in Subtable 1 are [0, 0, 0] (index).
     2. For the row 3, 4 and 5 in Subtable 1, the best matching row in Subtable 2 is 1 (index).
     Likewise, as the Subtable 1 is extracted from main table in row order, the corresponding rows in main table for the row 3, 4 and 5 are 3, 4 and 5 (index).
     Thus, we can determine their drug name from main table are B2, so the best matching rows in Subtable 2 for the row 3, 4 and 5 in main table are [1, 1, 1] (index).
     3. For the row 6, 7 and 8 in Subtable 1, the best matching row in Subtable 2 is 2 (index).
     Likewise, as Subtable 1 is extracted from main table in row order, we can determine the best matching rows in main table for the row 6, 7 and 8 in Subtable 1 are 6, 7 and 8 (index).
     Thus, we can determine their drug name from main table are B3, so the best matching rows in Subtable 2 for the row 6, 7 and 8 in main table are [2, 2, 2] (index).

     So, we get the best matching rows in Subtable 2 for the row 0, 1, 2, 3, 4, 5, 6, 7 and 8 in Subtable 1 are [0, 0, 0, 1, 1, 1, 2, 2, 2] (index).
```

#### extractor/agents/pk_summary/pk_sum_header_categorize_agent.py

##### HEADER_CATEGORIZE_PROMPT

```text
The following table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
{column_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all column headers and categorize each one into one of the following groups:  
   - **"Parameter type"**: Columns that describe the type of pharmacokinetic parameter.  
   - **"Parameter unit"**: Columns that **only** specify the unit of the parameter type. e.g. "fentanyl (ng/ml)" is not Parameter unit.  
   - **"Parameter value"**: Columns that contain numerical parameter values.  
   - **"P value"**: Columns that represent statistical P values.  
   - **"Uncategorized"**: Columns that do not fit into any of the above categories.  
(2) if a column is only about the subject number, it is considered as "Uncategorized"
(3) Return a categorized headers dictionary where each key is a column header, and the corresponding value is its assigned category, e.g.
{categorized_headers_example}
```

#### extractor/agents/pk_summary/pk_sum_individual_data_del_agent.py

##### INDIVIDUAL_DATA_DEL_PROMPT

```text
There is now a table related to pharmacokinetics (PK). 
{processed_md_table}
Carefully examine the table and follow these steps:
(1) Remove any information that pertains to **specific individuals**, such as individual-level results or personally identifiable data.
(2) **Do not remove** summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
Please return the result with the following format:
processed: boolean value, False represents the table have already meets the requirement, don't need to be processed. Otherwise, it will be True
row_list: an array of row indices that satisfy the requirement, that is the rows have no individual-level results or personally identifiable data.
col_list: an array of column names that satisfy the requirement, that is the columns in the above rows have no individual-level results or personally identifiable data.
```

#### extractor/agents/pk_summary/pk_sum_param_type_align_agent.py

##### PARAMETER_TYPE_ALIGN_PROMPT

```text
There is now a table related to pharmacokinetics (PK). 
{md_table_summary}
The extracted table headers are as follows:
{md_table_summary_header}
Carefully examine the pharmacokinetics (PK) table and follow these steps to determine how the PK parameter type is represented:
(1) Identify how the PK parameter type (e.g., Cmax, tmax, t1/2, etc.) is structured in the table:
Please answer in the following format:
col_name: column name, it represents the PK parameter type serves as the row header or is listed under the specific column. If the PK parameter type is represented as column headers, return None.
(2) Ensure a thorough analysis of the table structure before selecting your answer.
```

#### extractor/agents/pk_summary/pk_sum_param_type_unit_extract_agent.py

##### UNIT_EXTRACTION_PROMPT

```text
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted some columns to create Subtable 1:  
Below is Subtable 1:
{processed_md_sub_table}
Please note that the column "{key_with_parameter_type}" in Subtable 1 roughly represents the parameter type.
Carefully analyze the table and follow these steps:  
(1) Refer to the "{key_with_parameter_type}" column in Subtable 1 to construct two separate lists: one for a new "Parameter type" and another for "Parameter unit". If the information in Subtable 1 is too coarse or ambiguous, you may need to refer to the main table and its caption to refine and clarify your summarized "Parameter type" and "Parameter unit".
(2) Return a tuple containing two lists:  
    - The first list should contain the extracted "Parameter type" values.  
    - The second list should contain the corresponding "Parameter unit" values.  
(3) **Strictly ensure that you process only rows 0 to {row_max_index} from the column "{key_with_parameter_type}".**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less. 
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row. 
(5) The returned list should be like this:  
    (["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...])>>
```

#### extractor/agents/pk_summary/pk_sum_param_value_agent.py

##### PARAMETER_VALUE_PROMPT

```text
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following columns to create Subtable 1:  
{extracted_param_types}  
Below is Subtable 1:
{processed_md_table_aligned_with_1_param_type_and_value}
Please review the information in Subtable 1 row by row and complete Subtable 2.
Subtable 2 should have the following column headers only:  

**Main value, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound, P value** 

Main value: the value of main parameter (not a range). 
Statistics type: the statistics method to summary the Main value, like 'Mean,' 'Median,' 'Count,' etc. **This column is required and must be filled in.**
Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' 'CV%,' etc.
Variation value: the value (not a range) that corresponds to the specific variation.
Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI,' 'Range,' 'IQR,' etc.
Lower bound: the lower bound value of the interval.
Upper bound: is the upper bound value of the interval.
P value: P-value.

Please Note:
(1) An interval consisting of two numbers must be placed separately into the Low limit and High limit fields; it is prohibited to place it in the Variation value field.
(2) For values that do not need to be filled, enter "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_aligned_with_1_param_type_and_value_max_row_index} from the Subtable 1 (which has {md_table_aligned_with_1_param_type_and_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row.
(3) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. Ensure all values are converted to strings.
(4) **Absolutely no calculations are allowed—every value must be taken directly from Subtable 1 without any modifications.**  
(5) **Important** The **P value** must be extracted directly from the main table.
  - First, identify the corresponding column in main table, then extract **P value** from that column.
(6) The final list should be like this:
[["0.162", "Mean", "SD", "0.090", "N/A", "N/A", "N/A", ".67"], ["0.428", "Mean", "SD", "0.162", "N/A", "N/A", "N/A", ".015"]]
```

#### extractor/agents/pk_summary/pk_sum_patient_info_agent.py

##### PATIENT_INFO_PROMPT

```text
the following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
Carefully analyze the table, **row by row and column by column**, and follow these steps:
(1) Identify how many unique [Population, Pregnancy stage, Subject N] combinations are present in the table.
Population is the patient age group.
Pregnancy stage is the pregnancy stages of patients mentioned in the study.
Subject N represents the number of subjects corresponding to the specific parameter or the number of samples with quantifiable levels of the respective analyte.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>.
(3) Ensure that all elements in the list of lists are **strings**, especially Subject N, which must be enclosed in double quotes (`""`).
(4) Verify the source of each [Population, Pregnancy stage, Subject N] combination before including it in your answer.
(5) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. **YOU MUST** include all the Ns for each age group.
    - Specifically, **YOU MUST** explain every number, in this list: {int_list} to determine if it should be listed in Subject N.
    - For example, if a population group has a Subject N of 8, but further analysis shows that 5, 6, and 7 of the 8 subjects correspond to different parameter values, then 5, 6, 7, and 8 must all be included as Subject N in different combinations in the final answer.
    - Fill in "N/A" when you don't know the exact N.
    - Important: Do not confuse Patient ID with Subject N. Subject N refers to the total number of patients.
(6) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_summary/pk_sum_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population" - "Pregnancy stage" - "Subject N," as follows:
{processed_md_table_patient}

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
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
        
    - **Subject N**: The number of subjects corresponding to the specific population.

(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["N/A", "N/A", "N/A", "15"], ...]

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_patient_max_row_index} from the Subtable 1 (which has {md_table_patient_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Subject N for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_summary/pk_sum_patient_matching_agent.py

##### MATCHING_PATIENT_SYSTEM_PROMPT

```text
You are given a main table containing pharmacokinetics (PK) data:  
**Main Table:**  
{processed_md_table_aligned}  

**Caption:**  
{caption}
From this main table, I have extracted a subset of rows based on specific parameters to form **Subtable 1**:  
{extracted_param_types}  

**Subtable 1 (Single Parameter Type and single Parameter Value per Row):**  
{processed_md_table_aligned_with_1_param_type_and_value}  

I have also compiled **Subtable 2**, where each row corresponds to a unique combination of:  
**"Population" – "Pregnancy stage" – "Subject N"**  
{processed_patient_md_table}  

### Task:
Carefully analyze the tables and follow these instructions step by step:

1. **Match Each Row in Subtable 1 to Subtable 2:**
   - For each row in Subtable 1 (rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}), find the **best matching row** in Subtable 2.
   - First, find the corresponding row in the **Main Table** using the **Parameter Value** or **P Value** from Subtable 1.
   - Then, use the associated **"Subject N"** value to identify the matching row in Subtable 2.

2. **Strict Row Range:**
   - Only process rows **0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}** in Subtable 1.
   - Your final output list must include exactly the same number of entries as there are rows in Subtable 1.

3. **Handle Variations in "Subject N":**
   - Within each population group, "Subject N" may vary across different parameters due to data availability.
   - Match each row using the correct "Subject N" from the context of the main table. For example, if the group has a total N of 10 but a specific parameter only applies to 9, then 9 is the correct "Subject N" to use for matching.

4. **Output Format:**
   - Return a Python-style list containing the row indices (integers) of Subtable 2 that best match each row in Subtable 1.
   - Do not sort or deduplicate the list. The output should follow the order of Subtable 1:
     ```
     [matched_index_row_0, matched_index_row_1, ..., matched_index_row_N]
     ```

5. **If No Match Found:**
   - If a row in Subtable 1 cannot be matched even after applying all criteria, return `-1` for that row.
   - Use this only as a last resort.
```

#### extractor/agents/pk_summary/pk_sum_split_by_col_agent.py

##### SPLIT_BY_COLUMNS_PROMPT

```text
There is a table related to pharmacokinetics (PK):
{processed_md_table}

This table contains multiple columns, categorized as follows:
{mapping_str}

This table can be split into multiple sub-tables {situation_str}.
Please follow these steps:
  (1) Carefully review all columns and analyze their relationships to determine logical groupings.
  (2) Ensure that each group contains exactly one 'Parameter type' column and at most one 'P value' column.

Return the results as a list of lists, where each inner list represents a sub-table with its included columns, like this:
[["ColumnA", "ColumnB", "ColumnC", "ColumnG"], ["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]
```

#### extractor/agents/pk_summary/pk_sum_time_unit_agent.py

##### TIME_AND_UNIT_PROMPT

```text
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted some information to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_post_processed}
Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Time value, Time unit].  
- **Time Value:** A specific moment (numerical or time range) when the row of data is recorded, or a drug dose is administered.  
  - Examples: Sampling times, dosing times, or reported observation times.  
- **Time Unit:** The unit corresponding to the recorded time point (e.g., "Hour", "Min", "Day").  
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour"], ["10", "Min"], ["N/A", "N/A"]]` (example)
(3) Strictly ensure that you process only rows 0 to {md_data_post_processed_max_row_index} from the Subtable 1 (which has {md_data_lines_after_post_process_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Time value, Time unit] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Time value, Time unit] combinations are found, return the default output:**  
`[["N/A", "N/A"]]`

**Examples:**
Include:  
   - "0-12" (indicating a dosing period)  
   - "24" (indicating a time of sample collection)  
   - "5 min" (indicating a measured event)  

Do NOT include:  
   - "Tmax" (this is a pharmacokinetic parameter, NOT a recorded time)
   - "T½Beta(hr)" values (half-life parameter value, not a recorded time) 
   - "Beta(hr)" values (elimination rate constant)
```


### PK Individual

#### extractor/agents/pk_individual/pk_ind_drug_info_agent.py

##### DRUG_INFO_PROMPT

```text
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
This table is from a paper titled: **{paper_title}**
Carefully analyze the paper title and the table and follow these steps:  
(1) Identify how many unique [Drug name, Analyte, Specimen] combinations are present in the table.  
Drug name is the name of the drug mentioned in the study.
Analyte is the substance measured in the study, which can be the primary drug, its metabolite, or another drug it affects, etc. When filling in "Analyte," only enter the name of the substance.
Specimen is the type of sample.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   [["Lorazepam", "Lorazepam", "Plasma"], ["Lorazepam", "Lorazepam", "Urine"]] (example)  
(3) Verify the source of each [Drug Name, Analyte, Specimen] combination before including it in your answer.  
(4) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
(5) If none of the elements are explicitly stated in the table or caption, infer them from the **paper title**.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_individual/pk_ind_drug_matching_agent.py

##### MATCHING_DRUG_PROMPT

```text
Analyze the following pharmacokinetics (PK) data tables and perform the specified matching operation:

MAIN TABLE (PK Data):
{processed_md_table_aligned}

Caption: {caption}

SUBTABLE 1 (Extracted from Main Table):
{processed_md_table_aligned_with_1_param_type_and_value}

SUBTABLE 2 (Drug-Analyte-Specimen Combinations):
{processed_drug_md_table}

---

### **TASK:**
1. For each of rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} in Subtable 1, find the BEST matching row in Subtable 2 based on:
   - For each row in Subtable 1, find **the best matching one** row in Subtable 2
   - Context from the table caption about the drug (e.g. lorazepam)

2. Processing Rules:
   - Only process rows 0-{max_md_table_aligned_with_1_param_type_and_value_row_index} from Subtable 1 (exactly {md_table_aligned_with_1_param_type_and_value_row_num} rows total)
   - Return indices of matching Subtable 2 rows as a Python list of integers
   - If no clear best match is identified for a given row, default to using -1. Important: This default should only be applied when no legitimate match exists after thorough evaluation of all available data.
   - Example output format: [0, 1, 2, 3, 4, 5, 6, ...]

### ** Important Instructions:**
   - You **must follow** the following steps to match the row in Subtable 1 to the row in Subtable 2:
     For each row in Subtable 1, 
      * First find the corresponding row in **main table** for the row in Subtable 1 according to row index (row index in main table is the same as the row index in Subtable 1), 
      * The row in main table provide more context,then find the best matching row in **Subtable 2** according to the row in main table.
   - As SUBTABLE 1 is extracted from MAIN TABLE in row order, if you cannot determine the best matching row in Subtable 2 for a given row in Subtable 1, 
     you can infer the best matching by referring to the row before it or after it.
     For example, main table is like this:
     | Drug Name      | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- | --------------- |
     | B1             | 1               | urine           | -               |
     | B1             | 2               | urine           | -               |
     | B1             | 3               | urine           | -               |
     | B2             | 1               | urine           | -               |
     | B2             | 2               | urine           | -               |
     | B2             | 3               | urine           | -               |
     | B3             | 1               | urine           | -               |
     | B3             | 2               | urine           | -               |
     | B3             | 3               | urine           | -               |
     then Subtable 1 is like this:
     | Patient ID      | Parameter Type | Parameter Value |
     | -------------- | --------------- | --------------- |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -               |
     | 1               | urine           | -               |
     | 2               | urine           | -               |
     | 3               | urine           | -             |
     then Subtable 2 is like this:
     | Drug Name       | Analyte       | Specimen       |
     | -------------- | --------------- | --------------- |
     | B1             | B1              | urine           |
     | B2             | B2              | urine           |
     | B3             | B3              | urine           |
     
     1. For the row 0, 1 and 2 in Subtable 1, the best matching row in Subtable 2 is 0 (index).
     As Subtable 1 is extracted from main table in row order, the corresponding rows in main table for row 0, 1 and 2 in Subtable 1 are 0, 1 and 2 (index).
     Then, we can determin their drug name from main table are B1, so the best matching rows in Subtable 2 for row 0, 1 and 2 in main table are [0, 0, 0] (index)
     Thus, the best matching rows in Subtable 2 for the row 0, 1 and 2 in Subtable 1 are [0, 0, 0] (index).
     2. For the row 3, 4 and 5 in Subtable 1, the best matching row in Subtable 2 is 1 (index).
     Likewise, as the Subtable 1 is extracted from main table in row order, the corresponding rows in main table for the row 3, 4 and 5 are 3, 4 and 5 (index).
     Thus, we can determine their drug name from main table are B2, so the best matching rows in Subtable 2 for the row 3, 4 and 5 in main table are [1, 1, 1] (index).
     3. For the row 6, 7 and 8 in Subtable 1, the best matching row in Subtable 2 is 2 (index).
     Likewise, as Subtable 1 is extracted from main table in row order, we can determine the best matching rows in main table for the row 6, 7 and 8 in Subtable 1 are 6, 7 and 8 (index).
     Thus, we can determine their drug name from main table are B3, so the best matching rows in Subtable 2 for the row 6, 7 and 8 in main table are [2, 2, 2] (index).

     So, we get the best matching rows in Subtable 2 for the row 0, 1, 2, 3, 4, 5, 6, 7 and 8 in Subtable 1 are [0, 0, 0, 1, 1, 1, 2, 2, 2] (index).
```

#### extractor/agents/pk_individual/pk_ind_param_type_align_agent.py

##### PARAMETER_TYPE_ALIGN_PROMPT

```text
There is now a table related to pharmacokinetics (PK). 
{md_table_individual}
The extracted table headers are as follows:
{md_table_individual_header}
Carefully examine the pharmacokinetics (PK) table and follow these steps to determine how the PK parameter type is represented:
(1) Identify how the PK parameter type (e.g., Cmax, tmax, t1/2, etc.) is structured in the table:
Please answer in the following format:
col_name: column name, it represents the PK parameter type serves as the row header or is listed under the specific column. If the PK parameter type is represented as column headers, return None.
(2) Ensure a thorough analysis of the table structure before selecting your answer.
```

#### extractor/agents/pk_individual/pk_ind_param_type_unit_extract_agent.py

##### UNIT_EXTRACTION_PROMPT

```text

You are an expert in pharmacokinetics (PK) data interpretation and table normalization.

You are given:

**Main PK Table (reference only):**
{processed_md_table_aligned}

**Table Caption (reference only):**
{caption}

**Subtable 1 (PRIMARY input):**
{processed_md_sub_table}

The column **"{key_with_parameter_type}"** in Subtable 1 *approximately* represents the pharmacokinetic **parameter type**, but it may be incomplete, abbreviated, or ambiguous.

---

## Task

For **each row in Subtable 1 (rows 0 to {row_max_index}, inclusive)**, extract and construct **three aligned outputs**:

1. **Parameter type**
2. **Parameter unit**
3. **Parameter value**

You must return these as **three parallel lists**, preserving row order exactly.

---

## Extraction Rules (Follow in Order)

### Step 1 — Row Scope (STRICT)

* Process **only** rows `0 … {row_max_index}` from Subtable 1.
* The number of output entries **must exactly equal** the number of processed rows.
* Do not skip, add, reorder, or merge rows.

---

### Step 2 — Parameter Type Construction (PRIMARY FOCUS)

Use the column **"{key_with_parameter_type}"** as the starting point.

#### 2a. No-Loss Normalization Rule (MANDATORY)

* **Do NOT drop, simplify away, or merge informative components.**
* Preserve **all meaningful tokens**, including:

  * biological matrix (e.g., plasma, serum, milk)
  * subject/context (e.g., maternal, fetal, cord)
  * timing or condition (e.g., trough, peak)
  * method or sampling source

#### 2b. Composite Values

If the parameter type appears as a tuple, list, or nested structure:

* **Flatten and concatenate all elements in order**
* Use `"-"` as the delimiter
* Remove brackets and quotes only
* Preserve original wording and capitalization

**Examples:**

* `('Cordocentesis', 'Serum')` → `Cordocentesis-Serum`
* `('Maternal', 'Plasma', 'Trough')` → `Maternal-Plasma-Trough`
* `(('Umbilical', 'Vein'), 'Plasma')` → `Umbilical-Vein-Plasma`

If the parameter type is already a single string:

* Use it **as-is**, trimming surrounding whitespace only.

---

### Step 3 — Parameter Type Semantics

* Keep the **core PK concept clear and simple**, such as:

  * `Concentration`
  * `Cmax`
  * `Tmax`
  * `AUC`
* While doing so, **do NOT remove contextual qualifiers** required by the No-Loss Rule.

Correct example:

* `Cmax-Maternal-Plasma`
* `Concentration-Cordocentesis-Serum`

Incorrect example:

* Reducing everything to just `Concentration`

---

### Step 4 — Parameter Unit and Value Extraction

* Prefer values and units **explicitly present in Subtable 1**.
* If unit or value is ambiguous or incomplete:

  * Refer to the **main table and caption** ONLY to clarify or refine.
* Do **not infer or invent** values or units that are not supported by the provided tables.

---

### Step 5 — Missing or Unextractable Rows

* If **any** of the three fields (type, unit, value) cannot be confidently extracted for a row:

  * Output `"N/A"` for **all three fields** for that row.

---

## Output Format (STRICT)

Return **exactly one tuple** containing **three lists**:

```json
{{
  "reasoning_process": "<Reasoning process>",
  "extracted_param_units": {{
    "parameter_types": ["Parameter type 1", "Parameter type 2", ...],
    "parameter_units": ["Parameter unit 1", "Parameter unit 2", ...],
    "parameter_values": ["Parameter value 1", "Parameter value 2", ...]
  }}
}}
```

Constraints:

* All three lists must have **identical length**
* List indices must correspond to the same row in Subtable 1
* Output **only** the tuple — no explanations, no markdown, no extra text

---

## Priority Rules (Highest to Lowest)

1. Row count and alignment correctness
2. No-Loss Normalization Rule
3. Fidelity to Subtable 1
4. Clarification via main table/caption
5. Simplicity of PK terminology without information loss

---
```

#### extractor/agents/pk_individual/pk_ind_patient_info_agent.py

##### PATIENT_INFO_PROMPT

```text
You are a **pharmacokinetics (PK) domain expert** specializing in structured data extraction from biomedical tables.

You are provided with:

* A pharmacokinetics (PK) data table in Markdown format:
{processed_md_table}
* The table caption:
{caption}

Your task is to **systematically extract population-level identifiers** from the table by carefully examining it **row by row and column by column**.

---

### **Extraction Task**

#### **Step 1: Identify Unique Combinations**

Determine all **unique combinations** of the following three elements present in the table:

1. **Patient ID**

   * Refers to an identifier explicitly assigned to a **unique individual patient** in the table.
   * Use the **exact text** as it appears in the table.
   * If no explicit individual patient identifier exists:

     * Infer a **unique unit** that clearly distinguishes rows (e.g., subject number, case number, mother–infant pair, cohort label).
     * Use this inferred unit as the Patient ID.
   * If **neither an individual patient nor a reasonable unique unit can be identified**, do **not** fabricate identifiers.

2. **Population**

   * Refers to the **age-based or demographic population group** (e.g., adult, neonate, pregnant women).
   * If not explicitly stated or reasonably inferable, use `"N/A"`.

3. **Pregnancy Stage**

   * Refers to pregnancy-related timing or stage (e.g., trimester, delivery, postpartum).
   * If not explicitly stated or reasonably inferable, use `"N/A"`.

---

#### **Step 2: Validation Rules**

* Verify that **each extracted combination is supported by the table content or caption**.
* If information is missing:

  * First attempt to infer it using contextual clues within the table, caption, or standard PK study conventions.
  * Use `"N/A"` **only if inference is not reasonably possible**.
* If you **cannot identify any individual Patient ID or infer any unique unit**, return an **empty list** and do **not** guess or fabricate data.

---

### **Output Requirements**

Your response **must be valid JSON** and **must exactly match** the structure below:

```json
{{
  "reasoning_process": "<1–2 concise sentences summarizing how the combinations were identified>",
  "patient_combinations": [
    ["Patient ID", "Population", "Pregnancy stage"]
  ]
}}
```

#### **Strict Formatting Rules**

* `patient_combinations` must be a **list of lists**.
* **All elements must be strings**, including `"N/A"`.
* **Patient ID values must be enclosed in double quotes**.
* Do **not** include any explanatory text outside the JSON object.
* Do **not** include Markdown, comments, or code fences.

---
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_individual/pk_ind_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
The following main table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population" - "Pregnancy stage" - "Subject N," as follows:
{processed_md_table_patient}

### **Task**
Carefully analyze the tables and follow these steps to refine Subtable 1 into a more detailed Subtable 2:  

(1) Identify all unique combinations of **[Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]** from the table.
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
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.

(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**.  
   - Your response should be enclosed in **double angle brackets** `<< >>` and formatted as a **single line**.

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_patient_max_row_index} from the Subtable 1 (which has {md_table_patient_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Subject N for each row in Subtable 2 must be the same as in Subtable 1.

### **Output**
Your output must be in compact json format, and **must exactly match** the following format:
{{
    "reasoning_process": <a string, detailed explanation of the reasoning process>,
    "refined_patient_combinations": <a list of lists of unique combinations [Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]
}}
```

#### extractor/agents/pk_individual/pk_ind_patient_matching_agent.py

##### MATCHING_PATIENT_SYSTEM_PROMPT

```text
You are given a main table containing pharmacokinetics (PK) data:  
**Main Table:**  
{processed_md_table_aligned}  

**Caption:**  
{caption}
From this main table, I have extracted a subset of rows based on specific parameters to form **Subtable 1**:  
{extracted_param_types}  

**Subtable 1 (Single Parameter Type and single Parameter Value per Row):**  
{processed_md_table_aligned_with_1_param_type_and_value}  

I have also compiled **Subtable 2**, where each row corresponds to a unique combination of:  
**"Patient ID" - "Population" – "Pregnancy stage"**  
{processed_patient_md_table}  

### Task:
Carefully analyze the tables and follow these instructions step by step:

1. **Match Each Row in Subtable 1 to Subtable 2:**
   - For each row in Subtable 1 (rows 0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}), find the **best matching row** in Subtable 2.
   - First, find the corresponding row in the **Main Table** using the **Parameter Value** from Subtable 1.
   - Then, use the associated **"Patient ID"** value to identify the matching row in Subtable 2.

2. **Strict Row Range:**
   - Only process rows **0 to {max_md_table_aligned_with_1_param_type_and_value_row_index}** in Subtable 1.
   - Your final output list must include exactly the same number of entries as there are rows in Subtable 1.

3. **Output Format:**
   - Return a Python-style list containing the row indices (integers) of Subtable 2 that best match each row in Subtable 1.
   - Do not sort or deduplicate the list. The output should follow the order of Subtable 1:
     ```
     [matched_index_row_0, matched_index_row_1, ..., matched_index_row_N]
     ```

5. **If No Match Found:**
   - If a row in Subtable 1 cannot be matched even after applying all criteria, return `-1` for that row.
   - Use this only as a last resort.

### ** Important Instructions:**
   - You **must follow** the following steps to match the row in Subtable 1 to the row in Subtable 2:
     For each row in Subtable 1, 
      * First find the corresponding row in **main table** for the row in Subtable 1 according to row index (row index in main table is the same as the row index in Subtable 1), 
      * The row in main table provide more context,then find the best matching row in **Subtable 2** according to the row in main table.
   - As SUBTABLE 1 is extracted from MAIN TABLE in row order, if you cannot determine the best matching row in Subtable 2 for a given row in Subtable 1, 
     you can infer the best matching by referring to the row before it or after it.

### ** Example:**
    
    Main Table:
     | Drug Name      | Patient ID      | Delivery         | Parameter Value | Cordocentesis | Parmeter Value |
     | -------------- | --------------- | --------------- | --------------- | ------------- | ------------- |
     | B1             | 1               | urine           | -               | Amnion        | -              |
     | B1             | 2               | urine           | -               | Amnion        | -              |
     | B1             | 3               | urine           | -               | Amnion        | -              |
     | B2             | 1               | urine           | -               | Amnion        | -              |
     | B2             | 2               | urine           | -               | Amnion        | -              |
     | B2             | 3               | urine           | -               | Amnion        | -              |
     | B3             | 1               | urine           | -               | Amnion        | -              |
     | B3             | 2               | urine           | -               | Amnion        | -              |
     | B3             | 3               | urine           | -               | Amnion        | -              |
    
    Subtable 1:
    | Patient ID | Parameter Type   | Parameter Value |
    | 1          | Delivery - urine | -               |
    | 2          | Delivery - urine | -               |
    | 3          | Delivery - urine | -               |
    | 1          | Delivery - urine | -               |
    | 2          | Delivery - urine | -               |
    | 3          | Delivery - urine | -               |
    | 1          | Delivery - urine | -               |
    | 2          | Delivery - urine | -               |
    | 3          | Delivery - urine | -               |

    Subtable 2:
    | Patient ID | Population   | Pregnancy Stage      | Pediatric/Gestational age |
    | 1          | N/A          | Delivery             | N/A                       |
    | 2          | N/A          | Delivery             | N/A                       |
    | 3          | N/A          | Delivery             | N/A                       |
    | 1          | N/A          | Cordocentesis        | N/A                       |
    | 2          | N/A          | Cordocentesis        | N/A                       |
    | 3          | N/A          | Cordocentesis        | N/A                       |

    1. For the row 0, 1 and 2 in Subtable 1, the best matching row in Subtable 2 is 0, 1, and 2 (index).
    As Subtable 1 is extracted from main table in row order, the corresponding rows in main table for row 0, 1 and 2 in Subtable 1 are 0, 1 and 2 (index).
    Then, we can determin their patient id from main table are 1, 2 and 3, so the best matching rows in Subtable 2 for row 0, 1 and 2 in main table are [0, 1, 2] (index).
    Thus, the best matching rows in Subtable 2 for the row 0, 1 and 2 in Subtable 1 are [0, 1, 2] (index).
    2. For the row 3, 4 and 5 in Subtable 1, the best matching row in Subtable 2 is 0, 1 and 2 (index).
    Likewise, as the Subtable 1 is extracted from main table in row order, the corresponding rows in main table for the row 3, 4 and 5 are 3, 4 and 5 (index).
    Thus, we can determine their patient id from main table are 1, 2 and 3, so the best matching rows in Subtable 2 for the row 3, 4 and 5 in main table are [0, 1, 2] (index).
    Thus, the best matching rows in Subtable 2 for the row 3, 4 and 5 in Subtable 1 are [0, 1, 2] (index).
    3. For the row 6, 7 and 8 in Subtable 1, the best matching row in Subtable 2 is 0, 1 and 2 (index).
    Likewise, as the Subtable 1 is extracted from main table in row order, the corresponding rows in main table for the row 6, 7 and 8 are 6, 7 and 8 (index).
    Thus, we can determine their patient id from main table are 1, 2 and 3, so the best matching rows in Subtable 2 for the row 6, 7 and 8 in main table are [0, 1, 2] (index).
    Thus, the best matching rows in Subtable 2 for the row 6, 7 and 8 in Subtable 1 are [0, 1, 2] (index).

    So, we get the best matching rows in Subtable 2 for the row 0, 1, 2, 3, 4, 5, 6, 7 and 8 in Subtable 1 are [0, 1, 2, 0, 1, 2, 0, 1, 2] (index).
```

#### extractor/agents/pk_individual/pk_ind_preprocess_step.py

##### CHECK_PATIENT_ID_SYSTEM_PROMPT

```text
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
```

##### INFER_PATIENT_ID_SYSTEM_PROMPT

```text


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
```

#### extractor/agents/pk_individual/pk_ind_summary_data_del_agent.py

##### SUMMARY_DATA_DEL_PROMPT

```text
There is now a table related to pharmacokinetics (PK). 

---

### **Task**
Your task is to delete the summary-level results from the PK table.

You can follow the following steps to complete the task:
(1) Remove any information that pertains to summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
    - Delete data entries that do not have an associated Patient ID (e.g., Patient 1, Case 1).
(2) **Do not remove** any information that pertains to specific individuals, such as individual-level results or personally identifiable data.
    - That is, if a row contains information referring to a specific individual, it must be retained — even if it's not a numeric result — because it's part of that individual's record.

---

### **Instructions**
 - When a row's classification is ambiguous, check its **adjacent rows** to infer grouping structure. 

        | Subject    | Parameter Type and Value | Parameter Type and Value |
        |------------|--------------------------|--------------------------|
 row 0: | 1          | 1                        | 1                        |
 row 1: | drug 1     | 40                       | 50                       |
 row 2: | drug 2     | 5                        | 8                        |
 row 3: | 2          | 2                        | 2                        |
 row 4: | drug 1     | 40                       | 50                       |
 row 5: | drug 2     | 5                        | 8                        |
 row 6: | 3          | 3                        | 3                        |
 row 7: | drug 1     | 40                       | 50                       |
 row 8: | drug 2     | 5                        | 8                        |
 ...

 In the above table, the row 1 and 2 are related to the subject 1, the row 3 and 4 are related to the subject 2, the row 5 and 6 are related to the subject 3, ...
 All the rows are related to individual-level results, so we should not delete them.

---

### **Input**
The input is a markdown table.
{processed_md_table}

---

### **Output**
Please return the result with the following format:
processed: boolean value, False represents the table have already meets the requirement, don't need to be processed. Otherwise, it will be True.
row_list: an array of row indices that satisfy the requirement, that is the rows have no summary-level results or personally identifiable data.
col_list: an array of column names that satisfy the requirement, that is the columns in the above rows have no summary-level results or personally identifiable data.

---

### **Example**
If the input is the above table, the output should be:
processed: False
row_list: [0, 1, 2, 3, 4, 5, 6, 7, 8]
col_list: ["Subject", "Parameter Type and Value", "Parameter Type and Value"]
```

#### extractor/agents/pk_individual/pk_ind_time_unit_agent.py

##### TIME_AND_UNIT_PROMPT

```text
The following table contains pharmacokinetics (PK) data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted some information to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_post_processed}
Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Time value, Time unit].  
- **Time Value:** A specific moment (numerical or time range) when the row of data is recorded, or a drug dose is administered.  
  - Examples: Sampling times, dosing times, or reported observation times.  
- **Time Unit:** The unit corresponding to the recorded time point (e.g., "Hour", "Min", "Day").  
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour"], ["10", "Min"], ["N/A", "N/A"]]` (example)
(3) Strictly ensure that you process only rows 0 to {md_data_post_processed_max_row_index} from the Subtable 1 (which has {md_data_lines_after_post_process_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Time value, Time unit] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Time value, Time unit] combinations are found, return the default output:**  
`[["N/A", "N/A"]]`

**Examples:**
Include:  
   - "0-12" (indicating a dosing period)  
   - "24" (indicating a time of sample collection)  
   - "5 min" (indicating a measured event)  

Do NOT include:  
   - "Tmax" (this is a pharmacokinetic parameter, NOT a recorded time)
   - "T½Beta(hr)" values (half-life parameter value, not a recorded time) 
   - "Beta(hr)" values (elimination rate constant)
```


### PK Population Summary

#### extractor/agents/pk_population_summary/pk_popu_sum_characteristic_info_agent.py

##### CHARACTERISTIC_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text] appear in the table.  
    - **Population characteristic**: Population-focused characteristics. Not PK parameter!!!
            · “Age," “Sex," "Weight," “Gender," “Race," “Ethnicity"
            · “Socioeconomic status," “Education," “Marital status"
            · “Comorbidity," “Drug indication," “Adverse events"
            · “Severity," “BMI," “Smoking status," “Alcohol use," "Blood pressure"
    - **Characteristic sub-category**: Levels or options under characteristics.
            · For sex: “Male", “Female"
            · For race: “White", “Black", “Asian", “Hispanic"
            · For comorbidity: “Diabetes", “Hypertension", “Asthma"
            · For adverse events: “Mild", “Moderate", “Severe"
            · If no sub-category, use "N/A"
    - **Characteristic values**: Include all numerical descriptors—such as means, ranges, and p-values. If multiple numerical descriptors are reported, you must include them all.
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Weight", "N/A", "76.8 (67.4-86.2)", "Pregnancy", "10", "... the sentence from the article ..."], ["Age", "N/A", "23.3 (19.02-27.58)", "Postpregnancy", "10", "... the sentence from the article ..."]] (example)  
(3) Confirm the source of each [Population characteristic, Characteristic values, Population, Population N, Source text] combination before including it in your answer.
(4) In particular, regarding Sample N, please clarify the basis for each value you selected. If there are multiple Sample N values mentioned in different parts of the text, each must be explicitly stated in the original text and should not be derived through calculation or inference. Please cite the exact sentence(s) from the paragraph that support each value.
(5) If both individual Sample N values (e.g., for specific timepoints or population subgroups) and a summed total are reported in the text, only include the individual values. Do not include the summed total, even if it is explicitly stated, to avoid duplication or overcounting.
    For example, if the text states “16 samples were collected in the first trimester, 18 in the second trimester, and a total of 34 across both," only report the 16 and 18, and exclude the total of 34.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_population_summary/pk_popu_sum_characteristic_info_refine_agent.py

##### DRUG_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population characteristic" - "Characteristic sub-category" - "Characteristic values" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Main value, Unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound]** from the table.
    - Main value: The primary value of the characteristic. If there is no variation value or interval, use the value from “Characteristic values” directly.
        For example, if the characteristic value is a ratio like "4/5/4/3", which doesn't follow a standard statistical format, simply enter "4/5/4/3" as the Main value.
    - Unit: The measurement unit of the Main value.
    - Statistics type: the statistics method to summary the Main value, like 'Mean,' 'Median,' 'Count,' etc. **This column is required and must be filled in.**
    - Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' 'Proportion (%),' etc.
    - Variation value: the value (not a range) that corresponds to the specific variation.
    - Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like 'Minmax,' 'IQR,' etc.
    - Lower bound: the lower bound value of the interval.
    - Upper bound: is the upper bound value of the interval.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["25.4", "year", "Mean", "SD", "0.5", "Minmax", "23.0", "26.1"], ...] (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(4) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_population_summary/pk_popu_sum_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population characteristic" - "Characteristic sub-category" - "Characteristic values" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Population, Pregnancy stage, Pediatric/Gestational age, Population N]** from the table.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
        
    - **Population N**: The number of people corresponding to the specific population.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Maternal", "Trimester 1", "N/A", "15"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```


### PK Population Individual

#### extractor/agents/pk_population_individual/pk_popu_ind_characteristic_info_agent.py

##### CHARACTERISTIC_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text] appear in the table.  
    - **Patient ID**: Patient ID refers to the identifier assigned to each patient.
    - **Patient characteristic**: Patient-focused characteristics. Not PK parameter!!!
            · “Age," “Sex," "Weight," “Gender," “Race," “Ethnicity"
            · “Socioeconomic status," “Education," “Marital status"
            · “Comorbidity," “Drug indication," “Adverse events"
            · “Severity," “BMI," “Smoking status," “Alcohol use," "Blood pressure"
    - **Characteristic sub-category**: Levels or options under characteristics.
            · For sex: “Male", “Female"
            · For race: “White", “Black", “Asian", “Hispanic"
            · For comorbidity: “Diabetes", “Hypertension", “Asthma"
            · For adverse events: “Mild", “Moderate", “Severe"
            · If no sub-category, use "N/A"
    - **Characteristic values**: The numerical descriptor. 
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["1", "Weight", "N/A", "76.8", "... the sentence from the article ..."], ["2", "Age", "N/A", "23", "... the sentence from the article ..."]] (example)  
(3) Confirm the source of each [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text] combination before including it in your answer.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_population_individual/pk_popu_ind_characteristic_info_refine_agent.py

##### DRUG_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Patient characteristic" - "Characteristic sub-category" - "Characteristic values" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the characteristic information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Patient ID, Main value, Unit]** from the table.
    - Patient ID
    - Main value: The primary value of the characteristic. Usually you can use the value from “Characteristic values” directly.
    - Unit: The measurement unit of the Main value.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["1", "25.4", "year"], ...] (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(4) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Patient ID for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_population_individual/pk_popu_ind_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Population characteristic" - "Characteristic sub-category" - "Characteristic values" - "Population" - "Patient ID" - "Source text" as follows:
{processed_md_table_characteristic}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]** from the table.
    - **Patient ID**: Patient ID refers to the identifier assigned to each patient.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.

(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["1", "Maternal", "Trimester 1", "N/A"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_characteristic_max_row_index} from the Subtable 1 (which has {md_table_characteristic_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Patient ID for each row in Subtable 2 must be the same as in Subtable 1.
```


### PK Drug Summary

#### extractor/agents/pk_drug_summary/pk_drug_sum_drug_info_agent.py

##### DRUG_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N] appear in the table.  
    - **Drug/Metabolite name**: The name of the drug or its metabolite that has been studied.
    - **Dose frequency**: The number of times the drug was taken. (e.g., Single, Multiple, 3, 4)
    - **Dose amount**: The amount of drug, a value, a list, or a range, (e.g., 5 mg; 1,2,3,4 g; 0.01 - 0.05 mg) each time the drug was taken.  
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Lorazepam", "2 doses", "0.01 mg", "Pregnancy", "10", "...the source text..."], ["Fentanyl", "Single dose", "0.01 mg", "Postpregnancy", "10", "...the source text..."]] (example)  
(3) Confirm the source of each [Drug/Metabolite name, Dose frequency, Dose amount, Population, Drug N] combination before including it in your answer.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_drug_summary/pk_drug_sum_drug_info_refine_agent.py

##### DRUG_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Drug/Metabolite name" - "Dose frequency" - "Dose amount" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_drug}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule, Dose route]** from the table.
    - **Drug/Metabolite name**: The name of the drug or its metabolite that has been studied.
    - **Dose amount**: The amount of drug, a value, a list, or a range, (e.g. 5; 1,2,3,4; 0.01 - 0.05) each time the drug was taken. 
    - **Dose unit**: The unit of the Dose amount, (e.g. mg) 
    - **Dose frequency**: The number of times the drug was taken. (e.g., Single, Multiple, 3, 4)
    - **Dose schedule**: The specific times or intervals at which the medication is administered, such as once a day, twice a day, or every 8 hours.
    - **Dose route**:  The route of administration of the drug, e.g., Oral, Intravenous (IV), Intramuscular (IM), Subcutaneous (SC), Epidural, Infusion, etc.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Lorazepam", "0.01", "mg", "2", "once a day", "Oral"], ...] (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(4) Strictly ensure that you process only rows 0 to {md_table_drug_max_row_index} from the Subtable 1 (which has {md_table_drug_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_drug_summary/pk_drug_sum_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Drug/Metabolite name" - "Dose frequency" - "Dose amount" - "Population" - "Population N" - "Source text" as follows:
{processed_md_table_drug}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Population, Pregnancy stage, Pediatric/Gestational age, Population N]** from the table.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
        
    - **Population N**: The number of people corresponding to the specific population.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Maternal", "Trimester 1", "N/A", "15"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_drug_max_row_index} from the Subtable 1 (which has {md_table_drug_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```


### PK Drug Individual

#### extractor/agents/pk_drug_individual/pk_drug_ind_drug_info_agent.py

##### DRUG_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Patient ID, Drug/Metabolite name, Dose frequency, Dose amount, Source text] appear in the article.  
    - **Patient ID**: Patient ID refers to the identifier assigned to each individual patient.

    - **Drug/Metabolite name**: The name of the drug or its metabolite that has been studied.
    - **Dose frequency**: The number of times the drug was taken. (e.g., Single, Multiple, 3, 4)
    - **Dose amount**: The amount of drug, a value, a list, or a range, (e.g., 5 mg; 1,2,3,4 g; 0.01 - 0.05 mg) each time the drug was taken.  

    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Lorazepam", "2 doses", "0.01 mg", "Pregnancy", "10", "...the source text..."], ["Fentanyl", "Single dose", "0.01 mg", "Postpregnancy", "10", "...the source text..."]] (example)  
(3) Confirm the source of each [Patient ID, Drug/Metabolite name, Dose frequency, Dose amount, Source text] combination before including it in your answer.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_drug_individual/pk_drug_ind_drug_info_refine_agent.py

##### DRUG_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Drug/Metabolite name" - "Dose frequency" - "Dose amount" - ""Source text" as follows:
{processed_md_table_drug}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule, Dose route]** from the table.
    - **Drug/Metabolite name**: The name of the drug or its metabolite that has been studied.
    - **Dose amount**: The amount of drug, a value, a list, or a range, (e.g. 5; 1,2,3,4; 0.01 - 0.05) each time the drug was taken. 
    - **Dose unit**: The unit of the Dose amount, (e.g. mg) 
    - **Dose frequency**: The number of times the drug was taken. (e.g., Single, Multiple, 3, 4)
    - **Dose schedule**: The specific times or intervals at which the medication is administered, such as once a day, twice a day, or every 8 hours.
    - **Dose route**:  The route of administration of the drug, e.g., Oral, Intravenous (IV), Intramuscular (IM), Subcutaneous (SC), Epidural, Infusion, etc.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Lorazepam", "0.01", "mg", "2", "once a day", "Oral"], ...] (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(4) Strictly ensure that you process only rows 0 to {md_table_drug_max_row_index} from the Subtable 1 (which has {md_table_drug_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_drug_individual/pk_drug_ind_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Drug/Metabolite name" - "Dose frequency" - "Dose amount" - ""Source text" as follows:
{processed_md_table_drug}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]** from the table.
    - **Patient ID**: Patient ID refers to the identifier assigned to each individual patient.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.

(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["1", "Maternal", "Trimester 1", "N/A"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_drug_max_row_index} from the Subtable 1 (which has {md_table_drug_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Patient ID for each row in Subtable 2 must be the same as in Subtable 1.
```


### PK Specimen Summary

#### extractor/agents/pk_specimen_summary/pk_spec_sum_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Specimen" - "Sample N" - "Sample time" - "Population" - "Population N" as follows:
{processed_md_table_specimen}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Population, Pregnancy stage, Pediatric/Gestational age, Population N]** from the table.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
        
    - **Population N**: The number of people corresponding to the specific population.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Maternal", "Trimester 1", "N/A", "15"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_specimen_max_row_index} from the Subtable 1 (which has {md_table_specimen_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Population N for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_specimen_summary/pk_spec_sum_specimen_info_agent.py

##### SPECIMEN_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Specimen, Sample N, Sample time, Population, Population N] appear in the table.  
    - **Specimen**: The type of biological sample collected (e.g., urine, blood).
    - **Sample N**: The number of samples analyzed for the corresponding specimen.
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled.   
    - **Population**: The group of individuals the samples were collected from (e.g., healthy adults, pregnant women).
    - **Population N**: The number of individuals in that population group.
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["Urine", "20", "... the sentence from the article ...", "Pregnancy", "10"], ["Urine", "20", "... the sentence from the article ...", "Postpregnancy", "10"]] (example)  
(3) Confirm the source of each [Specimen, Sample N, Sample time, Population, Population N] combination before including it in your answer.
(4) In particular, regarding Sample N, please clarify the basis for each value you selected. If there are multiple Sample N values mentioned in different parts of the text, each must be explicitly stated in the original text and should not be derived through calculation or inference. Please cite the exact sentence(s) from the paragraph that support each value.
(5) If both individual Sample N values (e.g., for specific timepoints or population subgroups) and a summed total are reported in the text, only include the individual values. Do not include the summed total, even if it is explicitly stated, to avoid duplication or overcounting.
    For example, if the text states “16 samples were collected in the first trimester, 18 in the second trimester, and a total of 34 across both,” only report the 16 and 18, and exclude the total of 34.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_specimen_summary/pk_spec_sum_time_unit_agent.py

##### TIME_AND_UNIT_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Specimen" - "Sample N" - "Sample time" - "Population" - "Population N" as follows:
{processed_md_table_specimen}

Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Sample time, Time unit, Source text].  
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled. **Keep the value(s) numerical! Keep the value(s) numerical! Keep the value(s) numerical!**  
    e.g., "0", "24", "0, 2, 4", "0-2", "0-2, 2-4, 4-6"
    - **Time Unit:** The unit corresponding to the sample time (e.g., "Second", "Minute", "Hour", "Day").  
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour", "... the sentence from the article ..."], ["10", "Minute", "... the sentence from the article ..."], ["0, 2, 4, 6", "Hour", "... the sentence from the article ..."], ["0-2, 2-4, 4-6, 6-8", "Hour", "... the sentence from the article ..."], ["N/A", "N/A", "... the sentence from the article ..."]]` (example)
    - If a field contains multiple values separated by commas, preserve the comma-separated string as a single element.  
    - If a range of time is given (like "0-1"), treat the entire range string as one item.  
    - If no valid data is available, use "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_specimen_max_row_index} from the Subtable 1 (which has {md_table_specimen_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Sample time, Time unit, Source text] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Sample time, Time unit, Source text] combinations are found, return the default output:**  
`[["N/A", "N/A", "N/A"]]`
```


### PK Specimen Individual

#### extractor/agents/pk_specimen_individual/pk_spec_ind_patient_info_refine_agent.py

##### PATIENT_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Specimen" - "Sample N" - "Sample time" as follows:
{processed_md_table_specimen}

Carefully review the article and follow these steps to convert the population information in Subtable 1 into a more detailed format in Subtable 2.

(1) Identify all unique combinations of **[Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]** from the table.
    - **Patient ID**: Patient ID refers to the identifier assigned to each individual patient.
    - **Population**: The age group of the subjects.  
      **Common categories include:**  
        - "Nonpregnant"
        - "Maternal" (pregnant individuals)
        - "Pediatric" (generally birth to ~17 years)  
        - "Adults" (typically 18 years or older)  
      
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
      **Common categories include:**  
        - "Pre-pregnancy"
        - "Trimester 1" (usually up to 14 weeks of pregnancy)  
        - "Trimester 2" (~15–28 weeks of pregnancy)  
        - "Trimester 3" (~≥ 28 weeks of pregnancy)  
        - "Fetus" (referring to the developing baby during pregnancy)  
        - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
        - "Postpartum" (~6–8 weeks after birth)  
        - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
 
    - **Pediatric/Gestational age**: The child's age (or age range) at a specific point in the study. Retain the original wording whenever possible. It can also be the pregnancy weeks.
        Note: Verify that the value explicitly states the age. Only consider it valid if the age is directly mentioned. Do not infer age from the timing of data recording or drug administration.
        For example: "Concentrations on Days 7" refers to a measurement time point, not an age, and should not be treated as such.
    
(2) Compile each unique combination in the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["1", "Maternal", "Trimester 1", "N/A"], ...] (example)

(3) For each Population, determine whether it can be classified under one or more of the common categories listed above. If it matches one or more standard categories, replace it with the corresponding standard category (or categories). If it does not fit any common category, retain the original wording.

(4) For each Pregnancy Stage, check whether it aligns with any of the common categories. If it does, replace it with the corresponding standard category. If it does not fit any common category, keep the original wording unchanged.

(5) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
   
(6) Strictly ensure that you process only rows 0 to {md_table_specimen_max_row_index} from the Subtable 1 (which has {md_table_specimen_row_num} rows in total).   
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - **The output must maintain the original row order** from Subtable 1—do not shuffle, reorder, or omit any rows. The Patient ID for each row in Subtable 2 must be the same as in Subtable 1.
```

#### extractor/agents/pk_specimen_individual/pk_spec_ind_specimen_info_agent.py

##### SPECIMEN_INFO_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Determine how many unique combinations of [Patient ID, Specimen, Sample N, Sample time] appear in the table.  
    - **Patient ID**: Patient ID refers to the identifier assigned to each individual patient.
    - **Specimen**: The type of biological sample collected (e.g., urine, blood).
    - **Sample N**: The number of samples analyzed for the corresponding specimen.
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled.   
(2) List each unique combination in Python list-of-lists syntax, like this:  
    [["1", "Urine", "20", "... the sentence from the article ..."], ["2", "Urine", "20", "... the sentence from the article ..."]] (example)  
(3) Confirm the source of each [Patient ID, Specimen, Sample N, Sample time] combination before including it in your answer.
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pk_specimen_individual/pk_spec_ind_time_unit_agent.py

##### TIME_AND_UNIT_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Specimen" - "Sample N" - "Sample time" as follows:
{processed_md_table_specimen}

Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Sample time, Time unit, Source text].  
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled. **Keep the value(s) numerical! Keep the value(s) numerical! Keep the value(s) numerical!**  
    e.g., "0", "24", "0, 2, 4", "0-2", "0-2, 2-4, 4-6"
    - **Time Unit:** The unit corresponding to the sample time (e.g., "Second", "Minute", "Hour", "Day").  
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour", "... the sentence from the article ..."], ["10", "Minute", "... the sentence from the article ..."], ["0, 2, 4, 6", "Hour", "... the sentence from the article ..."], ["0-2, 2-4, 4-6, 6-8", "Hour", "... the sentence from the article ..."], ["N/A", "N/A", "... the sentence from the article ..."]]` (example)
    - If a field contains multiple values separated by commas, preserve the comma-separated string as a single element.  
    - If a range of time is given (like "0-1"), treat the entire range string as one item.  
    - If no valid data is available, use "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_specimen_max_row_index} from the Subtable 1 (which has {md_table_specimen_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Sample time, Time unit, Source text] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Sample time, Time unit, Source text] combinations are found, return the default output:**  
`[["N/A", "N/A", "N/A"]]`
```


### PE Study Outcome (v2)

#### extractor/agents/pe_study_outcome_ver2/pe_study_out_param_value_agent.py

##### PARAMETER_VALUE_PROMPT

```text
The following main table contains Pharmacoepidemiology data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted a few numerical values to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_with_1_value}
Please review the information in Subtable 1 row by row and complete Subtable 2 accordingly.
Specifically, you need to interpret the meaning of each entry in the "Value" column of Subtable 1 and rewrite it in a more structured and standardized format in Subtable 2.
Subtable 2 should include the following column headers only:
**Main value, Main value unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound, P value**

Main value: the value of main parameter (not a range). 
Main value unit: The unit of the main parameter (e.g. kg, g, Count) **DO NOT USE Statistics type, such as SD, as the unit!!**
Statistics type: The statistical method used to summarize the Main value, such as Mean, Median, Sum, Proportion, or %, etc. This column is required and must be completed.
Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' etc.
Variation value: the value (not a range) that corresponds to the specific variation.
    **Please note:** In addition to common cases like standard deviations (SD), there is a special case that should also be handled using the Variation type and Variation value columns:
    Often, datasets report both count and percentage values together. In such cases, enter the count into Main value, set Main value unit to "Count", and choose "Sum" for Statistics type. 
    Then, record the percentage or proportion under Variation type and Variation value, respectively.

Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI,' 'Range,' 'IQR,' etc.
Lower bound: the lower bound value of the interval.
Upper bound: is the upper bound value of the interval.
P value: Its P-value.

Please Note:
(1) An interval consisting of two numbers must be placed separately into the Low limit and High limit fields; it is prohibited to place it in the Variation value field.
(9) Important: Every row in Subtable 2 must contain exactly {COLUMN_NUMBER} values.
    - Even if you don’t know the value for some columns, you must still fill them with "N/A".
    - Rows with fewer than {COLUMN_NUMBER} values will be considered invalid.
(3) Strictly ensure that you process only rows 0 to {md_table_with_1_value_max_row_index} from the Subtable 1 (which has {md_table_with_1_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) For rows in Subtable 1 that can not be extracted, enter "N/A" for the entire row.
(5) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. Ensure all values are converted to strings.
(6) **Absolutely no calculations are allowed—every value must be taken directly from Subtable 1 without any modifications.**  
(7) **P value is very important:** The P-value usually won't be in the row you're processing. You must check the **entire main table** to see if there's a corresponding P-value, and fill it in if found.  
    - **If the same P-value corresponds to multiple rows in Subtable 1, you must fill in that P-value for all of those rows in Subtable 2.**  
    - Do not skip or omit the P-value in any row that should have it.
(8) The final list should be like this:
[["10", "Count", "Sum", "%", "1", "N/A", "N/A", "N/A", "N/A"], ["50", "N/A", "%", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]]
```

#### extractor/agents/pe_study_outcome_ver2/pe_study_out_study_info_agent.py

##### PARAMETER_VALUE_PROMPT

```text
The following main table contains Pharmacoepidemiology data:  
{processed_md_table}
Here is the table caption:  
{caption}
From the main table above, I have extracted a few numerical values to create Subtable 1:  
Below is Subtable 1:
{processed_md_table_with_1_value}
Please review the information in Subtable 1 row by row and complete Subtable 2 accordingly.
Specifically, you need to locate each row from Subtable 1 in the main table, understand its context and meaning, and then use this understanding to populate Subtable 2.
Pay special attention to how the values in the main table relate to both the row and column headers — this will often determine what should be classified as Characteristic, Exposure, or Outcome.

Subtable 2 should include the following column headers only:
**Characteristic, Exposure, Outcome**

    - **Characteristic**: Any geographic, demographic or biological feature of the subjects in the study (e.g., age, sex, race, weight, genetic markers).
    - **Exposure**: Any factor that might be associated with an outcome of interest (e.g., drugs, medical conditions, medications, etc.). If the column header name includes a drug name, it is very likely to be classified as "Exposure".
    - **Outcome**: A measure or set of measures that reflect the effects or consequences of the exposure (e.g., drug treatment, condition, or intervention). These are typically endpoints used to assess the impact of the exposure on the subjects, such as birth weight, symptom reduction, or lab results. In the context of the study outcomes sheet, this category should include all such relevant measures and their associated statistics.    
    
Please Note:
(1) Important: Every row in Subtable 2 must contain exactly {COLUMN_NUMBER} values.
    - Even if you don’t know the value for some columns, you must still fill them with "N/A".
    - Rows with fewer than {COLUMN_NUMBER} values will be considered invalid.
(2) Strictly ensure that you process only rows 0 to {md_table_with_1_value_max_row_index} from the Subtable 1 (which has {md_table_with_1_value_rows} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(3) Carefully review both the row and column headers in the main table, as they are often directly relevant to completing the Characteristic, Exposure, and Outcome columns — or they may provide important context or hints.
(4) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. 
(5) Note: The Outcome column is intended to describe what the main value represents — that is, it conveys the meaning or purpose of the value extracted from the main table. It should not contain the numerical value itself. Only the contextual outcome that the value measures should be included.
(6) The final list should be like this:
[["infants of substance abuse mothers", "cocaine unexposed", "total sleep time"], ["infants of substance abuse mothers", "cocaine exposed", "total sleep time"]]
```


### PE Study Information

#### extractor/agents/pe_study_info/pe_study_info_design_info_agent.py

##### STUDY_DESIGN_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

(1) Summarize the Article in the format of [[Study type, Study design, Data source]]. 
    - **Study type**: This deals with the area of interest related to a particular article; 
        choose from Pharmacoepidemiology / Clinical Trials / Pharmacokinetics / Pharmacodynamics / Pharmacogenetics. 
    - **Study design**: Identify the study design described in the article. Common examples include:
        - *Prospective cohort study*
        - *Retrospective cohort study*
        - *Randomized controlled trial (RCT)*
        - *Double-blind randomized trial*
        - *Case-control study*
        - *Cross-sectional study*
        - *Systematic review and meta-analysis*
        - *Open-label study*
        - *Nested case-control study*
        - *Pilot study*
        - *Chart review*
        - *Observational study*  
        If multiple designs are mentioned (e.g., "prospective, randomized, double-blind"), list them as one string in the same order.
    - **Data source**: The primary locations(s) where data is accessed from or primary site where a study was conducted (ex: hospitals, database, geographic locations etc.)
(2) List the combination in Python list-of-lists syntax, like this:  
    [["Pharmacoepidemiology", "Prospective Randomized Double-blinded Invesigation", "OSUMC"]] (example)
```

##### INSTRUCTION_PROMPT

```text
Do not give the final result immediately. First, explain your thought process, then provide the answer.
```

#### extractor/agents/pe_study_info/pe_study_info_design_info_refine_agent.py

##### DESIGN_INFO_REFINE_PROMPT

```text
{title}
{full_text}
Read the article and answer the following:

Based on the article, Subtable 1 has already been created by extracting and combining information on "Study type" – "Study design" – "Data source" into a one-row format.
{processed_md_table_design}
Now, please carefully review the article and extract the Study-designn-related information to create Subtable 2, also in a one-row format. Subtable 2 should complement and enhance the information in Subtable 1.

(1) Identify all unique combinations of **[Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N, Drug name, Outcomes]** from the table.
    - **Population**: The age group of the subjects.  
        **Common categories include:**  
            - "Nonpregnant"
            - "Maternal" (pregnant individuals)
            - "Pediatric" (generally birth to ~17 years)  
            - "Adults" (typically 18 years or older)  
    - **Inclusion criteria**: Characteristics that study participants must have to be qualified to be part of a study. (use exact wording from the article)
    - **Exclusion criteria**: Characteristics that disqualify interested participants from participating in a study. (use exact wording from the article)
    - **Pregnancy stage**: The stage of pregnancy for the patients in the study.  
        **Common categories include:**  
            - "Trimester 1" (usually up to 14 weeks of pregnancy)  
            - "Trimester 2" (~15–28 weeks of pregnancy)  
            - "Trimester 3" (~≥ 28 weeks of pregnancy)  
            - "Fetus" or "Fetal Stage" (referring to the developing baby during pregnancy)  
            - "Parturition," "Labor," or "Delivery" (the process of childbirth)  
            - "Postpartum" (~6–8 weeks after birth)  
            - "Nursing," "Breastfeeding," or "Lactation" (refers to the period of breastfeeding after birth) 
    - **Subject N**: The number of subjects corresponding to the specific population.
    - **Drug name**: All drugs of interest as they relate to the outcomes of a particular study.
    - **Outcomes**: A measure(s) of interest that an investigator(s) considers the most important among the many outcomes to be examined in the study. (use exact wording from the article)

(2) Write the one-row subtable 2 into the format of a **list of lists**, using **Python string syntax**. The result should be like this:
[["Maternal", "Inclusion criteria should be retrieved from the article, use the exact wording from the article", "Exclusion criteria should be retrieved from the article, use the exact wording from the article", "Trimester 3", "20", "Lorazepam", "Outcomes should be retrieved from the article, use the exact wording from the article"]] (example)

(3) Use **"N/A"** as the placeholder if the information **cannot** be reasonably inferred.
```


## Step 4 — PKPE Curated Tables Verification

### extractor/agents/pk_pe_agents/pk_pe_verification_step.py

#### PKPE_VERIFICATION_SYSTEM_PROMPT

```text
You are a biomedical data verification assistant with expertise in {domain} and data accuracy validation. 
Your task is to carefully examine the **source paper title and tables**, and determine whether the **curated {domain} data table** is an accurate and faithful representation of the information provided in the source.

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

You must respond using the **exact json compact format** below:

```
{{
  "reasoning_process": <string, a concise explanation of the thought process or reasoning steps taken to reach a conclusion (no more than 200 words)>,
  "correct": <boolean, True / False>,
  "explanation": <string, brief explanation of whether the curated table is accurate. If incorrect, explain what is wrong, including specific mismatched values or structure issues>,
  "suggested_fix": <string or None, if incorrect, provide a corrected version of the curated table or the corrected values/rows/columns.>
}}
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

{paper_title}

#### **Paper Abstract**

{paper_abstract}

#### **Source Table(s) or full text**

{source_tables}

#### **Curated Table**

{curated_table}

---
```

## Step 5 — PKPE Curated Tables Correction Code

### extractor/agents/pk_pe_agents/pk_pe_correction_code_step.py

#### PKPE_CORRECTION_SYSTEM_PROMPT

```text
You are a biomedical data correction engineer with expertise in {domain} and robust Python data wrangling.

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

FORBIDDEN (must not appear anywhere in your output code):
- Any function definition or assignment for markdown_to_dataframe, including:
  - "def markdown_to_dataframe"
  - "markdown_to_dataframe ="
  - any custom markdown parsing implementation intended to replace it

If you violate the FORBIDDEN rule, your output is considered invalid.

Your task:
Write Python code ONLY that corrects the curated table according to the Reasoning Process and the source table(s).

Hard requirements:
- Your output must be valid JSON (compact) matching EXACTLY this schema:
  {{"code": "<python code as a single string WITHOUT code fences>"}}
- The code inside "code" must be runnable as-is (no placeholders).
- Do NOT include any explanatory text outside the JSON object.
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

Now produce the JSON object with the "code" field only.

--------------------
Paper Title:
{paper_title}

Paper Abstract:
{paper_abstract}

Source Table(s) or full text:
{source_tables}

Curated Table:
(curated markdown table string will be assigned to curated_md at runtime)
{curated_table}

Reasoning Process:
{reasoning_process}
--------------------
```
