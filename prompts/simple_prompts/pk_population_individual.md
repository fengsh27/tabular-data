# pk-population-individual — simple single-shot prompt

One-shot prompt for per-individual population characteristics (the
`pk_population_individual` pipeline). Fast path of the cascade; escalates to
`PKPopuIndWorkflow` when the check fails.

Placeholder: `{TABLE}` holds a selected demographic table (markdown with
caption and footnote) when the paper has one, otherwise the paper's prepared
full text. `{PMID}` is substituted if present.

The 9 columns match `PKPopuIndWorkflow.go_full_text`'s output after its final
rename (`Patient characteristic` -> `Characteristic`, `Characteristic
sub-category` -> `Characteristic subcategory`, `Unit` -> `Characteristic unit`,
`Main value` -> `Characteristic value`, `Source text` -> `Note`) and
`scripts/simple_prompt/schema_specs.py`.

---

Extract the characteristics (demographic, anthropometric and clinical descriptors) of every identified individual (patient, case, subject, mother, infant) from the source below, which is either a table of the paper or the paper's text.

Output CSV with exactly these 9 columns, in this order:

Patient ID,Characteristic,Characteristic subcategory,Characteristic unit,Characteristic value,Population,Pregnancy stage,Pediatric/Gestational age,Note

Write one row per individual per characteristic (one individual with age, weight and sex is three rows). Characteristics reported only as group statistics (mean, median, count) with no individual identifier are not rows in this table.

Column rules:

- Patient ID: the identifier the source uses for the individual: a number ("1", "2"), "Case 3", "Patient A", "Mother", "Infant", "Twin 1". Required on every row. Keep the source's own labels; do not renumber.
- Characteristic: the descriptor: Age, Sex, Gender, Weight, Height, BMI, Race, Ethnicity, Gestational age, Birth weight, Apgar score, Parity, Smoking status, Comorbidity, Drug indication, Severity, ... Required. Pharmacokinetic parameters and drug concentrations are NOT characteristics: never a row here.
- Characteristic subcategory: the level of a categorical characteristic (Sex: Male / Female; Race: White / Black; Severity: Mild / Severe). Empty for a numeric characteristic.
- Characteristic unit: the unit of Characteristic value (year, kg, cm, weeks, g). Empty when there is none.
- Characteristic value: the individual's value as written (25, 3.2, "38+2"). For a categorical characteristic whose level is in the subcategory column, leave this empty unless the source gives a value.
- Population: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; otherwise the source's wording.
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum; Nursing, Breastfeeding or Lactation. Keep the source's wording when none fits; empty when none is given.
- Pediatric/Gestational age: the individual's age or gestational age only when explicitly stated as such. Never inferred from a measurement time. Otherwise empty.
- Note: a short verbatim excerpt of the sentence or table cell reporting this value (quote the cell). Empty if none can be cited.

General rules:

- Take every value from the source; never calculate, convert or invent.
- Leave a cell empty when the source does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the source reports no per-individual characteristics, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Source:

{TABLE}
