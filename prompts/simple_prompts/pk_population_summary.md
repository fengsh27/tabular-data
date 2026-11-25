# pk-population-summary — simple single-shot prompt

One-shot prompt for study-level population characteristics (the
`pk_population_summary` pipeline). Fast path of the cascade; escalates to
`PKPopuSumWorkflow` when the check fails.

Placeholder: `{TABLE}` holds the source the production tool would use: a
selected demographic table (markdown, with its caption and footnote) when the
paper has one, otherwise the paper's prepared full text. `{PMID}` is
substituted if present.

The 15 columns match `PKPopuSumWorkflow.go_full_text`'s output after its final
rename (`Population characteristic` -> `Characteristic`, `Characteristic
sub-category` -> `Characteristic subcategory`, `Unit` -> `Characteristic unit`,
`Main value` -> `Characteristic value`, `Population N` -> `Subject N`,
`Source text` -> `Note`) and `scripts/simple_prompt/schema_specs.py`.

---

Extract the characteristics of the study population (demographic, anthropometric and clinical descriptors with their summary statistics) from the source below, which is either a table of the paper or the paper's text.

Output CSV with exactly these 15 columns, in this order:

Characteristic,Characteristic subcategory,Characteristic unit,Characteristic value,Statistics type,Variation type,Variation value,Interval type,Lower bound,Upper bound,Population,Pregnancy stage,Pediatric/Gestational age,Subject N,Note

Write one row per reported statistic per population group: a characteristic reported for two groups is two rows; a characteristic reported as both a mean and a median is two rows; each level of a categorical characteristic (Male / Female; White / Black / Asian) is its own row.

Column rules:

- Characteristic: the population descriptor: Age, Sex, Gender, Weight, Height, BMI, Race, Ethnicity, Gestational age, Birth weight, Apgar score, Parity, Smoking status, Alcohol use, Comorbidity, Drug indication, Severity, Socioeconomic status, Education, Marital status, Blood pressure, ... Required on every row. Pharmacokinetic parameters and drug concentrations (AUC, Cmax, clearance, half-life, plasma level) are NOT characteristics: never a row here.
- Characteristic subcategory: the level of a categorical characteristic (Sex: Male / Female; Race: White / Black / Asian / Hispanic; Comorbidity: Diabetes / Hypertension; Severity: Mild / Moderate / Severe). Empty for a continuous characteristic.
- Characteristic unit: the unit of Characteristic value (year, kg, cm, kg/m2, weeks, %, Count). Empty when there is none.
- Characteristic value: the main number: the mean, median or count. When the source gives a ratio or pattern that is not a standard statistic ("4/5/4/3"), write it as is.
- Statistics type: how the main value summarises the group: Mean, Median, Count, Proportion, Percentage, Range, ... Required whenever Characteristic value is filled.
- Variation type: the spread measure attached to the main value: SD, SEM, CV%, IQR, or % when a count is accompanied by its percentage. Empty when none.
- Variation value: the number of that spread measure (one number, never a range).
- Interval type: the kind of interval reported with the value: Range, Minmax, IQR, 95% CI, ... Empty when none.
- Lower bound / Upper bound: the two ends of that interval, as plain numbers. A range "23.0-26.1" goes here, never into Variation value.
- Population: the group: Nonpregnant, Maternal (pregnant individuals), Pediatric (birth to ~17 years) or Adults (18 years or older) when one fits; otherwise the source's wording ("neonates", "controls").
- Pregnancy stage: Pre-pregnancy; Trimester 1 (up to 14 weeks); Trimester 2 (~15-28 weeks); Trimester 3 (>= 28 weeks); Fetus; Parturition, Labor or Delivery; Postpartum; Nursing, Breastfeeding or Lactation. Keep the source's wording when none fits; empty when none is given.
- Pediatric/Gestational age: the age or gestational age of the group only when explicitly stated as such ("28-32 weeks", "term infants"). Never inferred from a measurement time. Otherwise empty.
- Subject N: the number of individuals in that group, as a plain number. Empty when not stated.
- Note: a short verbatim excerpt of the sentence or table cell reporting this value (quote the cell). Empty if none can be cited.

General rules:

- Take every value from the source; never calculate, convert or invent. "25.4 ± 0.5" is value 25.4 with Variation type SD (or SEM when the source says so) and Variation value 0.5. "12 (60%)" is value 12, unit Count, Statistics type Count, Variation type %, Variation value 60. "30 (26-34)" is value 30 with Interval type Range, bounds 26 and 34.
- Leave a cell empty when the source does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the source reports no population characteristics, output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Source:

{TABLE}
