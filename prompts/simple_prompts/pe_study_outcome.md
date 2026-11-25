# pe-study-outcome — simple single-shot prompt

One-shot table-to-CSV prompt for pharmacoepidemiology outcome tables (the
`pe_study_outcome` pipeline, ver2 workflow). Fast path of the cascade;
escalates to `PEStudyOutWorkflow` when the check fails.

Placeholder: `{TABLE}` holds one selected PE table as markdown with its
caption and footnote. `{PMID}` is substituted if present.

The 12 columns match `PEStudyOutWorkflow.go_md_table`'s output after its row
cleanup (`Main value` -> `Parameter value`, `Statistics type` -> `Parameter
statistic`, `Main value unit` -> `Parameter unit`) and
`scripts/simple_prompt/schema_specs.py`. The benchmark's gold spells the third
column `Outcomes`; its preprocessing maps `Outcome` onto it.

---

Extract every numeric outcome result from the pharmacoepidemiology table below.

Output CSV with exactly these 12 columns, in this order:

Characteristic,Exposure,Outcome,Parameter unit,Parameter statistic,Parameter value,Variation type,Variation value,Interval type,Lower bound,Upper bound,P value

Write one row per numeric result cell of the table: every cell that holds a number (a count, a percentage, a mean, a rate, an odds or risk ratio, ...) becomes one row, in table order (row by row, left to right). A cell such as "12 (60%)" or "3.1 ± 0.4" or "1.8 (1.2-2.7)" is one row that carries its main value and the accompanying percentage, variation or interval in the dedicated columns. Do not write rows for header cells, footnotes or cells with no number.

Column rules:

- Characteristic: the demographic, geographic or biological feature of the subjects the result refers to (age group, sex, race, weight class, genotype), read from the row or column headers. Empty when the result is not stratified by a characteristic.
- Exposure: the factor the result is associated with: the drug, treatment arm, dose group, condition or exposure status (e.g. "cocaine exposed", "HAART", "control"). When a column header names a drug, it is the exposure. Empty when none applies.
- Outcome: what the number measures: the endpoint or outcome named by the row or column header (birth weight, preterm birth, neutropenia grade 3-4, total sleep time, ...). Required on every row. This is the name of the measure, never the number itself.
- Parameter unit: the unit of Parameter value (kg, g, %, Count, mmol/L, weeks, days). For a count write Count. Never put a statistic name (SD, Mean) here.
- Parameter statistic: how Parameter value summarises the data: Mean, Median, Sum (for a count), Proportion, Percentage, Rate, Odds ratio, Relative risk, Hazard ratio, ... Required whenever Parameter value is filled.
- Parameter value: the main number of the cell (never a range). For "12 (60%)" it is 12 with Parameter unit Count and Parameter statistic Sum.
- Variation type: the spread measure given with the value: SD, SEM, CV%, IQR, or % when a count is accompanied by its percentage. Empty when none.
- Variation value: the number of that spread measure (one number, never a range). For "12 (60%)" it is 60 with Variation type %.
- Interval type: the kind of interval reported with the value: 95% CI, Range, IQR, ... Empty when none.
- Lower bound / Upper bound: the two ends of the interval, as plain numbers. An interval always goes here, never into Variation value.
- P value: the p value that applies to this result, as written ("<0.001", "0.04"). Look across the whole table (a p value column, a footnote marker) and repeat the same p value on every row it applies to. Empty when none.

General rules:

- Take every value from the table; never calculate, convert or invent. A percentage column written as "60" is Parameter value 60 with Parameter unit % and Parameter statistic Percentage.
- Leave a cell empty when the table does not state it. Do not write N/A, NR or a dash.
- Quote any cell that contains a comma.
- If the table holds no numeric outcome results (e.g. only definitions, only baseline characteristics with no outcome), output only the header line.
- Output the CSV only: the header line and the data rows, no explanation, no markdown fence.

Table:

{TABLE}
