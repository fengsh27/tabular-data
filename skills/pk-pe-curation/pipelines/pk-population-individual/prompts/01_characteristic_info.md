# Stage 01 — Patient characteristic information (from full text)

This file is loaded by `procedure.md` as the first stage of PK population individual
curation.

## What you are doing
Read the paper's full text and extract every **unique** per-patient
characteristic combination it describes, as rows of
`[Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
- **Patient ID** — the identifier of the individual patient/case (e.g. `1`,
  `Case 2`). A single-patient case report uses `1`; a multi-case paper uses the
  text's own case numbers.
- **Patient characteristic** — a patient-focused characteristic, **not a PK
  parameter**. Examples:
  - `Age`, `Sex`, `Weight`, `Gender`, `Race`, `Ethnicity`
  - `Socioeconomic status`, `Education`, `Marital status`
  - `Comorbidity`, `Drug indication`, `Adverse events`
  - `Severity`, `BMI`, `Smoking status`, `Alcohol use`, `Blood pressure`
- **Characteristic sub-category** — a level/option under it (Sex → `Male`/
  `Female`; Race → `White`/`Black`/`Asian`/`Hispanic`; Comorbidity → `Diabetes`/
  `Hypertension`/`Asthma`; Adverse events → `Mild`/`Moderate`/`Severe`). If none,
  use `"N/A"`.
- **Characteristic values** — the numerical descriptor (the patient's raw value).
- **Source text** — the original sentence/excerpt. Use `"N/A"` if none.

Confirm each combination against the text. Do not invent or calculate.

## Reasoning then answer
Explain how many patients/cases the article describes and which sentences give
each one's characteristics, then produce the table.

## Output of this stage
A markdown table with columns exactly
`Patient ID`, `Patient characteristic`, `Characteristic sub-category`,
`Characteristic values`, `Source text`:

```
| Patient ID | Patient characteristic | Characteristic sub-category | Characteristic values | Source text |
| --- | --- | --- | --- | --- |
| 1 | Weight | N/A | 76.8 | ...the sentence... |
| 2 | Age | N/A | 23 | ...the sentence... |
```

If the article describes no characteristics, emit a single all-`N/A` row across
the five columns.

Before continuing, sanity-check:
- columns are exactly the five above, in order,
- `Patient characteristic` is a characteristic, **not** a PK parameter,
- `Patient ID` is consistent with the case structure you read from the text.

If a check fails, redo this stage once. Then **write the result to
`01_characteristic_info.md`**. Stages 2, 3, and 4 read it.
