# Stage 01 — Drug / dosing information per patient (from full text)

This file is loaded by `procedure.md` as the first stage of PK drug individual
curation.

## What you are doing
Read the paper's full text and extract every **unique** per-patient drug-dosing
combination it describes, as rows of
`[Patient ID, Drug/Metabolite name, Dose frequency, Dose amount, Source text]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
Read the article and identify each unique combination of the fields below. A new
combination of *any* field is a new row.

- **Patient ID** — the identifier of the individual patient/case (e.g. `1`,
  `Case 2`). Much individual data is in prose: a single-patient case report uses
  `1` for every row; a multi-case paper uses the text's own case numbers.
- **Drug/Metabolite name** — the drug or its metabolite studied.
- **Dose frequency** — how many times the drug was taken (e.g. `Single`,
  `Multiple`, `3`, `4`).
- **Dose amount** — the amount each time: a value, list, or range
  (e.g. `5 mg`; `1,2,3,4 g`; `0.01 - 0.05 mg`). Keep it as written here (stage 3
  splits the unit off).
- **Source text** — the original sentence/excerpt where this was reported. Use
  `"N/A"` if no source sentence can be identified.

Confirm each combination against the text before including it. Do not invent
dosing the article does not state.

## Reasoning then answer
First explain your thought process — how many patients/cases the article
describes, which sentences give each one's dosing — then produce the table.

## Output of this stage
A markdown table with columns exactly
`Patient ID`, `Drug/Metabolite name`, `Dose frequency`, `Dose amount`,
`Source text`:

```
| Patient ID | Drug/Metabolite name | Dose frequency | Dose amount | Source text |
| --- | --- | --- | --- | --- |
| 1 | Lorazepam | Multiple | 0.01 mg | ...the source sentence... |
```

If the article describes no dosing at all, emit a single row of `N/A` across all
five columns (see procedure.md's error-handling rule).

Before continuing, sanity-check:
- columns are exactly the five above, in order,
- one row per **unique** patient-dose combination,
- `Patient ID` is consistent with the case structure you read from the text,
- `Source text` quotes the article (or is `"N/A"`).

If a check fails, redo this stage once. Then **write the result to
`01_drug_info.md`** in the scratch directory. Stages 2, 3, and 4 read it.
