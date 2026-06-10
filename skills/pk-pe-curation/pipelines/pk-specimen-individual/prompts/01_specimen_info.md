# Stage 01 — Specimen / sampling information per patient (from full text)

This file is loaded by `procedure.md` as the first stage of PK specimen individual
curation.

## What you are doing
Read the paper's full text and extract every **unique** per-patient
specimen-sampling combination it describes, as rows of
`[Patient ID, Specimen, Sample N, Sample time]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
Identify each unique combination of the fields below. A new combination of *any*
field is a new row.

- **Patient ID** — the identifier of the individual patient/case (e.g. `1`,
  `Case 2`). A single-patient case report uses `1` for every row; a multi-case
  paper uses the text's own case numbers.
- **Specimen** — the type of biological sample collected (e.g. `urine`, `blood`,
  `plasma`, `cord blood`, `milk`).
- **Sample N** — the number of samples analyzed for that specimen. It must be
  **explicitly stated** in the text, never derived by calculation. Cite the
  supporting sentence.
- **Sample time** — the specific moment (numeric value or time range) when the
  specimen is sampled.

Confirm each combination against the text before including it. Do not invent or
calculate.

## Reasoning then answer
First explain your thought process — how many patients/cases the article
describes, which sentences give each one's specimen sampling — then produce the
table.

## Output of this stage
A markdown table with columns exactly
`Patient ID`, `Specimen`, `Sample N`, `Sample time`:

```
| Patient ID | Specimen | Sample N | Sample time |
| --- | --- | --- | --- |
| 1 | Urine | 20 | 0-2 h |
```

If the article describes no specimen sampling at all, emit a single row of `N/A`
across all four columns (see procedure.md's error-handling rule).

Before continuing, sanity-check:
- columns are exactly the four above, in order,
- one row per **unique** patient-specimen combination,
- `Patient ID` is consistent with the case structure you read from the text.

If a check fails, redo this stage once. Then **write the result to
`01_specimen_info.md`** in the scratch directory. Stages 2, 3, and 4 read it.
