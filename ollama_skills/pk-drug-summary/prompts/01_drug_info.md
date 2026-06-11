# Stage 01 — Drug / dosing information (from full text)

This file is loaded by `procedure.md` as the first stage of PK drug summary curation.

## What you are doing
Read the paper's full text and extract every **unique** drug-dosing combination
it describes, as rows of
`[Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N, Source text]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
Read the article and identify each unique combination of the fields below. A new
combination of *any* field is a new row.

- **Drug/Metabolite name** — the drug or its metabolite studied.
- **Dose frequency** — how many times the drug was taken (e.g. `Single`,
  `Multiple`, `3`, `4`).
- **Dose amount** — the amount each time: a value, a list, or a range
  (e.g. `5 mg`; `1,2,3,4 g`; `0.01 - 0.05 mg`). Keep it as written here (stage 3
  splits the unit off).
- **Population** — the group the data was collected from (e.g. `healthy adults`,
  `pregnant women`).
- **Population N** — the number of individuals in that population group.
- **Source text** — the original sentence/excerpt where this was reported, for
  traceability. Use `"N/A"` if no source sentence can be identified.

Confirm each combination against the text before including it. Do not invent
dosing the article does not state.

## Reasoning then answer
First explain your thought process — which sentences describe dosing, how you
grouped them into unique combinations — then produce the table.

## Output of this stage
A markdown table with columns exactly
`Drug/Metabolite name`, `Dose frequency`, `Dose amount`, `Population`,
`Population N`, `Source text`:

```
| Drug/Metabolite name | Dose frequency | Dose amount | Population | Population N | Source text |
| --- | --- | --- | --- | --- | --- |
| Lorazepam | Multiple | 0.01 mg | Pregnant women | 10 | ...the source sentence... |
```

If the article does not describe any dosing at all, emit a single row of
`N/A` across all six columns (see procedure.md's error-handling rule).

Before continuing, sanity-check:
- columns are exactly the six above, in order,
- one row per **unique** dosing combination,
- `Source text` quotes the article (or is `"N/A"`).

If a check fails, redo this stage once. Then **write the result to
`01_drug_info.md`** in the scratch directory. Stages 2, 3, and 4 read it.
