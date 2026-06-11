# Stage 01 — Population characteristic information (from full text)

This file is loaded by `procedure.md` as the first stage of PK population summary
curation.

## What you are doing
Read the paper's full text and extract every **unique** population-characteristic
combination it describes, as rows of
`[Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now. Provide all **6 values** per combination; use `"N/A"` where a value
is unavailable.

## How to extract
- **Population characteristic** — a population-focused characteristic, **not a PK
  parameter**. Examples:
  - `Age`, `Sex`, `Weight`, `Gender`, `Race`, `Ethnicity`
  - `Socioeconomic status`, `Education`, `Marital status`
  - `Comorbidity`, `Drug indication`, `Adverse events`
  - `Severity`, `BMI`, `Smoking status`, `Alcohol use`, `Blood pressure`
- **Characteristic sub-category** — a level/option under the characteristic:
  - Sex → `Male`, `Female`; Race → `White`, `Black`, `Asian`, `Hispanic`;
    Comorbidity → `Diabetes`, `Hypertension`, `Asthma`; Adverse events → `Mild`,
    `Moderate`, `Severe`. If none, use `"N/A"`.
- **Characteristic values** — all numerical descriptors as reported (means,
  ranges, p-values, …). If several are reported, include them all (stage 3 splits
  them apart).
- **Population** — the group the data came from (e.g. `healthy adults`,
  `pregnant women`).
- **Population N** — the number of individuals in that group. State the basis: it
  must be **explicitly stated** in the text, not calculated. Cite the sentence.
- **Source text** — the original sentence/excerpt. Use `"N/A"` if none.

### Critical rule — exclude summed totals
If the text reports both individual `Population N` values and a summed total,
include **only the individual values** (e.g. "16 in T1, 18 in T2, 34 total" →
report `16` and `18`, exclude `34`).

Confirm each combination against the text. Do not invent or calculate.

## Reasoning then answer
Explain which sentences describe the population and how you grouped them, then
produce the table.

## Output of this stage
A markdown table with columns exactly
`Population characteristic`, `Characteristic sub-category`, `Characteristic values`,
`Population`, `Population N`, `Source text`:

```
| Population characteristic | Characteristic sub-category | Characteristic values | Population | Population N | Source text |
| --- | --- | --- | --- | --- | --- |
| Weight | N/A | 76.8 (67.4-86.2) | Pregnant women | 10 | ...the sentence... |
| Age | N/A | 23.3 (19.0-27.6) | Pregnant women | 10 | ...the sentence... |
```

If the article describes no population characteristics, emit a single all-`N/A`
row across the six columns.

Before continuing, sanity-check:
- columns are exactly the six above, in order,
- every `Population characteristic` is a characteristic, **not** a PK parameter,
- no summed-total row is present when its parts are.

If a check fails, redo this stage once. Then **write the result to
`01_characteristic_info.md`**. Stages 2, 3, and 4 read it.
