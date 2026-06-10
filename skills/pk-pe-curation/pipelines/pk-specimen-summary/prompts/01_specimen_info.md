# Stage 01 — Specimen / sampling information (from full text)

This file is loaded by `procedure.md` as the first stage of PK specimen summary
curation.

## What you are doing
Read the paper's full text and extract every **unique** specimen-sampling
combination it describes, as rows of
`[Specimen, Sample N, Sample time, Population, Population N]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
Identify each unique combination of the fields below. A new combination of *any*
field is a new row.

- **Specimen** — the type of biological sample collected (e.g. `urine`, `blood`,
  `plasma`, `cord blood`, `milk`).
- **Sample N** — the number of samples analyzed for that specimen. State the
  basis for each value: it must be **explicitly stated** in the text, never
  derived by calculation or inference. Cite the supporting sentence.
- **Sample time** — the specific moment (numeric value or time range) when the
  specimen is sampled.
- **Population** — the group the samples came from (e.g. `healthy adults`,
  `pregnant women`).
- **Population N** — the number of individuals in that population group.

### Critical rule — exclude summed totals
If the text reports **both** individual `Sample N` values (e.g. per timepoint or
per subgroup) **and** a summed total, include **only the individual values** —
do not include the total, even if explicitly stated, to avoid double-counting.
Example: "16 samples in the first trimester, 18 in the second, and 34 across
both" → report `16` and `18`, exclude `34`.

Confirm each combination against the text before including it. Do not invent or
calculate.

## Reasoning then answer
First explain your thought process — which sentences describe sampling, how you
grouped them, and (per Sample N) the exact sentence that justifies each count —
then produce the table.

## Output of this stage
A markdown table with columns exactly
`Specimen`, `Sample N`, `Sample time`, `Population`, `Population N`:

```
| Specimen | Sample N | Sample time | Population | Population N |
| --- | --- | --- | --- | --- |
| Urine | 20 | 0-2 h | Pregnant women | 10 |
```

If the article describes no specimen sampling at all, emit a single row of `N/A`
across all five columns (see procedure.md's error-handling rule).

Before continuing, sanity-check:
- columns are exactly the five above, in order,
- one row per **unique** combination,
- no summed-total row is present when its individual parts are.

If a check fails, redo this stage once. Then **write the result to
`01_specimen_info.md`** in the scratch directory. Stages 2, 3, and 4 read it.
