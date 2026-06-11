# Stage 02 — Study design refine (population + criteria + outcomes)

This file is loaded by `procedure.md`. It extracts the complementary study-metadata
fields as a **single row**, enhancing the stage-1 design info.

## What you are doing
Read the full text and extract the one-row
`[Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N, Drug name, Outcomes]`.

## Inputs (read from the scratch directory)
- `01_design_info.md` — the stage-1 `[Study type, Study design, Data source]` row.
- `inputs.md` — title + full text.

Read these now.

## How to extract
- **Population** — the age/demographic group. Map to a canonical category when it
  fits: `Nonpregnant`, `Maternal` (pregnant), `Pediatric` (birth–~17 y),
  `Adults` (≥18 y); otherwise keep the original wording.
- **Inclusion criteria** — characteristics participants must have to qualify.
  **Use the article's exact wording.**
- **Exclusion criteria** — characteristics that disqualify participants. **Use the
  article's exact wording.**
- **Pregnancy stage** — map to a canonical category when it fits: `Trimester 1`
  (≤14 wk), `Trimester 2` (~15–28 wk), `Trimester 3` (~≥28 wk), `Fetus`,
  `Parturition`/`Labor`/`Delivery`, `Postpartum`, `Nursing`/`Breastfeeding`/
  `Lactation`; otherwise original wording. `N/A` if not obstetric.
- **Subject N** — the number of subjects in the population.
- **Drug name** — the drug(s) of interest relating to the outcomes.
- **Outcomes** — the key outcome(s) the investigators considered most important.
  **Use the article's exact wording.**

Use `"N/A"` for anything that cannot be reasonably inferred.

## Reasoning then answer
Explain where each field came from in the text, then produce the one-row table.

## Output of this stage
A markdown table with columns exactly
`Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N,
Drug name, Outcomes`, **one row**.

Before continuing, sanity-check:
- columns are exactly the seven above, in order, with **one** data row,
- inclusion/exclusion/outcomes use the article's exact wording,
- `Subject N` is an integer or `"N/A"`.

If a check fails, redo this stage once. Then **write the result to
`02_design_refined.md`**. Stage 3 reads it.
