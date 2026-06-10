# Refine population / patient demographics (generic, shared)

This is the shared **patient/population-refinement** step for the full-text
curation skills (pk-drug, pk-specimen, pk-population, …). They all normalize the
same demographic fields the same way, so the logic lives here once. A pipeline's
refine stage points here and supplies:

- **`<INPUT_TABLE>`** — the stage-1 combinations table (its rows are the unit of
  work; one refined row out per input row, in the same order).
- **`<KEY-COLUMN>`** — the column that ties a refined row back to its input row:
  `Patient ID` for individual pipelines, `Population N` for summary pipelines.
- **`<OUTPUT-COLUMNS>`** — the exact output column list and order the calling
  stage wants (it always contains `Population, Pregnancy stage,
  Pediatric/Gestational age` plus the key column; the key goes **first** for
  `Patient ID`, **last** for `Population N`).
- **`<TITLE+FULLTEXT>`** — the paper title and full text (read from the run's
  `inputs.md`), used to resolve demographics the stage-1 table only hints at.

## What you are doing
For **each row** of `<INPUT_TABLE>`, in order, determine the refined demographics
`[Population, Pregnancy stage, Pediatric/Gestational age]` (carrying the
`<KEY-COLUMN>` value through unchanged), using the full text to disambiguate.

## The fields and their canonical categories
1. **Population** — the age/demographic group. Map to one of these standard
   categories when it fits (otherwise keep the original wording):
   - `Nonpregnant`
   - `Maternal` (pregnant individuals)
   - `Pediatric` (≈ birth to ~17 years)
   - `Adults` (typically ≥ 18 years)

2. **Pregnancy stage** — pregnancy-related timing. Map to a standard category
   when it fits (otherwise keep the original wording):
   - `Pre-pregnancy`
   - `Trimester 1` (up to ~14 weeks), `Trimester 2` (~15–28 weeks),
     `Trimester 3` (~≥ 28 weeks)
   - `Fetus` (the developing baby during pregnancy)
   - `Parturition` / `Labor` / `Delivery` (childbirth)
   - `Postpartum` (~6–8 weeks after birth)
   - `Nursing` / `Breastfeeding` / `Lactation` (the breastfeeding period)

3. **Pediatric/Gestational age** — the child's age or age-range at a point in the
   study, or the pregnancy weeks. **Retain the original wording.**
   - Only fill it when the age is **explicitly stated**. Do **not** infer age
     from the timing of a measurement or of drug administration. For example
     "concentrations on Day 7" is a *time point*, not an age — leave the age
     `"N/A"`.

Use `"N/A"` for any field that cannot be reasonably inferred.

## Hard rules
- Process **exactly** the rows of `<INPUT_TABLE>` — one output row per input row,
  **no more, no less**, in the **same order**. Do not shuffle, merge, or drop rows.
- The `<KEY-COLUMN>` value of each output row must equal that of the
  corresponding input row (verbatim).
- Reason explicitly (how you mapped each Population / Pregnancy stage, where in
  the text the age came from) **before** emitting the table.

## Output of this stage
A markdown table with columns exactly `<OUTPUT-COLUMNS>`, one row per input row.

Before continuing, sanity-check:
- the row count equals `<INPUT_TABLE>`'s row count and the order is preserved,
- the `<KEY-COLUMN>` column matches `<INPUT_TABLE>` row-for-row,
- every Population / Pregnancy stage is either a canonical category or
  deliberately-retained original wording,
- no Pediatric/Gestational age was inferred from a measurement/administration
  time.

If a check fails, redo this stage once, then carry the result forward (noting any
residual issue) and write it to the calling stage's output file.
