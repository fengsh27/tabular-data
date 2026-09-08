# Stage 03 — Patient cohort refinement

This file is loaded by `procedure.md` during the PK summary curation procedure.

## What you are doing
Refine the patient table from stage 2 into a more detailed table. You will:
- normalize **Population** and **Pregnancy stage** to standard categories, and
- add a new **Pediatric/Gestational age** column,

while keeping exactly the same rows, in the same order, with the same
Subject N values. This is a row-preserving transform, not a re-extraction.

## Inputs (read from the scratch directory)
- `02_patient_table.md` — the patient table from stage 2 (the rows to refine).
- `00_markdown_table.md` — the source PK table, for context.
- `inputs.md` — the caption, footnotes, and paper title, verbatim.

Read these files now; do not rely on text remaining in the conversation, which
may have been summarized.

## Output columns
`[Population, Pregnancy stage, Pediatric/Gestational age, Subject N]` — one row
for each row of `02_patient_table.md`, in the same order.

## Definitions and normalization

**Population** — the age group. If it matches one or more of these standard
categories, replace the original wording with the standard category
(categories, if more than one applies); otherwise keep the original wording:
- "Maternal" (pregnant individuals)
- "Preterm" / "Premature" (typically ≤ 37 weeks gestation)
- "Neonates" / "Newborns" (birth to ~1 month)
- "Infants" (~1 month to ~1 year)
- "Children" (~1 year to ~12 years)
- "Adolescents" / "Teenagers" (~13 to ~17 years)
- "Adults" (18 years or older)

**Pregnancy stage** — if it matches one of these standard categories, replace
it; otherwise keep the original wording:
- "Trimester 1" (up to ~14 weeks)
- "Trimester 2" (~15–28 weeks)
- "Trimester 3" (~≥ 28 weeks)
- "Fetus" / "Fetal Stage"
- "Parturition" / "Labor" / "Delivery"
- "Postpartum" (~6–8 weeks after birth)
- "Nursing" / "Breastfeeding" / "Lactation"

**Pediatric/Gestational age** — the child's age (or age range) at a specific
point in the study, or the pregnancy week count. Retain the original wording
where possible. Only fill this if an age is **explicitly stated**.

- **Look before you write `N/A`.** Scan the row labels and column headers of
  `00_markdown_table.md`, then `inputs.md`, for a stated age. Qualifying labels
  include "Age", "Gestational age", "Gestational age at birth", "GA",
  "Postmenstrual age", "Postnatal age", "Age at delivery". Under such a label,
  values may look like `38w0d`, "38 + 3", "at 32 weeks gestation", "38–40 wk",
  or "6 months" — a bare duration with no age label is a time point, not an age.
- **Gestational age qualifies, and it is not the mothers' age in years.** For a
  maternal cohort measured at or around delivery, the pregnancy's gestational
  age *is* that cohort's value for this column. Do not reject it because the
  mothers' chronological age is absent or sits in another table.
- Do **not** infer age from the timing of data recording or drug
  administration. For example, "Concentrations on Day 7" is a measurement time
  point, not an age, and must not be placed here.
- Use `"N/A"` only after that scan finds nothing.

**Subject N** — carry through unchanged from `02_patient_table.md`.

Use `"N/A"` for any element that cannot be reasonably inferred.

## Row-preservation constraints (strict)
- Process **only** the rows in `02_patient_table.md`, rows 0 through the last,
  and output **exactly** that many rows — no more, no fewer.
- Keep the **original row order**. Do not shuffle, merge, split, or omit rows.
- The **Subject N of each output row must equal the Subject N of the
  corresponding input row.**

## Reasoning then answer
For each input row, state how you mapped its Population and Pregnancy stage to
standard categories (or why you kept the original), and what explicitly
justifies a Pediatric/Gestational age. If you are writing `N/A` for the age,
name the row labels and headers you checked — "the source states no age" is a
valid reason only once you have quoted what the table's labels actually are.
Then produce the table.

## Output of this stage
Produce a markdown table with exactly these columns:

```
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
|------------|-----------------|---------------------------|-----------|
| Adults | N/A | N/A | 12 |
```

Before continuing, sanity-check:
- the output has the **same number of rows** as `02_patient_table.md`,
- the Subject N column matches `02_patient_table.md` row-for-row, in order,
- every cell is filled (a literal `N/A` counts as filled),
- **no stated age was missed** — if `00_markdown_table.md` has a row or column
  naming an age or a gestational age, this column is not all `N/A`.

If the row count or Subject N ordering does not match the input, redo this
stage once — re-anchor on the input rows one at a time rather than
re-extracting from the source table.

Then **write the table to `03_patient_refined.md`** in the scratch directory.
Stage 10 (patient matching) reads that file, not this message — so write the
complete, exact table.

## Worked example

**`02_patient_table.md`**:

```
| Population | Pregnancy stage | Subject N |
|------------|-----------------|-----------|
| Pregnant women, 3rd trimester | N/A | 20 |
| Newborns at birth | N/A | 18 |
| Pregnant women at delivery | N/A | 15 |
```

with `00_markdown_table.md` carrying a row labelled
`Gestational age at delivery | ... | 38-40 wk`.

**Reasoning**: "Pregnant women, 3rd trimester" → Population "Maternal",
Pregnancy stage "Trimester 3"; the source states no age under any age label, so
Pediatric/Gestational age is N/A. "Newborns at birth" → Population "Neonates";
not pregnancy-related, so Pregnancy stage N/A; "at birth" is a stage
descriptor, not a stated age → N/A. "Pregnant women at delivery" → Population
"Maternal", Pregnancy stage "Delivery"; the source table has
a `Gestational age at delivery` row reading "38-40 wk", which is an explicitly
stated gestational age → Pediatric/Gestational age "38-40 wk". All Subject Ns
carried through unchanged; 3 rows in, 3 rows out.

**Result**:

```
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
|------------|-----------------|---------------------------|-----------|
| Maternal | Trimester 3 | N/A | 20 |
| Neonates | N/A | N/A | 18 |
| Maternal | Delivery | 38-40 wk | 15 |
```
