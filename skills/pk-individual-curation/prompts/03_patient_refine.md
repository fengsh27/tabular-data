# Stage 03 — Patient refine

This file is loaded by `SKILL.md` during the PK individual curation procedure.

## What you are doing
Refine the patient table from stage 2 into
`[Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]`:
normalize Population and Pregnancy stage to standard categories, and add a
Pediatric/Gestational age column. **One row in, one row out** — keep the same
Patient IDs in the same order.

## Inputs (read from the scratch directory)
- `02_patient_table.md` — the `[Patient ID, Population, Pregnancy stage]` rows.
- `00_markdown_table.md` — the source PK table, for context.
- `inputs.md` — the caption, footnotes, and paper title, verbatim.

Read these files now.

## How to refine each row
- **Population** — map to a standard category when it matches one, else keep the
  original wording:
  - "Maternal" (pregnant individuals)
  - "Preterm" / "Premature" (≤ ~37 weeks gestation)
  - "Neonates" / "Newborns" (birth to ~1 month)
  - "Infants" (~1 month to ~1 year)
  - "Children" (~1–12 years)
  - "Adolescents" (~13–17 years)
  - "Adults" (≥ 18 years)
- **Pregnancy stage** — map to a standard category when it matches, else keep
  the original:
  - "Trimester 1" (≤ ~14 weeks), "Trimester 2" (~15–28 weeks),
    "Trimester 3" (≥ ~28 weeks)
  - "Fetus", "Parturition/Labor/Delivery", "Postpartum",
    "Nursing/Breastfeeding/Lactation"
- **Pediatric/Gestational age** — the subject's age (or age range), or pregnancy
  weeks, **only if the source explicitly states an age**. Keep the original
  wording. Do **not** infer age from a sampling time or dosing day (e.g.
  "Concentrations on Day 7" is a time point, not an age). Use `"N/A"` otherwise.

Use `"N/A"` where information cannot be reasonably inferred.

## Reasoning then answer
For each row, note any Population / Pregnancy-stage normalization and where a
Pediatric/Gestational age (if any) came from, then produce the table.

## Output of this stage
A markdown table, columns exactly
`Patient ID`, `Population`, `Pregnancy stage`, `Pediatric/Gestational age`:

```
| Patient ID | Population | Pregnancy stage | Pediatric/Gestational age |
| --- | --- | --- | --- |
| 1 | Maternal | Trimester 3 | N/A |
| 3 | Maternal | Trimester 3 | N/A |
```

Before continuing, sanity-check:
- **same Patient IDs, same count, same order** as `02_patient_table.md`,
- columns are exactly the four above,
- no age was invented from a time point.

If that check fails, redo this stage once.

Then **write the result to `03_patient_refined.md`** in the scratch directory.
Stage 10 (patient matching) reads that file.
