# Stage 03 — Study info (context tagging)

This file is loaded by `procedure.md`. It tags each retained value with the
characteristic / exposure / outcome it belongs to.

## What you are doing
For **each row** of the stage-1 `[Value]` table, locate that value in the source
table, understand its context, and emit `[Characteristic, Exposure, Outcome]`
(3 columns) — one output row per stage-1 row, in the same order.

## Inputs (read from the scratch directory)
- `01_values.md` — the retained numeric values (the rows to tag).
- `00_markdown_table.md` — the full source table (locate each value here; its row
  and column headers determine the tags).
- `inputs.md` — caption + footnotes, for context.

Read these now.

## How to tag each value
Find the value's cell in the table and read its **row and column headers** — they
usually determine the classification:
- **Characteristic** — a geographic, demographic, or biological feature of the
  subjects (age, sex, race, weight, genetic markers).
- **Exposure** — a factor that might be associated with an outcome (a drug,
  condition, medication). If a column header contains a **drug name**, it is very
  likely an `Exposure`.
- **Outcome** — what the value **measures** (e.g. birth weight, total sleep time,
  symptom reduction). The Outcome describes the *meaning* of the value; it must
  **never** be the numeric value itself.

Use `"N/A"` for any of the three that does not apply.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — every row has
  all **3** values (use `"N/A"` for unknowns).
- `Outcome` is a description, not a number.

## Output of this stage
A markdown table with columns exactly `Characteristic, Exposure, Outcome`, one row
per stage-1 row:

```
| Characteristic | Exposure | Outcome |
| --- | --- | --- |
| infants of substance abuse mothers | cocaine unexposed | total sleep time |
| infants of substance abuse mothers | cocaine exposed | total sleep time |
```

Before continuing, sanity-check:
- row count equals `01_values.md`'s row count, same order,
- no `Outcome` cell is just a number,
- columns are exactly the three above.

If a check fails, redo this stage once. Then **write the result to
`03_study_info.md`**. Stage 4 reads it.
