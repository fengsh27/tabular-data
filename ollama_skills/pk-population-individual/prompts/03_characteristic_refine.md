# Stage 03 — Characteristic value refine

This file is loaded by `procedure.md`. It reduces each stage-1 row's raw
`Characteristic values` to the patient's value plus its unit. Unlike the summary
skill there is **no** statistic decomposition — each row is one raw value.

## What you are doing
For **each row** of the stage-1 table, emit `[Patient ID, Main value, Unit]` — one
output row per stage-1 row, in the same order.

## Inputs (read from the scratch directory)
- `01_characteristic_info.md` — the stage-1 table (its `Characteristic values`
  column is the raw signal; carry `Patient ID` through).
- `inputs.md` — title + full text, to resolve the unit when ambiguous.

Read these now.

## How to refine each row
- **Patient ID** — carry through from stage 1 (verbatim).
- **Main value** — the patient's value of the characteristic. Usually you can use
  the `Characteristic values` directly.
- **Unit** — the measurement unit of the Main value (e.g. `year`, `kg`).

Use `"N/A"` for any field that cannot be reasonably inferred.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — no more, no less.
- The `Patient ID` of each output row must equal that of the stage-1 row.
- Take values from the source; do not calculate.

## Reasoning then answer
Explain how you read each value/unit, then produce the table.

## Output of this stage
A markdown table with columns exactly `Patient ID, Main value, Unit`, one row per
stage-1 row.

Before continuing, sanity-check:
- row count equals `01_characteristic_info.md`'s row count, same order,
- `Patient ID` matches stage 1 row-for-row.

If a check fails, redo this stage once. Then **write the result to
`03_characteristic_refined.md`**. Stage 4 reads it.
