# Stage 03 — Characteristic value refine (decompose statistics)

This file is loaded by `procedure.md`. It decomposes each stage-1 row's raw
`Characteristic values` into the structured statistic columns.

## What you are doing
For **each row** of the stage-1 table, parse the `Characteristic values` into
`[Main value, Unit, Statistics type, Variation type, Variation value, Interval
type, Lower bound, Upper bound]` — one output row per stage-1 row, in the same
order.

## Inputs (read from the scratch directory)
- `01_characteristic_info.md` — the stage-1 table (its `Characteristic values`
  column is the raw signal to decompose).
- `inputs.md` — title + full text, to resolve the statistic/unit when ambiguous.

Read these now.

## How to refine each row
- **Main value** — the primary value of the characteristic. If there is no
  variation or interval, use the `Characteristic values` directly. A non-standard
  ratio like `4/5/4/3` is kept verbatim as the Main value.
- **Unit** — the measurement unit of the Main value (e.g. `year`, `kg`, `kg/m²`).
- **Statistics type** — how the Main value is summarized: `Mean`, `Median`,
  `Count`, etc. **This column is required and must be filled in.**
- **Variation type** — the variability measure: `Standard Deviation (SD)`,
  `Proportion (%)`, etc.
- **Variation value** — the single value (not a range) of that variation.
- **Interval type** — the interval describing uncertainty/spread: `Minmax`,
  `IQR`, etc.
- **Lower bound** / **Upper bound** — the interval's bounds.

Use `"N/A"` for any field that cannot be reasonably inferred.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — no more, no less.
- Take values from the source; do not calculate.

## Reasoning then answer
Explain how you parsed each `Characteristic values` string into the statistic
fields, then produce the table.

## Output of this stage
A markdown table with columns exactly
`Main value, Unit, Statistics type, Variation type, Variation value, Interval
type, Lower bound, Upper bound`, one row per stage-1 row.

Before continuing, sanity-check:
- row count equals `01_characteristic_info.md`'s row count, same order,
- `Statistics type` is filled on every row,
- a bare ratio/string with no stat sits in `Main value` with the rest `"N/A"`.

If a check fails, redo this stage once. Then **write the result to
`03_characteristic_refined.md`**. Stage 4 reads it.
