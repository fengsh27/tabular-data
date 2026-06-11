# Stage 02 — Parameter value parsing

This file is loaded by `procedure.md`. It parses each retained `Value` into structured
statistic columns.

## What you are doing
For **each row** of the stage-1 `[Value]` table, interpret the value and rewrite
it as `[Main value, Main value unit, Statistics type, Variation type, Variation
value, Interval type, Lower bound, Upper bound, P value]` (9 columns) — one output
row per stage-1 row, in the same order.

## Inputs (read from the scratch directory)
- `01_values.md` — the retained numeric values (the rows to parse).
- `00_markdown_table.md` — the full source table, for context **and for finding
  p-values** (a value's p-value usually lives in a different cell).
- `inputs.md` — caption + footnotes, for units and group labels.

Read these now.

## How to parse each value
- **Main value** — the main parameter value (not a range).
- **Main value unit** — its unit (e.g. `kg`, `g`, `Count`). **Do NOT** put a
  statistic like `SD` here.
- **Statistics type** — how the Main value is summarized: `Mean`, `Median`,
  `Sum`, `Proportion`, `%`, etc. **Required — must be filled.**
- **Variation type** / **Variation value** — the variability measure (e.g. `SD`)
  and its single value.
- **Interval type** / **Lower bound** / **Upper bound** — an interval like
  `95% CI`, `Range`, `IQR`, with its two bounds. An interval of two numbers must
  go into Lower/Upper bound, **never** into Variation value.
- **P value** — the value's p-value. It usually is **not** in the same cell —
  **search the whole main table** for the corresponding p-value and fill it in. If
  the same p-value applies to several rows, fill it into all of them.

### Special case — count + percentage together
If a cell reports both a count and a percentage (e.g. `10 (5%)`): put the count in
`Main value`, set `Main value unit` = `Count` and `Statistics type` = `Sum`, and
put the percentage in `Variation type` = `%` and `Variation value` = the number.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — every row must
  have all **9** values (use `"N/A"` for unknowns).
- **No calculations** — take every value directly from the table.
- A row that cannot be parsed → all `"N/A"`.

## Output of this stage
A markdown table with columns exactly
`Main value, Main value unit, Statistics type, Variation type, Variation value,
Interval type, Lower bound, Upper bound, P value`, one row per stage-1 row.

Before continuing, sanity-check:
- row count equals `01_values.md`'s row count, same order,
- every row has all 9 columns filled (value or `"N/A"`),
- no unit cell holds a statistic; no interval sits in `Variation value`.

If a check fails, redo this stage once. Then **write the result to
`02_param_values.md`**. Stage 4 reads it.
