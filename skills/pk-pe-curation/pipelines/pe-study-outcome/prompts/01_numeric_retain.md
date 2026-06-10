# Stage 01 — Numeric retain

This file is loaded by `procedure.md` as the first stage of PE study outcome
curation. It is a purely mechanical scan — every numeric cell becomes a row.

## What you are doing
List **every cell of the source table that contains a digit**, as a one-column
`[Value]` table, in **row-major order** (read left-to-right across each row, top
row first).

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the source table in markdown (Stage 0 output).

Read it now.

## How to extract
- Walk the table cell by cell, row by row, in reading order.
- For each cell whose text **contains at least one digit (0–9)**, emit one row
  with that cell's **verbatim** text in the `Value` column.
- Include the cell even if it also contains non-numeric text (e.g. `10 (5%)`,
  `3.2 ± 0.4`, `0.8-1.2`, `p<0.05`) — keep the whole cell text.
- Skip cells with no digit (pure text labels, blanks).
- Do **not** deduplicate, reorder, calculate, or reformat. One source numeric
  cell → one row, in order.

## Output of this stage
A markdown table with a single column `Value`:

```
| Value |
| --- |
| 10 (5%) |
| 3.2 ± 0.4 |
| 0.8-1.2 |
```

If the table has **no** numeric cells, say so and stop (there is nothing to
curate — see procedure.md's error-handling rules).

Before continuing, sanity-check:
- the column is exactly `Value`,
- every emitted cell contains a digit,
- the rows are in the table's reading order (row-major), none skipped or
  reordered.

If a check fails, redo this stage once. Then **write the result to
`01_values.md`**. Stages 2, 3, and 4 read it.
