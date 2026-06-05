# Stage 12 — Time extraction

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
For each row of each sub-table, add a `[Time value, Time unit]` pair: the
recorded time point at which the data was sampled / the dose administered.
Many rows legitimately have no time → `[N/A, N/A]`.

## Inputs (read from the scratch directory)
- `07_subtables.md` — the per-parameter sub-tables (the rows to annotate).
- `05_param_aligned.md` — the aligned main table, for context.
- `inputs.md` — caption + title.

Read these files now.

## What counts as a time
- **Time value** — a specific recorded moment or range when data was recorded
  or a dose given (a sampling time, dosing time, or observation time).
- **Time unit** — its unit: `Hour`, `Min`, `Day`, …

**Include**, e.g.: `0-12` (a dosing period), `24` (a collection time),
`5 min` (a measured event), `293.4` minutes (a reported collection time).

**Do NOT include** PK parameters that merely sound time-like:
- `Tmax` (a parameter, not a recorded time),
- `t½` / half-life values,
- elimination-rate-constant values.

## Rules
- Process **exactly the rows of the sub-table**, keeping each row's `Row`
  join-key value — same `Row` values as the sub-table.
- **No calculations** — copy times verbatim from the table/caption.
- If a sub-table yields no valid times at all, every row is `[N/A, N/A]`.

## Reasoning then answer
For each row, state whether a recorded time applies and its source, then
produce the table.

## Output of this stage
For each sub-table, a three-column markdown table — the `Row` join key (copied
verbatim from the sub-table) plus the time pair, one row per sub-table row. Use
the same `## Sub-table N` headings as `07_subtables.md`:

```
| Row | Time value | Time unit |
| --- | --- | --- |
| 1 | N/A | N/A |
| 3 | 293.4 | minutes |
```

(Keep the sub-table's `Row` order; only the values change per row.)

Before continuing, sanity-check:
- each output table has the **same `Row` values** as its sub-table (same set,
  same order),
- no PK parameter (Tmax, t½, …) was mistaken for a recorded time.

If that check fails, redo this stage once.

Then **write the result to `12_time.md`** in the scratch directory.
Stage 13 (assembly) reads that file.
