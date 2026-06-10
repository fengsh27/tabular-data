# Stage 03 — Sample time + unit extraction

This file is loaded by `procedure.md`. It derives the numeric sampling time and its
unit for each stage-1 row.

## What you are doing
For **each row** of the stage-1 table, emit `[Sample time, Time unit, Source
text]` — one output row per stage-1 row, in the same order.

## Inputs (read from the scratch directory)
- `01_specimen_info.md` — the stage-1 specimen table (the rows to process; its
  `Sample time` column is the raw signal to normalize).
- `inputs.md` — title + full text, to confirm the time and resolve the unit.

Read these now.

## How to extract each row
- **Sample time** — the moment(s) the specimen was sampled, kept strictly
  **numeric**. Examples: `0`, `24`, `0, 2, 4`, `0-2`, `0-2, 2-4, 4-6`.
  - If multiple values are separated by commas, **preserve the comma-separated
    string as one cell** (do not split into rows).
  - If a range is given (e.g. `0-1`), keep the **whole range string** as one item.
  - Strip any unit/words from this cell — the unit goes in `Time unit`.
- **Time unit** — the unit of the sample time: `Second`, `Minute`, `Hour`, `Day`.
- **Source text** — the original sentence/excerpt the time came from. Use `"N/A"`
  if none can be found.

## Hard rules
- Exactly one output row per stage-1 row, in the **same order** — no more, no
  less.
- **No calculations.** Every value comes directly from the text; do not convert
  units or compute times.
- If no valid time data exists for a row, use `["N/A", "N/A", "<source or N/A>"]`.

## Reasoning then answer
Explain how you read each sampling time and unit from the text, then produce the
table.

## Output of this stage
A markdown table with columns exactly `Sample time, Time unit, Source text`, one
row per stage-1 row.

Before continuing, sanity-check:
- row count equals `01_specimen_info.md`'s row count, same order,
- every `Sample time` is numeric / a numeric list / a numeric range / `"N/A"`
  (no stray words or units).

If a check fails, redo this stage once. Then **write the result to `03_time.md`**
in the scratch directory. Stage 4 reads it.
