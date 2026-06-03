# Stage 05 — Parameter-type alignment

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are doing
Reshape the summary table so that **PK parameter types live in a column named
`Parameter type`**, one parameter per row. PK tables represent parameters two
different ways; this stage normalizes both into the same orientation for the
stages that follow.

## Inputs (read from the scratch directory)
- `04_summary_only.md` — the summary-only table from stage 4.

Read this file now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## Decide the table's orientation
Examine how the PK parameter type (e.g. Cmax, tmax, t½, AUC, or a measured
quantity / specimen-concentration column) is expressed:

- **Case A — parameters are row values under one column.** One column holds the
  parameter names (e.g. a "Parameter" column whose cells are `Cmax`, `tmax`,
  …), and other columns hold the values. → **Rename that column to
  `Parameter type`.** Keep every row and the other columns as-is.

- **Case B — parameters are the column headers.** Each column header *is* a
  parameter / measured quantity (e.g. `Cmax (ng/mL)`, `Cord blood (ng/ml)`,
  `Maternal blood (ng/ml)`), and the values sit beneath them. → **Transpose**
  the table so each former column header becomes a row under a new first column
  named `Parameter type`. The first original column's values become the new
  value-column header.

If you are unsure, ask: "is there a single column whose cells name the
parameters?" If yes → Case A. If the parameters are spread across the headers →
Case B.

## Reasoning then answer
State which case applies and why (point to the column or the headers that carry
the parameter names), then produce the aligned table.

## Output of this stage
A markdown table whose **first column is `Parameter type`**. Do not reformat
the values or intervals; only change the orientation / column name.

Before continuing, sanity-check:
- the table has a column named exactly `Parameter type`,
- no value cells were lost, merged, or reformatted,
- (Case B) every former column header now appears as a row in `Parameter type`.

If that check fails, redo this stage once.

Then **write the result to `05_param_aligned.md`** in the scratch directory.
Stages 6 and 7 read that file, not this message — so write the complete, exact
table.

## Worked example (Case B — transpose)

**`04_summary_only.md`**:

```
| Parturient | Cord blood (ng/ml) | Maternal blood (ng/ml) | Collection time(min) | Cord blood/maternal blood |
| --- | --- | --- | --- | --- |
| Mean CI 95% | 6.78 (5.39–8.17) | 9.91 (7.68–12.14) | 293.4 (163.2–423) | 0.73 (0.52–0.94) |
```

**Reasoning**: there is no single column whose cells name the parameters; the
measured quantities are the column headers (Cord blood, Maternal blood, …). →
Case B → transpose. The first column's cell ("Mean CI 95%") becomes the value
column's header.

**Result** (`05_param_aligned.md`):

```
| Parameter type | Mean CI 95% |
| --- | --- |
| Cord blood (ng/ml) | 6.78 (5.39–8.17) |
| Maternal blood (ng/ml) | 9.91 (7.68–12.14) |
| Collection time(min) | 293.4 (163.2–423) |
| Cord blood/maternal blood | 0.73 (0.52–0.94) |
```

## Worked example (Case A — rename)

**Input** (parameters already in a column):

```
| Parameter | Value | Unit |
| --- | --- | --- |
| Cmax | 95 | ng/mL |
| tmax | 3.1 | h |
```

**Reasoning**: the "Parameter" column's cells name the parameters → Case A →
rename that column to `Parameter type`; keep all rows.

**Result**:

```
| Parameter type | Value | Unit |
| --- | --- | --- |
| Cmax | 95 | ng/mL |
| tmax | 3.1 | h |
```
