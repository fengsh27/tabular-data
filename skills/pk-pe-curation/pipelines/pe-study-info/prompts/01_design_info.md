# Stage 01 — Study design info (from full text)

This file is loaded by `procedure.md` as the first stage of PE study info curation.

## What you are doing
Read the paper's full text and summarize the study in a **single row** of
`[Study type, Study design, Data source]`.

## Inputs (read from the scratch directory)
- `inputs.md` — the paper title + full text, verbatim.

Read it now.

## How to extract
- **Study type** — the area of interest. Choose one of:
  `Pharmacoepidemiology` / `Clinical Trials` / `Pharmacokinetics` /
  `Pharmacodynamics` / `Pharmacogenetics`.
- **Study design** — the design described. Common examples: `Prospective cohort
  study`, `Retrospective cohort study`, `Randomized controlled trial (RCT)`,
  `Double-blind randomized trial`, `Case-control study`, `Cross-sectional study`,
  `Systematic review and meta-analysis`, `Open-label study`, `Nested case-control
  study`, `Pilot study`, `Chart review`, `Observational study`.
  - If several designs are mentioned (e.g. "prospective, randomized,
    double-blind"), list them as **one string** in the same order.
- **Data source** — the primary location(s)/site(s) where the data was collected
  or the study conducted (hospital, database, registry, geographic location).

## Reasoning then answer
Explain how you classified the study type/design and where the data came from,
then produce the one-row table.

## Output of this stage
A markdown table with columns exactly `Study type`, `Study design`, `Data source`,
**one row**:

```
| Study type | Study design | Data source |
| --- | --- | --- |
| Pharmacoepidemiology | Prospective Randomized Double-blind Investigation | OSUMC |
```

Use `"N/A"` for anything the article does not state.

Before continuing, sanity-check:
- columns are exactly the three above, in order, with **one** data row,
- `Study type` is one of the allowed values.

If a check fails, redo this stage once. Then **write the result to
`01_design_info.md`**. Stages 2 and 3 read it.
