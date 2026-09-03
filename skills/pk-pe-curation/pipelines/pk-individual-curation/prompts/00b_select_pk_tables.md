# Stage 0b — Select the PK tables

This file is loaded by `procedure.md` when the input contains **more than one
table**. Its job is to decide *which* of the input tables are pharmacokinetics
(PK) tables worth curating, so the rest of the pipeline (stages 1–14) runs only
on those — once per selected table.

Skip this stage entirely when there is exactly one input table: that table is
the one to curate.

## What you are doing
You are a biomedical data assistant specializing in pharmacokinetics (PK).
Read every input table and identify **all** tables relevant to PK —
specifically those reporting **ADME** data (Absorption, Distribution,
Metabolism, Excretion). Be inclusive of genuine PK tables and strict about the
exclusions below.

## Input (read from the scratch directory)
- `00_all_tables.md` — every input table, each preceded by a unique label
  (`## table_1`, `## table_2`, …) and its caption/footnote.

Read it now.

## Inclusion criteria — select a table if it contains ANY of:
- **Drug concentration measurements** in a biological matrix: plasma, serum,
  whole blood, urine, cord blood (umbilical venous/arterial), breast milk,
  amniotic fluid, cerebrospinal fluid (CSF), tissue.
- **PK parameters** (usually abbreviated, with units like ng/mL, L/h, h):
  - AUC (area under the curve)
  - Cmax (maximum concentration), Tmax (time to Cmax)
  - t½ (half-life), Kel (elimination rate constant)
  - Vd (volume of distribution), CL (clearance)
  - concentration ratios (e.g. milk-to-plasma, mother-to-child serum ratio)
- **ADME-related characteristics**, including concentration–time profiles and
  cumulative excretion data.

## Exclusion criteria — do NOT select a table that:
- primarily presents **regression models**, **covariate analyses**,
  **population-PK (PopPK) modeling results**, **statistical modeling**, or
  **correlations between PK parameters and other variables**;
- focuses only on **patient demographics / baseline characteristics**,
  **treatment groups or study-arm information**, or **non-PK safety outcomes**;
- **summarises cases reported in OTHER publications** — a literature-review
  table. The tell is a per-row citation, author name, or bracketed reference
  number (e.g. `Klenske et al. 2019 [12]`), often sitting in a column whose
  header is empty and therefore rendered `Unnamed_0`. Those subjects belong to
  the cited papers, not to this one, so none of their values may be curated.
  **A citation is not a subject identifier**: exclude a table whose only
  per-row identifier is a citation, however much its caption ("Cases of …")
  and its concentration columns look like per-individual PK data.

When a table is borderline, judge by its main content: if its primary purpose
is reporting measured PK concentrations or parameters, include it; if its
primary purpose is modeling, demographics, or safety, exclude it.

This skill curates **individual, per-subject** data. Among the PK tables, prefer
those that report **per-individual** values — rows keyed by a patient / subject /
case id, each with that subject's own measurements. You do not need to exclude a
table merely because it also contains some aggregate rows (Mean / Median / N) —
the summary-data deletion stage (stage 4) drops those downstream, keeping the
individual rows. A table that is *only* cohort-level summary statistics, with no
per-subject rows, is better handled by pk-summary-curation.

## Reasoning
Before answering, think table by table: name each table's main content and
state whether it is included or excluded and why. Keep it concise.

## Output of this stage
Write `00_selection.md` to the scratch directory with:
1. A short **reasoning** paragraph (≤200 words) covering each table's verdict.
2. A **Selected tables** line: the labels you selected, e.g. `table_1, table_3`.
   - If no table qualifies, write `Selected tables: (none)` and say so plainly
     in your reply to the user — there is nothing to curate.

The procedure.md procedure then runs stages 1–14 once per selected table, each in
its own `table_<n>/` scratch sub-directory, and combines the results.
