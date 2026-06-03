# Stage 02 — Patient / cohort extraction

This file is loaded by `SKILL.md` during the PK summary curation procedure.

## What you are extracting
The set of unique **[Population, Pregnancy stage, Subject N]** combinations
described by the PK table. This set defines the cohorts that parameter values
will later be attributed to.

## Inputs (read from the scratch directory)
- `00_markdown_table.md` — the PK table in markdown (Stage 0 output).
- `inputs.md` — the caption, footnotes, and paper title, verbatim.

Read these files now; do not rely on the table text remaining in the
conversation, which may have been summarized.

## Definitions
- **Population** — the patient age/condition group (e.g. "Healthy adults",
  "Neonates", "Patients with hepatic impairment").
- **Pregnancy stage** — the pregnancy stage of the subjects if the study is
  obstetric (e.g. "1st trimester", "Postpartum"); otherwise `"N/A"`.
- **Subject N** — the number of subjects for the specific parameter, or the
  number of samples with quantifiable levels of the analyte. This is a count
  of people, **not** a patient ID.

## Procedure
1. Build the integer worklist. Scan `00_markdown_table.md` **and** the caption
   in `inputs.md` for every pure integer, then dedupe. Exclude:
   - numbers with a decimal point (3.14, 1.0, .99),
   - numbers immediately followed by `%` or `％` (8%, 8 %),
   - the number 0.
   Call this list the **N-candidates**.
2. Analyze the table row by row and column by column, and determine every
   unique [Population, Pregnancy stage, Subject N] combination.
3. **Account for every N-candidate explicitly.** For each integer in the
   worklist, state whether it is a Subject N and, if so, which population it
   belongs to. Subject N often varies slightly across parameters within one
   population because data availability differs per parameter — when it does,
   you must emit a separate combination for each distinct N.
   - Example: a population has 8 subjects, but 5, 6, and 7 of them have
     quantifiable values for different parameters → emit combinations with
     Subject N of 5, 6, 7, **and** 8.
   - Do not confuse a patient ID with Subject N.
   - Use `"N/A"` for an N you genuinely cannot determine.
4. Verify each combination against the table or caption before including it.
5. If Population or Pregnancy stage is missing, first try to infer it from
   context (related rows, caption, common PK knowledge). Use `"N/A"` only as a
   last resort.

## Reasoning then answer
Work through your reasoning explicitly first — walk the N-candidate list and
justify each combination against specific rows or caption text — and only then
produce the result.

## Output of this stage
Produce a markdown table with exactly these columns, one row per unique
combination (drop exact duplicate rows):

```
| Population | Pregnancy stage | Subject N |
|------------|-----------------|-----------|
| Healthy adults | N/A | 12 |
| Cirrhotic patients | N/A | 8 |
```

Before continuing, sanity-check the table:
- at least one row (if no cohort information exists at all, emit a single row
  `| N/A | N/A | N/A |`),
- every cell filled (a literal `N/A` counts as filled),
- every N-candidate from step 1 has been either placed in a Subject N cell or
  explicitly explained as not being a Subject N.

If that check fails, redo this stage once, paying attention to what was wrong
the first time.

Then **write the table to `02_patient_table.md`** in the scratch directory.
Stage 3 (patient refine) reads that file, not this message — so write the
complete, exact table.

## Worked example

**Table** (truncated):

```
| Parameter | Healthy (n=12) | Cirrhotic (n=8) |
|-----------|----------------|-----------------|
| AUC0-∞ (ng·h/mL) | 1240 ± 180 (n=12) | 2890 ± 410 (n=7) |
| Cmax (ng/mL)     | 95 ± 14 (n=11)    | 142 ± 22 (n=8)   |
```

**Caption**: "… in 12 healthy volunteers and 8 patients with hepatic
cirrhosis."

**N-candidates**: 12, 8, 7, 11 (0 and any percentages excluded).

**Reasoning**: 12 and 11 both attach to the healthy group (12 overall, 11 with
a quantifiable Cmax); 8 and 7 both attach to the cirrhotic group. All four are
Subject Ns, none is a patient ID.

**Result**:

```
| Population | Pregnancy stage | Subject N |
|------------|-----------------|-----------|
| Healthy volunteers | N/A | 12 |
| Healthy volunteers | N/A | 11 |
| Cirrhotic patients | N/A | 8 |
| Cirrhotic patients | N/A | 7 |
```
