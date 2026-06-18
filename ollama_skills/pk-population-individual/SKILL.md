---
name: pk-population-individual
description: Extract per-patient population/demographic characteristics from a PK paper's full text into a 9-column dataset. For cohort-level stats use pk-population-summary.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Population Individual Curation

Extracts **each individual patient's demographic/clinical characteristics** from a
paper's running text — age, sex, weight, BMI, comorbidity, etc. — each with its
raw value. This is a **full-text** skill — it reads prose, not a table. Its
sibling `pk-population-summary` aggregates by population group with full
statistics. The two share the demographic-refinement, the row-cleanup script, and
the verify/correct procedure in `this skill`.

## Inputs you need
1. **Full text** — the paper's body text (the per-patient/case descriptions). The
   primary source.
2. **Paper title** — recommended.

There is **no input table**. If the user only supplies a PMID or URL, ask them to
paste the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 9 columns, in order:

| # | Column | Notes |
|---|--------|-------|
| 1 | Patient ID | the individual patient/case identifier |
| 2 | Characteristic | the characteristic (Age, Sex, Weight, BMI, Race, Comorbidity, …) — **not** a PK parameter |
| 3 | Characteristic subcategory | level/option (Male/Female, White/Black, …); `N/A` if none |
| 4 | Characteristic unit | unit of the value (e.g. `year`, `kg`) |
| 5 | Characteristic value | the patient's raw value for that characteristic |
| 6 | Population | canonical group: `Nonpregnant`, `Maternal`, `Pediatric`, `Adults`, … |
| 7 | Pregnancy stage | `N/A` unless obstetric |
| 8 | Pediatric/Gestational age | age/age-range or pregnancy weeks, **only if explicitly stated** |
| 9 | Note | the source sentence/excerpt the row was extracted from (traceability) |

Use `"N/A"` (string) for cells that cannot be filled. There is **no** statistic/
variation/interval block and **no Subject N** — each row is one patient's single
raw value (that block is the summary skill).

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pk_population_individual_scratch/<pmid>/` relative to where the user is
working. Never write scratch files under `pipelines/pk-population-individual/`. If
unsure of the working directory, run `pwd` and create the scratch folder there.

This is a **full-text** skill: no input table, so **no Stage-0 conversion, no
table selection, no `table_<n>/` nesting** — the run is flat:

```
.pk_population_individual_scratch/<pmid>/
├── inputs.md            # the paper title + full text, verbatim (source for every stage)
├── 01_characteristic_info.md  # Stage 1: [Patient ID, Patient characteristic, Characteristic sub-category, Characteristic values, Source text]
├── 02_patient_refined.md      # Stage 2: [Patient ID, Population, Pregnancy stage, Pediatric/Gestational age]
├── 03_characteristic_refined.md # Stage 3: [Patient ID, Main value, Unit]
├── 04_assembled.csv     # Stage 4: the 9-column join (pre-cleanup)
├── 05_final.csv         # Stage 5: cleanup-script output (corrected in place by stage 6)
└── 06_verification_report.md # Stage 6 output
```

Write the title + full text to `inputs.md` once, up front.

## Procedure
Run the stages **in order**. For each stage: read its input file(s) and its
prompt file, reason explicitly, produce the output, sanity-check it (redo **once**
if a check fails, then carry forward noting any residual issue), and **write** the
output to its scratch file before moving on. Stages 1–3 carry the **same row set
in the same order** — column-wise refinements of the stage-1 table — so all three
have an identical row count, which stage 4 relies on.

1. **Characteristic info** — `prompts/01_characteristic_info.md`
   Reads `inputs.md` → writes `01_characteristic_info.md`. Extract every unique
   `[Patient ID, Patient characteristic, Characteristic sub-category,
   Characteristic values, Source text]` combination from the full text.

2. **Patient refine** — `prompts/02_patient_refine.md`
   Reads `01_characteristic_info.md` + `inputs.md` → writes
   `02_patient_refined.md`. Thin wrapper over
   `refine_population.md` with `<KEY-COLUMN>` =
   `Patient ID` → `[Patient ID, Population, Pregnancy stage, Pediatric/Gestational
   age]` row-for-row (demographics inferred from the full text per patient).

3. **Characteristic refine** — `prompts/03_characteristic_refine.md`
   Reads `01_characteristic_info.md` + `inputs.md` → writes
   `03_characteristic_refined.md`. Reduce each row to `[Patient ID, Main value,
   Unit]` (the raw value + its unit — **no** statistic decomposition).

4. **Assembly** — `prompts/04_assembly.md`
   Reads the three stage files → writes `04_assembled.csv`. Positional join + the
   rename into the 9-column schema.

5. **Row cleanup** — `prompts/05_row_cleanup.md`
   Runs `scripts/clean_population_individual_rows.py` on
   `04_assembled.csv` → `05_final.csv`. Deterministic: drops rows whose
   `Characteristic value` is blank or `N/A`.

6. **Verification + correction** — `prompts/06_verify_and_correct.md`
   Reads `05_final.csv` + `inputs.md` → corrects `05_final.csv` in place and
   writes `06_verification_report.md`. The quality gate; see below.

## Validation
After cleanup (stage 5), before verification, check the table row by row:
- 9 columns, `Patient ID` first, in the schema order.
- `Characteristic` is a characteristic, **never** a PK parameter.
- no row has `Characteristic value` == `"N/A"` (cleanup drops those).

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 6 follows `verify_and_correct.md`. Because the source
is **prose, not a table**, run the provenance script in **existence-only** mode
(no `--attribution`). The stage prompt fills in the exact parameters.

## Error handling rules
- **No characteristics found** at stage 1: record a single all-`N/A` row instead
  of failing, and say so.
- **No Patient ID in the text**: a single-patient case report → assign `1` to
  every row; multiple cases → use the text's own case numbers. If individuals
  genuinely cannot be told apart, this is population-level data — say so and
  suggest `pk-population-summary`.
- **A refine stage returns a different row count than stage 1**: redo it once,
  insisting on one output row per stage-1 row, same order; if it still disagrees,
  align by the carried `Patient ID` and note the discrepancy.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract PK parameter values, nor compute statistics — that is
  `pk-population-summary` / `pk-individual-curation`.
- Its verify → correct loop (stage 6) is **bounded** (≤2 rounds).
- It does not score itself against a gold standard — that's `benchmark/`.
