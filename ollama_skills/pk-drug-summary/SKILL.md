---
name: pk-drug-summary
description: Extract summary (cohort-level) drug dosing regimens (dose amount / unit / frequency / route) from a PK paper's full text into an 11-column dataset. For per-patient dosing use pk-drug-individual.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Drug Summary Curation

Extracts the **dosing regimen per population group** from a paper's running text:
which drug, how much, how often, by what route, for which cohort. This is a
**full-text** skill — it reads prose, not a table. Its sibling
`pk-drug-individual` does the same per individual patient. The two share the
demographic-refinement and verify/correct procedures in
`this skill`.

## Inputs you need
1. **Full text** — the paper's body text (Methods/Results/dosing paragraphs at
   minimum). This is the primary source; everything is read from it.
2. **Paper title** — recommended; helps disambiguate the drug/population.

There is **no input table**. If the user only supplies a PMID or URL, ask them to
paste the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 11 columns, in order:

| # | Column | Notes |
|---|--------|-------|
| 1 | Drug/Metabolite name | the drug or metabolite administered |
| 2 | Dose amount | numeric value, list, or range (e.g. `5`, `1,2,3,4`, `0.01 - 0.05`) |
| 3 | Dose unit | unit of the dose amount (e.g. `mg`, `mg/kg`) |
| 4 | Dose frequency | how many times taken (e.g. `Single`, `Multiple`, `3`) |
| 5 | Dose schedule | timing/interval (e.g. `once a day`, `every 8 hours`) |
| 6 | Dose route | `Oral`, `IV`, `IM`, `SC`, `Epidural`, `Infusion`, … |
| 7 | Population | canonical group: `Nonpregnant`, `Maternal`, `Pediatric`, `Adults`, … |
| 8 | Pregnancy stage | `N/A` unless obstetric |
| 9 | Pediatric/Gestational age | age/age-range or pregnancy weeks, **only if explicitly stated** |
| 10 | Population N | number of individuals in that population group |
| 11 | Note | the source sentence/excerpt the row was extracted from (traceability) |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized, which
would corrupt the exact table text later stages depend on).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pk_drug_summary_scratch/<pmid>/` relative to where the user is working (the
current working directory), so outputs live alongside the user's data. Never
write scratch files under `pipelines/pk-drug-summary/` (the skill folder is
read-only skill content). If you are unsure of the working directory, run `pwd`
and create the scratch folder there.

This is a **full-text** skill: there is no input table, so there is **no Stage-0
conversion, no table selection, and no `table_<n>/` nesting** — the run is flat:

```
.pk_drug_summary_scratch/<pmid>/
├── inputs.md            # the paper title + full text, verbatim (the source for every stage)
├── 01_drug_info.md      # Stage 1: [Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N, Source text]
├── 02_patient_refined.md # Stage 2: [Population, Pregnancy stage, Pediatric/Gestational age, Population N]
├── 03_drug_refined.md   # Stage 3: [Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose schedule, Dose route]
├── 04_final.csv         # Stage 4: the assembled 11-column result (corrected in place by stage 5)
└── 05_verification_report.md # Stage 5 output
```

Write the title + full text to `inputs.md` once, up front, so every stage reads
the source back verbatim rather than from a possibly-summarized conversation.

## Procedure
Run the stages **in order**. For each stage: read its input file(s) and its
prompt file, reason explicitly, produce the output, sanity-check it (redo **once**
if a check fails, then carry forward noting any residual issue), and **write** the
output to its scratch file before moving on. Stages 1–3 each carry the **same row
set in the same order** — they are column-wise refinements of the stage-1 table,
so all three have an identical row count, which stage 4 relies on.

1. **Drug info** — `prompts/01_drug_info.md`
   Reads `inputs.md` → writes `01_drug_info.md`. Extract every unique
   `[Drug/Metabolite name, Dose frequency, Dose amount, Population, Population N,
   Source text]` combination described in the full text.

2. **Patient refine** — `prompts/02_patient_refine.md`
   Reads `01_drug_info.md` + `inputs.md` → writes `02_patient_refined.md`.
   A thin wrapper over `refine_population.md` with
   `<KEY-COLUMN>` = `Population N`, producing `[Population, Pregnancy stage,
   Pediatric/Gestational age, Population N]` row-for-row.

3. **Drug refine** — `prompts/03_drug_refine.md`
   Reads `01_drug_info.md` + `inputs.md` → writes `03_drug_refined.md`. Split the
   dose into `[Drug/Metabolite name, Dose amount, Dose unit, Dose frequency, Dose
   schedule, Dose route]` row-for-row.

4. **Assembly** — `prompts/04_assembly.md`
   Reads `01_drug_info.md`, `02_patient_refined.md`, `03_drug_refined.md` →
   writes `04_final.csv`. Join the three row-aligned tables into the 11-column
   schema (drug-refine columns + patient columns + the stage-1 `Source text`
   renamed to `Note`).

5. **Verification + correction** — `prompts/05_verify_and_correct.md`
   Reads `04_final.csv` + `inputs.md` → corrects `04_final.csv` in place and
   writes `05_verification_report.md`. The quality gate; see below.

## Validation
After assembly (stage 4), check the table yourself, row by row, before
verification:
- 11 columns, in the order given in "Output schema".
- `Population N` is a positive integer or `"N/A"`.
- `Dose amount` is a number, a comma list, a range, or `"N/A"` — and `Dose unit`
  holds the unit, not the number.
- `Dose route` is one of the recognized routes or `"N/A"`.

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 5 follows `verify_and_correct.md` (the shared
quality gate). Because the source here is **prose, not a table**, run the
provenance script in **existence-only** mode (no `--attribution` — there is no
source table for attribution to match against). The stage prompt fills in the
exact parameters. It never silently drops a row; unresolved findings are reported.

## Error handling rules
- **No drug/dosing information found** at stage 1: record the single combination
  `["N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]` instead of failing, and say so.
- **A refine stage returns a different row count than stage 1**: redo it once,
  insisting on one output row per stage-1 row in the same order; if it still
  disagrees, align by the carried key (`Population N`) and note the discrepancy.
- **Dose amount and unit run together** (e.g. `5mg`): split them in stage 3 —
  `Dose amount` = `5`, `Dose unit` = `mg`.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract PK parameter values (AUC/Cmax/CL) — that is
  `pk-summary-curation` / `pk-individual-curation`.
- Its verify → correct loop (stage 5) is **bounded** (≤2 rounds); remaining
  issues are reported, not endlessly retried.
- It does not score itself against a gold standard — that's `benchmark/`.

## Write out the result (do this last)
When the procedure above finishes, copy its **final deliverable CSV** (the `combined_final.csv` / `NN_final.csv` written by the last stage) to the output location, leaving the scratch copy in place:

```bash
# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory
OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"
cp <final-csv-in-scratch> "$OUT/<pmid>/pk-drug-summary.csv"
```

If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/pk-drug-summary.csv` in the user's current working directory. Then tell the user the exact path you wrote.
