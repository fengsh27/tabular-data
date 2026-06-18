---
name: pk-specimen-summary
description: Extract summary (cohort-level) specimen-sampling information (specimen type, sampling times) from a PK paper's full text into a 9-column dataset. For per-patient sampling use pk-specimen-individual.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Specimen Summary Curation

Extracts the **specimen-sampling design per population group** from a paper's
running text: which specimen, how many samples, sampled when, for which cohort.
This is a **full-text** skill — it reads prose, not a table. Its sibling
`pk-specimen-individual` does the same per individual patient. The two share the
demographic-refinement, the row-cleanup script, and the verify/correct procedure
in `this skill`.

## Inputs you need
1. **Full text** — the paper's body text (Methods/sampling paragraphs at
   minimum). The primary source.
2. **Paper title** — recommended.

There is **no input table**. If the user only supplies a PMID or URL, ask them to
paste the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 9 columns, in order:

| # | Column | Notes |
|---|--------|-------|
| 1 | Specimen | biological sample collected (urine, blood, plasma, cord blood, milk, …) |
| 2 | Sample N | number of samples analyzed for that specimen |
| 3 | Population | canonical group: `Nonpregnant`, `Maternal`, `Pediatric`, `Adults`, … |
| 4 | Pregnancy stage | `N/A` unless obstetric |
| 5 | Pediatric/Gestational age | age/age-range or pregnancy weeks, **only if explicitly stated** |
| 6 | Population N | number of individuals in that population group |
| 7 | Sample time | sampling time(s), **numeric** (e.g. `0`, `24`, `0, 2, 4`, `0-2`) |
| 8 | Time unit | unit of column 7 (`Second`, `Minute`, `Hour`, `Day`) |
| 9 | Note | the source sentence/excerpt the row was extracted from (traceability) |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pk_specimen_summary_scratch/<pmid>/` relative to where the user is working.
Never write scratch files under `pipelines/pk-specimen-summary/`. If unsure of the
working directory, run `pwd` and create the scratch folder there.

This is a **full-text** skill: no input table, so **no Stage-0 conversion, no
table selection, no `table_<n>/` nesting** — the run is flat:

```
.pk_specimen_summary_scratch/<pmid>/
├── inputs.md            # the paper title + full text, verbatim (source for every stage)
├── 01_specimen_info.md  # Stage 1: [Specimen, Sample N, Sample time, Population, Population N]
├── 02_patient_refined.md # Stage 2: [Population, Pregnancy stage, Pediatric/Gestational age, Population N]
├── 03_time.md           # Stage 3: [Sample time, Time unit, Source text]
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
in the same order** — they are column-wise refinements of the stage-1 table — so
all three have an identical row count, which stage 4 relies on.

1. **Specimen info** — `prompts/01_specimen_info.md`
   Reads `inputs.md` → writes `01_specimen_info.md`. Extract every unique
   `[Specimen, Sample N, Sample time, Population, Population N]` combination
   described in the full text. **Exclude summed totals** when the individual
   parts are also reported (see the prompt).

2. **Patient refine** — `prompts/02_patient_refine.md`
   Reads `01_specimen_info.md` + `inputs.md` → writes `02_patient_refined.md`.
   Thin wrapper over `refine_population.md` with
   `<KEY-COLUMN>` = `Population N`, producing `[Population, Pregnancy stage,
   Pediatric/Gestational age, Population N]` row-for-row.

3. **Time extraction** — `prompts/03_time_extraction.md`
   Reads `01_specimen_info.md` + `inputs.md` → writes `03_time.md`. For each
   stage-1 row emit `[Sample time, Time unit, Source text]` — Sample time kept
   strictly **numeric**, comma-lists and ranges preserved as single cells.

4. **Assembly** — `prompts/04_assembly.md`
   Reads `01_specimen_info.md`, `02_patient_refined.md`, `03_time.md` → writes
   `04_assembled.csv`. Positional join into the 9-column schema (`Specimen,
   Sample N` from stage 1; the patient columns from stage 2; `Sample time, Time
   unit` from stage 3; the stage-3 `Source text` renamed to `Note`).

5. **Row cleanup** — `prompts/05_row_cleanup.md`
   Runs `scripts/clean_specimen_rows.py` on
   `04_assembled.csv` → `05_final.csv`. Deterministic: drops a redundant summed-
   total row and collapses duplicate specimen/time/cohort rows to the largest
   `Sample N`.

6. **Verification + correction** — `prompts/06_verify_and_correct.md`
   Reads `05_final.csv` + `inputs.md` → corrects `05_final.csv` in place and
   writes `06_verification_report.md`. The quality gate; see below.

## Validation
After cleanup (stage 5), before verification, check the table row by row:
- 9 columns, in the order given in "Output schema".
- `Sample N` and `Population N` are positive integers or `"N/A"`.
- `Sample time` is numeric / a numeric list / a numeric range / `"N/A"` — never
  prose; `Time unit` holds the unit.

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 6 follows `verify_and_correct.md`. Because the source
is **prose, not a table**, run the provenance script in **existence-only** mode
(no `--attribution`). The stage prompt fills in the exact parameters.

## Error handling rules
- **No specimen/sampling information found** at stage 1: record a single all-`N/A`
  row `["N/A","N/A","N/A","N/A","N/A"]` instead of failing, and say so.
- **A refine/time stage returns a different row count than stage 1**: redo it
  once, insisting on one output row per stage-1 row in the same order; if it
  still disagrees, align by the carried `Population N` and note the discrepancy.
- **A summed total slipped through stage 1**: the cleanup script drops a row whose
  `Sample N` equals exactly half the column total — but prefer to exclude totals
  at stage 1 per its prompt.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract PK parameter values — that is `pk-summary-curation`.
- Its verify → correct loop (stage 6) is **bounded** (≤2 rounds).
- It does not score itself against a gold standard — that's `benchmark/`.
