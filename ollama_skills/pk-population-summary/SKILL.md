---
name: pk-population-summary
description: Extract summary population/demographic characteristics with statistics from a PK paper's full text into a 15-column dataset. For per-patient characteristics use pk-population-individual.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Population Summary Curation

Extracts the **study population's demographic/clinical characteristics per group**
from a paper's running text — age, sex, weight, BMI, comorbidity, etc. — each with
its summary statistic (mean/median/count), variation, and interval. This is a
**full-text** skill — it reads prose, not a table. Its sibling
`pk-population-individual` does the same per individual patient. The two share the
demographic-refinement and verify/correct procedures in `this skill`.

## Inputs you need
1. **Full text** — the paper's body text (the demographics / "Table 1"-style
   population description in prose). The primary source.
2. **Paper title** — recommended.

**If the paper was already prepared** by `pk-pe-prepare`, do not ask for a paste:
read the inputs from `./.paper_assets/<pmid>/` (rooted at `$SKILL_SCRATCH_FOLDER`
when that variable is set) — `paper_text.md` for the full text, `abstract.md` if
you want the abstract, and the paper title from that file's H1 or from
`manifest.json`.

There is **no input table**. If the user only supplies a PMID or URL, ask them to
paste the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 15 columns, in order:

| # | Column | Notes |
|---|--------|-------|
| 1 | Characteristic | the population characteristic (Age, Sex, Weight, BMI, Race, Comorbidity, …) — **not** a PK parameter |
| 2 | Characteristic subcategory | level/option under it (Male/Female, White/Black, Mild/Severe, …); `N/A` if none |
| 3 | Characteristic unit | unit of the value (e.g. `year`, `kg`, `kg/m²`) |
| 4 | Characteristic value | the primary value (mean/median/count, or a raw ratio like `4/5/4/3`) |
| 5 | Statistics type | how the value is summarized: `Mean`, `Median`, `Count`, … (**required**) |
| 6 | Variation type | variability measure: `SD`, `Proportion (%)`, … |
| 7 | Variation value | the single value of that variation |
| 8 | Interval type | `Minmax`, `IQR`, … |
| 9 | Lower bound | lower end of the interval |
| 10 | Upper bound | upper end of the interval |
| 11 | Population | canonical group: `Nonpregnant`, `Maternal`, `Pediatric`, `Adults`, … |
| 12 | Pregnancy stage | `N/A` unless obstetric |
| 13 | Pediatric/Gestational age | age/age-range or pregnancy weeks, **only if explicitly stated** |
| 14 | Subject N | number of individuals in that population group |
| 15 | Note | the source sentence/excerpt the row was extracted from (traceability) |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pk_population_summary_scratch/<pmid>/` relative to where the user is working.
Never write scratch files under `pipelines/pk-population-summary/`. If unsure of the
working directory, run `pwd` and create the scratch folder there.

This is a **full-text** skill: no input table, so **no Stage-0 conversion, no
table selection, no `table_<n>/` nesting** — the run is flat:

```
.pk_population_summary_scratch/<pmid>/
├── inputs.md            # the paper title + full text, verbatim (source for every stage)
├── 01_characteristic_info.md  # Stage 1: [Population characteristic, Characteristic sub-category, Characteristic values, Population, Population N, Source text]
├── 02_patient_refined.md      # Stage 2: [Population, Pregnancy stage, Pediatric/Gestational age, Population N]
├── 03_characteristic_refined.md # Stage 3: [Main value, Unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound]
├── 04_final.csv         # Stage 4: the assembled 15-column result (corrected in place by stage 5)
└── 05_verification_report.md # Stage 5 output
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
   `[Population characteristic, Characteristic sub-category, Characteristic
   values, Population, Population N, Source text]` combination from the full text.

2. **Patient refine** — `prompts/02_patient_refine.md`
   Reads `01_characteristic_info.md` + `inputs.md` → writes
   `02_patient_refined.md`. Thin wrapper over
   `refine_population.md` with `<KEY-COLUMN>` =
   `Population N` → `[Population, Pregnancy stage, Pediatric/Gestational age,
   Population N]` row-for-row.

3. **Characteristic refine** — `prompts/03_characteristic_refine.md`
   Reads `01_characteristic_info.md` + `inputs.md` → writes
   `03_characteristic_refined.md`. Decompose the raw `Characteristic values` into
   `[Main value, Unit, Statistics type, Variation type, Variation value, Interval
   type, Lower bound, Upper bound]` row-for-row.

4. **Assembly** — `prompts/04_assembly.md`
   Reads the three stage files → writes `04_final.csv`. Positional join +
   the rename into the 15-column final schema.

5. **Verification + correction** — `prompts/05_verify_and_correct.md`
   Reads `04_final.csv` + `inputs.md` → corrects `04_final.csv` in place and
   writes `05_verification_report.md`. The quality gate; see below.

(There is **no separate row-cleanup stage** — the legacy population-summary
cleanup is a no-op, so the assembled table is already final.)

## Validation
After assembly (stage 4), before verification, check the table row by row:
- 15 columns, in the order given in "Output schema".
- `Characteristic` is a population characteristic, **never** a PK parameter.
- `Statistics type` is filled (it is required) — `Mean`, `Median`, `Count`, etc.
- `Subject N` is a positive integer or `"N/A"`; `Characteristic value`,
  `Variation value`, `Lower bound`, `Upper bound` are numbers, a raw ratio, or
  `"N/A"`.

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 5 follows `verify_and_correct.md`. Because the source
is **prose, not a table**, run the provenance script in **existence-only** mode
(no `--attribution`). The stage prompt fills in the exact parameters.

## Error handling rules
- **No population characteristics found** at stage 1: record a single all-`N/A`
  row instead of failing, and say so.
- **A summed total slipped in** at stage 1: exclude totals when the individual
  parts are reported (per the stage-1 prompt).
- **A refine stage returns a different row count than stage 1**: redo it once,
  insisting on one output row per stage-1 row in the same order; if it still
  disagrees, align by the carried `Population N` and note the discrepancy.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract PK parameter values — characteristics only.
- Its verify → correct loop (stage 5) is **bounded** (≤2 rounds).
- It does not score itself against a gold standard — that's `benchmark/`.

## Write out the result (do this last)
When the procedure above finishes, copy its **final deliverable CSV** (the `combined_final.csv` / `NN_final.csv` written by the last stage) to the output location, leaving the scratch copy in place:

```bash
# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory
OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"
cp <final-csv-in-scratch> "$OUT/<pmid>/pk-population-summary.csv"
```

If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/pk-population-summary.csv` in the user's current working directory. Then tell the user the exact path you wrote.
