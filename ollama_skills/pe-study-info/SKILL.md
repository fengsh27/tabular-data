---
name: pe-study-info
description: Extract pharmacoepidemiology (PE) study metadata (design, population, exposure, outcome definitions) from a paper's full text into a 10-column single-row dataset.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

# PE Study Info Curation

Extracts one row of **study-level metadata** from a pharmacoepidemiology paper's
running text: what kind of study it is, its design, where the data came from, who
the population was, the inclusion/exclusion criteria, the drug, the outcomes, and
the subject count. This is a **full-text** skill — it reads prose, not a table.
For the numeric outcome tables of a PE study, use the sibling `pe-study-outcome`.

## Inputs you need
1. **Full text** — the paper's body text (Methods / study-design / population
   sections). The primary source.
2. **Paper title** — recommended.

There is **no input table**. If the user only supplies a PMID or URL, ask them to
paste the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 10 columns, in order. It is a
**single-row** result — one study per run:

| # | Column | Notes |
|---|--------|-------|
| 1 | Study type | `Pharmacoepidemiology`, `Clinical Trials`, `Pharmacokinetics`, `Pharmacodynamics`, `Pharmacogenetics` |
| 2 | Population | canonical group: `Nonpregnant`, `Maternal`, `Pediatric`, `Adults`, … |
| 3 | Study design | e.g. `Prospective cohort study`, `Randomized controlled trial`, `Case-control study` (multiple → one string, same order) |
| 4 | Pregnancy stage | `N/A` unless obstetric |
| 5 | Drug name | the drug(s) of interest relating to the outcomes |
| 6 | Data source | primary site(s)/database(s) where data was collected (hospital, registry, geography) |
| 7 | Inclusion criteria | exact wording from the article |
| 8 | Exclusion criteria | exact wording from the article |
| 9 | Outcomes | the key outcome(s) of interest, exact wording |
| 10 | Subject N | number of subjects in the population |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pe_study_info_scratch/<pmid>/` relative to where the user is working. Never
write scratch files under `pipelines/pe-study-info/`. If unsure of the working
directory, run `pwd` and create the scratch folder there.

This is a **full-text** skill: no input table, so **no Stage-0 conversion, no
table selection, no `table_<n>/` nesting** — the run is flat:

```
.pe_study_info_scratch/<pmid>/
├── inputs.md            # the paper title + full text, verbatim (source for every stage)
├── 01_design_info.md    # Stage 1: [Study type, Study design, Data source] (one row)
├── 02_design_refined.md # Stage 2: [Population, Inclusion criteria, Exclusion criteria, Pregnancy stage, Subject N, Drug name, Outcomes] (one row)
├── 03_final.csv         # Stage 3: the assembled 10-column single row (corrected in place by stage 4)
└── 04_verification_report.md # Stage 4 output
```

Write the title + full text to `inputs.md` once, up front.

## Procedure
Run the stages **in order**. For each stage: read its input file(s) and its
prompt file, reason explicitly, produce the output, sanity-check it (redo **once**
if a check fails, then carry forward noting any residual issue), and **write** the
output to its scratch file before moving on. Stages 1 and 2 each produce a
**single row** (this skill curates one study).

1. **Design info** — `prompts/01_design_info.md`
   Reads `inputs.md` → writes `01_design_info.md`. Extract the one-row
   `[Study type, Study design, Data source]`.

2. **Design refine** — `prompts/02_design_refine.md`
   Reads `01_design_info.md` + `inputs.md` → writes `02_design_refined.md`.
   Extract the complementary one-row `[Population, Inclusion criteria, Exclusion
   criteria, Pregnancy stage, Subject N, Drug name, Outcomes]`.

3. **Assembly** — `prompts/03_assembly.md`
   Reads both stage files → writes `03_final.csv`. Join the two one-row tables and
   reorder into the 10-column schema.

4. **Verification + correction** — `prompts/04_verify_and_correct.md`
   Reads `03_final.csv` + `inputs.md` → corrects `03_final.csv` in place and
   writes `04_verification_report.md`. The quality gate; see below.

## Validation
After assembly (stage 3), before verification, check the row:
- 10 columns, in the order given in "Output schema".
- exactly **one** data row.
- `Subject N` is a positive integer or `"N/A"`.
- `Inclusion criteria`, `Exclusion criteria`, `Outcomes` use the article's exact
  wording (not paraphrased).

If the row fails, fix it inline and re-check.

## Verification + correction
Stage 4 follows `verify_and_correct.md`. Because the source
is **prose, not a table**, run the provenance script in **existence-only** mode
(no `--attribution`). Note most columns here are free text; the only numeric value
to provenance-check is `Subject N`. The stage prompt fills in the parameters.

## Error handling rules
- **A field is not stated** in the article: use `"N/A"` rather than inventing it.
- **Multiple study designs** are described: list them as one string in the order
  the article presents them (do not split into multiple rows).
- **The paper describes several distinct studies**: this skill curates one study
  per run; curate the primary study and note the others to the user.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract numeric outcome data — that is `pe-study-outcome`.
- Its verify → correct loop (stage 4) is **bounded** (≤2 rounds).
- It does not score itself against a gold standard — that's `benchmark/`.
