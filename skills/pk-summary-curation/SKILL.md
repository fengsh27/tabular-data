---
name: pk-summary-curation
description: Curate a pharmacokinetics (PK) summary table from a biomedical paper
  into a normalized 19-column dataset. Use when the user provides a single PK
  table (HTML or markdown) reporting summary statistics — mean / median / SD /
  range / CI of parameters such as AUC, Cmax, t½, CL, Vd — together with the
  table caption and (optionally) the paper title, and asks for structured
  extraction. Do NOT use for individual-subject PK tables (use
  pk-individual-curation instead) or for population/demographic PK tables.
---

# PK Summary Curation

## Inputs you need
1. **Table body** — HTML `<table>...</table>` or a markdown table.
2. **Caption + footnotes** — the original caption and any footnote text. Many
   units, abbreviations, and cohort labels are only resolvable from footnotes.
3. **Paper title** — optional but strongly recommended; used as a fallback to
   infer drug / analyte when the table is ambiguous.

If the user only supplies a PMID or a URL, ask them to paste the table HTML and
caption — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 19 columns, in order. This matches
the column-wise assembly in stage 13 (drug → patient → type/unit → values →
time):

| # | Column | Notes |
|---|---|---|
| 1 | Drug name | drug administered in the study |
| 2 | Analyte | substance measured (parent drug, metabolite, affected drug…) |
| 3 | Specimen | plasma, serum, urine, cord blood, … |
| 4 | Population | e.g. "Healthy adults", "Maternal" |
| 5 | Pregnancy stage | "N/A" unless the study is obstetric |
| 6 | Pediatric/Gestational age | age/age-range or pregnancy weeks, only if explicitly stated |
| 7 | Subject N | integer count for the cohort the row describes |
| 8 | Parameter type | refined parameter name (e.g. AUC0-∞, Cmax, t½, CL/F) |
| 9 | Parameter unit | normalized unit (e.g. ng·h/mL, L/h, h) |
| 10 | Parameter value | the central / main numeric value (renamed from "Main value") |
| 11 | Parameter statistic | what value (10) represents: Mean, Median, Geometric mean, … (renamed from "Statistics type") |
| 12 | Variation type | variability measure: SD, CV%, SEM, … |
| 13 | Variation value | the single value of that variation |
| 14 | Interval type | 95% CI, Range, IQR, … |
| 15 | Lower bound | lower end of the interval |
| 16 | Upper bound | upper end of the interval |
| 17 | P value | extracted directly from the source's P-value column |
| 18 | Time value | sampling / observation time, numeric |
| 19 | Time unit | unit of column 18 |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
This pipeline has many stages, each consuming the previous stage's output. Do
**not** rely on keeping those intermediate tables in the conversation alone —
a long run can have its context summarized, which would corrupt the exact
table text later stages depend on. Instead, persist each stage's output to a
file and read it back when the next stage needs it.

At the start of a run, create a scratch directory keyed by the paper (use the
PMID if known, otherwise any stable label the user gives):

```
.pk_curation_scratch/<pmid>/
├── 00_markdown_table.md     # Stage 0 output (the source table in markdown)
├── inputs.md                # caption + footnotes + paper title, verbatim
├── 01_drug_table.md         # Stage 1 output
├── 02_patient_table.md      # Stage 2 output
├── 03_patient_refined.md    # Stage 3 output
├── 04_summary_only.md       # Stage 4 output
├── 05_param_aligned.md      # Stage 5 output
├── 06_header_categories.md  # Stage 6 output
├── 07_subtables.md          # Stage 7 output (the list of per-parameter tables)
├── 08_type_unit.md          # Stage 8 output
├── 09_drug_matched.md       # Stage 9 output
├── 10_patient_matched.md    # Stage 10 output
├── 11_param_values.md       # Stage 11 output
├── 12_time.md               # Stage 12 output
├── 13_final.csv             # Stage 13 output (the 19-column result; corrected in place by stage 14)
└── 14_verification_report.md # Stage 14 output (what was checked / fixed / unresolved)
```

Write the caption, footnotes, and paper title to `inputs.md` once, up front, so
every stage can read them back verbatim rather than from a possibly-summarized
conversation.

## Procedure
Run the stages below **in order**. Each stage has a dedicated prompt file in
`prompts/`. For each stage:
1. **Read** the stage's input file(s) from the scratch directory (the prompt
   file names which ones it needs) and read the stage's prompt file.
2. Follow the prompt file's instructions. Reason explicitly before answering,
   then produce the stage's output in the shape that file specifies.
3. Sanity-check your output against the checks the prompt file lists. If a
   check fails, redo that stage **once**, paying attention to what was wrong,
   then carry the result forward even if imperfect (note any residual issue to
   the user). Do not loop more than once per stage.
4. **Write** the (re-checked) output to that stage's scratch file before moving
   on. Later stages read this file, not your message text — so it must be the
   complete, exact table, not a summary.

### Stage 0 — Preprocessing (not a prompt file)
If the input is HTML, convert it to a markdown table with the bundled script —
**do not** parse the HTML by hand:

```
python scripts/html_to_markdown_table.py <path-to-html>   # or pipe HTML via stdin
```

The script deterministically handles colspan/rowspan, multi-row headers, and
empty/duplicate columns; doing this by eye is error-prone and Stage 0 feeds
every downstream stage, so a parsing slip here corrupts the whole run. It needs
`beautifulsoup4` (see `scripts/requirements.txt`). Only fall back to converting
inline if the script cannot run (e.g. `beautifulsoup4` is unavailable and
cannot be installed) — and if you do, note that to the user.

Skip the conversion entirely if the user already supplied markdown. Write the
markdown table to `00_markdown_table.md`, and write the caption, footnotes, and
paper title to `inputs.md`. The numbered stages below all operate on these
files, and each prompt file's number matches its stage number.

### Stages
Each line below lists the stage's scratch input → output files. Always read the
inputs from disk and write the output to disk (see Procedure, steps 1 and 4).

1. **Drug info** — `prompts/01_drug_info.md`
   Reads `00_markdown_table.md` + `inputs.md` → writes `01_drug_table.md`.
   Produces a drug table of unique `[Drug name, Analyte, Specimen]` rows.

2. **Patient info** — `prompts/02_patient_info.md`
   Reads `00_markdown_table.md` + `inputs.md` → writes `02_patient_table.md`.
   Extract cohort grouping (Population / Pregnancy stage / Subject N).

3. **Patient refine** — `prompts/03_patient_refine.md`
   Reads `02_patient_table.md` → writes `03_patient_refined.md`.
   Split rows that combine multiple cohorts into separate rows.

4. **Individual-data deletion** — `prompts/04_individual_data_del.md`
   Reads `00_markdown_table.md` → writes `04_summary_only.md`.
   If the table mixes summary rows and per-subject rows, drop the per-subject
   rows. This skill curates summary stats only.

5. **Parameter-type alignment** — `prompts/05_param_type_align.md`
   Reads `04_summary_only.md` → writes `05_param_aligned.md`.
   Normalize parameter names to canonical forms (e.g. "T1/2" → "t½").

6. **Header categorize** — `prompts/06_header_categorize.md`
   Reads `05_param_aligned.md` → writes `06_header_categories.md`.
   Classify each column as Patient / Parameter type / Parameter value / Time /
   Other. Used to drive the column split.

7. **Split by columns** — `prompts/07_split_by_col.md`
   Reads `05_param_aligned.md` + `06_header_categories.md` → writes
   `07_subtables.md` (the list of per-parameter sub-tables).

8. **Type + unit extract** — `prompts/08_type_unit_extract.md`
   Reads `07_subtables.md` → writes `08_type_unit.md`.
   For each sub-table, emit `(Parameter type, Parameter unit)`.

9. **Drug matching** — reads `07_subtables.md` + `01_drug_table.md` → writes
   `09_drug_matched.md`.
   - **Shortcut**: if `01_drug_table.md` has exactly one row, assign that
     `[Drug, Analyte, Specimen]` to every data row directly — no reasoning
     needed, the assignment is unambiguous.
   - Otherwise: follow `prompts/09_drug_matching.md` to match each row to a
     drug combination.

10. **Patient matching** — reads `07_subtables.md` + `03_patient_refined.md` →
    writes `10_patient_matched.md`.
    - **Shortcut**: if `03_patient_refined.md` has exactly one row, assign that
      cohort to every data row directly.
    - Otherwise: follow `prompts/10_patient_matching.md`.

11. **Parameter value** — `prompts/11_param_value.md`
    Reads `07_subtables.md` + `05_param_aligned.md` + `inputs.md` → writes
    `11_param_values.md`. For each sub-table emit the eight value fields:
    `(Main value, Statistics type, Variation type, Variation value, Interval
    type, Lower bound, Upper bound, P value)`.

12. **Time extraction** — `prompts/12_time_extraction.md`
    Reads `07_subtables.md` → writes `12_time.md`.
    Emit `(Time value, Time unit)` per row. Many rows will be `(N/A, N/A)`.

13. **Assembly + row cleanup** — `prompts/13_assembly_and_cleanup.md`
    Reads `08_type_unit.md`, `09_drug_matched.md`, `10_patient_matched.md`,
    `11_param_values.md`, `12_time.md` → writes `13_final.csv`.
    Join the per-stage tables column-wise into the 19-column schema above, drop
    fully-empty rows, and rename internal columns:
    - `Main value` → `Parameter value`
    - `Statistics type` → `Parameter statistic`

14. **Verification + correction** — `prompts/14_verify_and_correct.md`
    Reads `13_final.csv` + `00_markdown_table.md` + `inputs.md` → corrects
    `13_final.csv` in place and writes `14_verification_report.md`.
    Runs the deterministic provenance script + an adversarial semantic review,
    then a **bounded** (≤2 round) correction loop. This is the quality gate;
    see "Verification + correction" below.

## Validation
After assembly (stage 13), check the final table yourself against these rules,
row by row, before the verification stage:
- 19 columns, in the order given in "Output schema".
- `Subject N` is a positive integer or `"N/A"`.
- `Parameter value`, `Variation value`, `Lower bound`, `Upper bound`, `P value`,
  and `Time value` are numbers or `"N/A"`.
- `Parameter statistic` is one of: Mean, Median, Geometric mean, Arithmetic
  mean, Range, N/A.

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 14 (`prompts/14_verify_and_correct.md`) is the quality gate and is **part
of the procedure**, not optional. In summary it:
1. runs `scripts/verify_provenance.py` — a deterministic check that every
   numeric value in `13_final.csv` appears in the source (catches hallucinated
   numbers without the model judging itself),
2. does an adversarial semantic pass — is each value attributed to the right
   cohort / analyte / specimen, with the right statistic and unit,
3. corrects only the flagged rows from the source and re-verifies, looping at
   most **2 rounds**, then stops.

Never silently drop a row to pass verification. Surface every unresolved finding
to the user, and clearly distinguish a verified-clean result from one with
caveats.

## Error handling rules
- **No drug information found** at stage 1: record the single combination
  `["N/A", "N/A", "N/A"]` instead of failing, and say so in your reply.
- **Parameter-type alignment yields more types than the table has columns**:
  redo that stage once; if it still disagrees, keep the original alignment and
  note the discrepancy.
- **A per-parameter sub-table ends up with zero data rows**: drop it before
  the parameter-value stage.
- **Conflicting units within one sub-table**: prefer the unit in the column
  header over the one in a footnote, and surface the conflict to the user.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- Its verify → correct loop (stage 14) is **bounded** (≤2 correction rounds);
  it does not iterate indefinitely toward a clean result. Remaining issues are
  reported, not endlessly retried.
- It does not score itself against a gold standard — that's the benchmark
  harness in `benchmark/`, which remains the source of truth for accuracy.
