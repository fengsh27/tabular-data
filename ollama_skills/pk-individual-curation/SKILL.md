---
name: pk-individual-curation
description: Curate individual-subject pharmacokinetics (PK) tables (one row per subject per parameter) from a biomedical paper into a normalized 12-column dataset. Use when tables report per-individual PK values; do NOT use for summary/aggregate tables (use pk-summary-curation).
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Individual Curation

Curates **individual-level** PK tables — where each row is one subject's own
measurement — into a normalized per-subject dataset. The sibling
`pk-summary-curation` skill handles aggregate (mean/median/SD/range) tables; the
two share the converter, the provenance checker, and the verify/correct
procedure in `this skill`.

## Inputs you need
1. **One or more table bodies** — each an HTML `<table>...</table>` or a markdown
   table. The input may contain several tables, only some of which are PK
   tables; the skill selects them in Stage 0 (see below).
2. **Caption + footnotes** — the original caption and footnote text *for each
   table*.
3. **Paper title** — optional but recommended; shared across all tables.
4. **Full text** — the paper's body text (or at least the sections describing
   the subjects). **Often essential for individual data**: many individual PK
   tables — especially single-patient case reports — keep the subject's identity
   in the prose, not the table. Stage 0c uses the full text to infer and inject a
   `Patient ID`; without it, such a table has no Patient ID and is discarded at
   stage 2. Ask for it when the table lacks an obvious per-row subject id.

If the user only supplies a PMID or a URL, ask them to paste the table HTML,
captions, and the full text — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 12 columns, in order:

| # | Column | Notes |
|---|---|---|
| 1 | Patient ID | the individual subject identifier (explicit or an inferred unique unit) |
| 2 | Drug name | drug administered |
| 3 | Analyte | substance measured (parent drug, metabolite, …) |
| 4 | Specimen | plasma, serum, cord blood, milk, … |
| 5 | Population | e.g. "Maternal", "Neonates" |
| 6 | Pregnancy stage | "N/A" unless obstetric |
| 7 | Pediatric/Gestational age | age/age-range or pregnancy weeks, only if explicitly stated |
| 8 | Parameter type | refined parameter name (e.g. Cmax, AUC, a specimen concentration) |
| 9 | Parameter unit | normalized unit (e.g. ng/mL, ng·h/mL, h) |
| 10 | Parameter value | the individual subject's single measured value |
| 11 | Time value | sampling / observation time, numeric |
| 12 | Time unit | unit of column 11 |

Use `"N/A"` (string) for cells that cannot be filled. There is **no** Subject N
and **no** statistic/variation/interval/P-value block — each row is one subject's
single raw value, not an aggregate.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized, which
would corrupt the exact table text later stages depend on).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pk_individual_scratch/<pmid>/` relative to where the user is working (the
current working directory), so outputs live alongside the user's data. Never
write scratch files under `pipelines/pk-individual-curation/` (the skill folder is
read-only skill content; writing there pollutes the skill and may be lost or
shared across unrelated runs). If you are unsure of the working directory, run
`pwd` and create the scratch folder there.

Because the input can hold several tables, **each selected PK table gets its own
sub-directory** (`table_<n>/`) holding the full 01–14 run; the cross-table files
live at the top level:

```
.pk_individual_scratch/<pmid>/
├── full_text.md             # the paper's full text, verbatim (shared by all tables; for Stage 0c)
├── 00_all_tables.md         # Stage 0a: every input table in markdown, labeled table_1..N
├── 00_selection.md          # Stage 0b: selected table labels + reasoning (omitted if only one table)
├── table_1/                 # one sub-directory PER SELECTED table
│   ├── 00_markdown_table.md # that table's source in markdown (Patient ID column injected by Stage 0c if missing)
│   ├── 00c_patient_id.md    # Stage 0c note: already_present / inferred / needs_full_text
│   ├── inputs.md            # that table's caption + footnotes + (shared) paper title, verbatim
│   ├── 01_drug_table.md     # Stage 1
│   ├── 02_patient_table.md  # Stage 2  [Patient ID, Population, Pregnancy stage]
│   ├── 03_patient_refined.md  # Stage 3  (+ Pediatric/Gestational age)
│   ├── 04_individual_only.md  # Stage 4  (summary rows removed)
│   ├── 05_param_aligned.md  # Stage 5
│   ├── 06_header_categories.md  # Stage 6
│   ├── 07_subtables.md      # Stage 7  (one sub-table per parameter column, with Row key)
│   ├── 08_type_unit_value.md  # Stage 8  [Row, Parameter type, Parameter unit, Parameter value]
│   ├── 09_drug_matched.md   # Stage 9
│   ├── 10_patient_matched.md  # Stage 10 (by Patient ID)
│   ├── 11_time.md           # Stage 11
│   ├── 12_assembled.csv     # Stage 12 (12-col join, pre-cleanup)
│   ├── 13_final.csv         # Stage 13 (cleanup script output; corrected in place by stage 14)
│   └── 14_verification_report.md  # Stage 14
├── table_3/                 # … another selected table, same layout
└── combined_final.csv       # vertical concatenation of every table_*/13_final.csv (the deliverable)
```

**Always nest, even for a single table** — one input table still gets its own
`table_<n>/` (`table_1/`) and runs the stages there, never directly in the run
root. One code path, fewer mistakes. `combined_final.csv` is then a copy of that
table's `13_final.csv`.

## Procedure
The top level is a loop over tables:

0. **Up front** — if the user supplied the paper's full text, write it verbatim
   to `full_text.md` at the run root (Stage 0c reads it to infer Patient IDs).
1. **Stage 0a** — convert every input table to markdown → `00_all_tables.md`.
2. **Stage 0b** — if more than one table, select the PK tables → `00_selection.md`.
   With a single table, skip the selection judgment (still nest it in `table_1/`).
3. **For each selected PK table**, in its own `table_<n>/`: run **Stage 0c**
   (ensure a Patient ID), then **stages 1–14 in order** (below).
4. **Combine** — vertically concatenate every table's `13_final.csv` into
   `combined_final.csv`.

For each per-table stage: read the stage's input file(s) and its prompt file,
reason explicitly, produce the output, sanity-check it (redo **once** if a check
fails, then carry forward noting any residual issue), and **write** the output to
its scratch file before moving on.

### Stage 0a — Convert all input tables (not a prompt file)
For each HTML table, convert it with the shared script (one `<table>` per
invocation) — do **not** parse HTML by hand:

```
python scripts/html_to_markdown_table.py <path-to-html>
```

It needs `beautifulsoup4` (see `scripts/requirements.txt`).
Skip conversion for tables already supplied as markdown. Write all converted
tables to `00_all_tables.md`, each preceded by a `## table_N` label and its
caption/footnote.

### Stage 0b — Select the PK tables (`prompts/00b_select_pk_tables.md`)
- If `00_all_tables.md` holds exactly one table, skip the selection judgment —
  curate that table (still in its own `table_<n>/`).
- Otherwise follow `prompts/00b_select_pk_tables.md`, writing chosen labels +
  reasoning to `00_selection.md`. If none qualify, say so and stop.

### Per-table setup
For each selected table, create `table_<n>/`, write its markdown to
`table_<n>/00_markdown_table.md`, and its caption/footnotes + shared title to
`table_<n>/inputs.md`. Each prompt file's number matches its stage number.

### Stage 0c — Ensure a Patient ID (`prompts/00c_infer_patient_id.md`)
Run this **before stage 1**. If the table has no per-row subject identifier,
infer one from `full_text.md` + caption (single patient → `1` for every row;
multiple cases → `1, 2, …`) and **inject a `Patient ID` column** into
`00_markdown_table.md`; if the table already identifies subjects, leave it.
Records its verdict in `00c_patient_id.md`. This is what lets single-patient
case-report tables survive — without it stage 2 finds no Patient ID and the
table is discarded.

### Stages
1. **Drug info** — `prompts/01_drug_info.md`
   Reads `00_markdown_table.md` + `inputs.md` → `01_drug_table.md`.
   Unique `[Drug name, Analyte, Specimen]` rows.

2. **Patient info** — `prompts/02_patient_info.md`
   Reads `00_markdown_table.md` (Patient ID present after Stage 0c) + `inputs.md`
   → `02_patient_table.md`. Unique `[Patient ID, Population, Pregnancy stage]`
   rows.

3. **Patient refine** — `prompts/03_patient_refine.md`
   Reads `02_patient_table.md` (+ source) → `03_patient_refined.md`.
   Normalizes Population / Pregnancy stage and adds Pediatric/Gestational age.

4. **Summary-data deletion** — `prompts/04_summary_data_del.md`
   Reads `00_markdown_table.md` → `04_individual_only.md`.
   Keeps per-subject rows, drops aggregate (N / Mean / Median / Range) rows.

5. **Parameter-type alignment** — `prompts/05_param_type_align.md`
   Reads `04_individual_only.md` → `05_param_aligned.md`.
   Orients to one subject per row with parameter types as headers (transpose
   condition is the **inverse** of pk-summary).

6. **Header categorize** — `prompts/06_header_categorize.md`
   Reads `05_param_aligned.md` → `06_header_categories.md`.
   Three categories only: `Patient ID` / `Parameter value` / `Uncategorized`.

7. **Split by columns** — `prompts/07_split_by_col.md`
   Reads `05_param_aligned.md` + `06_header_categories.md` → `07_subtables.md`.
   One sub-table per `Parameter value` column: `[Row, Patient ID, Parameter
   type, Parameter value]`, the header becoming `Parameter type`. Adds the
   **`Row`** join key carried through stages 8–11.

8. **Type + unit + value extract** — `prompts/08_type_unit_value_extract.md`
   Reads `07_subtables.md` (+ context) → `08_type_unit_value.md`.
   Per row: `(Row, Parameter type, Parameter unit, Parameter value)` — the value
   copied verbatim (no statistic decomposition).

9. **Drug matching** — reads `07_subtables.md` + `01_drug_table.md` →
   `09_drug_matched.md`.
   - **Shortcut**: if `01_drug_table.md` has one row, broadcast it to every row
     (single row, no `Row` key).
   - Otherwise follow `prompts/09_drug_matching.md` → `(Row, Drug name, Analyte,
     Specimen)`.

10. **Patient matching** — reads `07_subtables.md` + `03_patient_refined.md` →
    `10_patient_matched.md`.
    - **Shortcut**: if `03_patient_refined.md` has one cohort, attach it (with
      each row's own Patient ID) to every row.
    - Otherwise follow `prompts/10_patient_matching.md` (lookup by Patient ID) →
      `(Row, Patient ID, Population, Pregnancy stage, Pediatric/Gestational age)`.

11. **Time extraction** — `prompts/11_time_extraction.md`
    Reads `07_subtables.md` (+ context) → `11_time.md`.
    `(Row, Time value, Time unit)` per row; many rows are `(N/A, N/A)`.

12. **Assembly** — `prompts/12_assembly.md`
    Reads `08_type_unit_value.md`, `09_drug_matched.md`, `10_patient_matched.md`,
    `11_time.md` → `12_assembled.csv`.
    Joins **on the `Row` key** into the 12-column schema, drops the `Row` helper.

13. **Row cleanup** — `prompts/13_row_cleanup.md`
    Runs `scripts/clean_individual_rows.py` on `12_assembled.csv` →
    `13_final.csv` (time normalization, drop N/A-value rows, dedupe, Patient ID
    first). Deterministic.

14. **Verification + correction** — `prompts/14_verify_and_correct.md`
    Follows `verify_and_correct.md` against the source:
    deterministic provenance + attribution check, adversarial semantic review,
    bounded ≤2-round correction. Corrects `13_final.csv` in place and writes
    `14_verification_report.md`.

### Combine (after every selected table finishes stage 14)
Vertically concatenate each `table_*/13_final.csv` into `combined_final.csv` at
the run root — a pure row stack (identical 12-column schema), **no
recomputation**. Drop a row only if byte-identical to one already written.
Present `combined_final.csv` to the user and note which tables contributed (and
any excluded in 0b). With a single input table it is a copy of that table's
`13_final.csv`.

## Validation
After cleanup (stage 13), before verification, check the table:
- 12 columns, `Patient ID` first, in the schema order;
- `Parameter value` and `Time value` are numbers or `"N/A"`;
- no row has `Parameter value` == `"N/A"` (cleanup drops those);
- no duplicate rows.

## Error handling rules
- **No PK table selected** at stage 0b: report that none of the input tables met
  the criteria, list what was provided, and stop.
- **No Patient ID in the table**: this is expected for single-patient case
  reports — Stage 0c should infer and inject one from the full text (single
  patient → all `1`). If `full_text.md` was **not** provided, ask the user for
  it and re-run Stage 0c. Only if, even with the full text, no per-row subject
  can be established (and the caption/title don't imply a single patient) is the
  table not individual data — then say so and stop (consider pk-summary-curation
  instead).
- **No drug information found** at stage 1: record `["N/A","N/A","N/A"]` and say
  so.
- **A sub-table ends up with zero usable rows**: drop it before assembly.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- Its verify → correct loop is **bounded** (≤2 rounds); remaining issues are
  reported, not endlessly retried.
- It does not aggregate or compute statistics — that is pk-summary-curation.
- It does not score itself against a gold standard — that's `benchmark/`.

## Write out the result (do this last)
When the procedure above finishes, copy its **final deliverable CSV** (the `combined_final.csv` / `NN_final.csv` written by the last stage) to the output location, leaving the scratch copy in place:

```bash
# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory
OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"
cp <final-csv-in-scratch> "$OUT/<pmid>/pk-individual-curation.csv"
```

If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/pk-individual-curation.csv` in the user's current working directory. Then tell the user the exact path you wrote.
