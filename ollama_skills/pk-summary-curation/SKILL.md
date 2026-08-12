---
name: pk-summary-curation
description: Curate aggregate/summary pharmacokinetics (PK) tables (mean / median / SD / range across a cohort) from a biomedical paper into a normalized 19-column dataset. Use for summary PK tables; for per-subject tables use pk-individual-curation.
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PK Summary Curation

## Inputs you need
1. **One or more table bodies** — each an HTML `<table>...</table>` or a markdown
   table. The input may contain several tables, only some of which are PK
   summary tables; the skill selects the PK ones in Stage 0 (see below).
2. **Caption + footnotes** — the original caption and any footnote text *for each
   table*. Many units, abbreviations, and cohort labels are only resolvable from
   footnotes.
3. **Paper title** — optional but strongly recommended; used as a fallback to
   infer drug / analyte when a table is ambiguous, and shared across all tables.

**If the paper was already prepared** by `pk-pe-prepare`, do not ask for a paste:
read the inputs from `./.paper_assets/<pmid>/` (rooted at `$SKILL_SCRATCH_FOLDER`
when that variable is set). Each `table_<n>.md` holds that table's caption,
footnotes, **and the full table as Markdown**; `table_<n>.html` is the same table
as HTML. `manifest.json` lists the tables with their row/column
counts, and the paper title is its `title` field (also the H1 of
`paper_text.md`).

If the user only supplies a PMID or a URL, ask them to paste the table HTML and
captions — this skill does not fetch papers.

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
PMID if known, otherwise any stable label the user gives). **Create it in the
user's current project/working directory — NOT inside this skill's folder.**
Concretely, the path is `./.pk_curation_scratch/<pmid>/` relative to where the
user is working (the current working directory), so outputs live alongside the
user's data. Never write scratch files under `pipelines/pk-summary-curation/` (the
skill folder is read-only skill content; writing there pollutes the skill and
may be lost or shared across unrelated runs). If you are unsure of the working
directory, run `pwd` and create the scratch folder there.

Because the input can hold several tables, **each selected PK table gets its own
sub-directory** (`table_<n>/`) holding the full 01–14 run for that table; the
cross-table files (all-tables listing, selection, combined result) live at the
top level:

```
.pk_curation_scratch/<pmid>/
├── 00_all_tables.md         # Stage 0a: every input table in markdown, labeled table_1..N
├── 00_selection.md          # Stage 0b: selected table labels + reasoning (omitted if only one table)
├── table_1/                 # one sub-directory PER SELECTED table
│   ├── 00_markdown_table.md # that table's source in markdown
│   ├── inputs.md            # that table's caption + footnotes + (shared) paper title, verbatim
│   ├── 01_drug_table.md     # Stage 1 output
│   ├── 02_patient_table.md  # Stage 2 output
│   ├── 03_patient_refined.md  # Stage 3 output
│   ├── 04_summary_only.md   # Stage 4 output
│   ├── 05_param_aligned.md  # Stage 5 output
│   ├── 06_header_categories.md  # Stage 6 output
│   ├── 07_subtables.md      # Stage 7 output (the list of per-parameter tables)
│   ├── 08_type_unit.md      # Stage 8 output
│   ├── 09_drug_matched.md   # Stage 9 output
│   ├── 10_patient_matched.md  # Stage 10 output
│   ├── 11_param_values.md   # Stage 11 output
│   ├── 12_time.md           # Stage 12 output
│   ├── 13_final.csv         # Stage 13 output (the 19-column result; corrected in place by stage 14)
│   └── 14_verification_report.md  # Stage 14 output (what was checked / fixed / unresolved)
├── table_3/                 # … another selected table, same layout
└── combined_final.csv       # vertical concatenation of every table_*/13_final.csv (the deliverable)
```

For each selected table, write that table's caption, footnotes, and the (shared)
paper title to its `table_<n>/inputs.md` once, up front, so every stage can read
them back verbatim rather than from a possibly-summarized conversation.

**Always nest, even for a single table** — one input table still gets its own
`table_<n>/` sub-directory (`table_1/` if it is the only table) and runs the
01–14 stages there, never directly in the run root. Keeping one code path
removes a branch the model can get wrong; the only cost is one extra directory
level in the common case. `combined_final.csv` is then simply a copy of that
single table's `13_final.csv`.

## Procedure
The top level is a loop over tables:

1. **Stage 0a** — convert every input table to markdown and list them in
   `00_all_tables.md`.
2. **Stage 0b** — if there is more than one table, select the PK summary tables
   (writing `00_selection.md`). With a single table, skip selection.
3. **For each selected PK table**, in its own `table_<n>/` sub-directory, run
   **stages 1–14 in order** (the per-table pipeline described below).
4. **Combine** — vertically concatenate every table's `13_final.csv` into
   `combined_final.csv`, the final deliverable.

Within the per-table pipeline, run the stages below **in order**. Each stage has
a dedicated prompt file in `prompts/`. For each stage:
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

### Stage 0a — Convert all input tables (not a prompt file)
**If the paper was prepared, the conversion is already done.** Each
`./.paper_assets/<pmid>/table_<n>.md` holds the caption, the footnotes, and the
table as Markdown under a `**Table:**` heading. Take that Markdown block as the
table, and the caption/footnotes above it for `inputs.md`. Do **not** re-convert
`table_<n>.html`: `prepare_paper.py` produced that block by running the very
script below on that very HTML, so re-running it can only reproduce the same
bytes or introduce a discrepancy. This path needs no `beautifulsoup4`.

Otherwise — the user pasted raw HTML — convert each input table with the bundled
script; **do not** parse the HTML by hand:

```
python scripts/html_to_markdown_table.py <path-to-html>   # or pipe HTML via stdin
```

The script converts **one `<table>` per invocation**, so run it once per input
table. It deterministically handles colspan/rowspan, multi-row headers, and
empty/duplicate columns; doing this by eye is error-prone and Stage 0 feeds
every downstream stage, so a parsing slip here corrupts the whole run. It needs
`beautifulsoup4` (see `scripts/requirements.txt`). It is
shared across the curation skills (see `this skill`). Only fall
back to converting
inline if the script cannot run (e.g. `beautifulsoup4` is unavailable and
cannot be installed) — and if you do, note that to the user. Skip the conversion
for any table the user already supplied as markdown.

Write all converted tables to `00_all_tables.md`, each preceded by a unique
label and its caption/footnote, like:

```
## table_1
Caption: <caption + footnote text, verbatim>

<markdown table>

## table_2
Caption: …

<markdown table>
```

### Stage 0b — Select the PK summary tables (`prompts/00b_select_pk_tables.md`)
- If `00_all_tables.md` holds **exactly one** table, skip the selection
  *judgment*: that table is the one to curate. (You still curate it in its own
  `table_<n>/` sub-directory — see "Always nest" above.)
- Otherwise, follow `prompts/00b_select_pk_tables.md` to choose the PK summary
  tables among the input, and write the chosen labels + reasoning to
  `00_selection.md`. If none qualify, say so and stop — there is nothing to
  curate.

### Per-table setup
For each selected table, create its `table_<n>/` sub-directory, write that
table's markdown to `table_<n>/00_markdown_table.md`, and write its caption,
footnotes, and the shared paper title to `table_<n>/inputs.md`. The numbered
stages below all operate on these files, and each prompt file's number matches
its stage number.

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
   `07_subtables.md` (the list of per-parameter sub-tables). Adds a leading
   **`Row`** column to each sub-table — the join key that stages 8–12 carry
   through and stage 13 joins on (so stages re-align by key, not by position).

8. **Type + unit extract** — `prompts/08_type_unit_extract.md`
   Reads `07_subtables.md` → writes `08_type_unit.md`.
   For each sub-table, emit `(Row, Parameter type, Parameter unit)`.

9. **Drug matching** — reads `07_subtables.md` + `01_drug_table.md` → writes
   `09_drug_matched.md`.
   - **Shortcut**: if `01_drug_table.md` has exactly one row, assign that
     `[Drug, Analyte, Specimen]` to every data row directly (written as a single
     broadcast row, no `Row` key) — no reasoning needed, the assignment is
     unambiguous.
   - Otherwise: follow `prompts/09_drug_matching.md` to match each row to a
     drug combination, carrying the `Row` key — output
     `(Row, Drug name, Analyte, Specimen)`.

10. **Patient matching** — reads `07_subtables.md` + `03_patient_refined.md` →
    writes `10_patient_matched.md`.
    - **Shortcut**: if `03_patient_refined.md` has exactly one row, assign that
      cohort to every data row directly (single broadcast row, no `Row` key).
    - Otherwise: follow `prompts/10_patient_matching.md`, carrying the `Row` key
      — output `(Row, Population, Pregnancy stage, Pediatric/Gestational age,
      Subject N)`.

11. **Parameter value** — `prompts/11_param_value.md`
    Reads `07_subtables.md` + `05_param_aligned.md` + `inputs.md` → writes
    `11_param_values.md`. For each sub-table emit the `Row` key plus the eight
    value fields: `(Row, Main value, Statistics type, Variation type, Variation
    value, Interval type, Lower bound, Upper bound, P value)`.

12. **Time extraction** — `prompts/12_time_extraction.md`
    Reads `07_subtables.md` → writes `12_time.md`.
    Emit `(Row, Time value, Time unit)` per row. Many rows will be `(N/A, N/A)`.

13. **Assembly + row cleanup** — `prompts/13_assembly_and_cleanup.md`
    Reads `08_type_unit.md`, `09_drug_matched.md`, `10_patient_matched.md`,
    `11_param_values.md`, `12_time.md` → writes `13_final.csv`.
    Join the per-stage tables **on the `Row` key** into the 19-column schema
    above (verifying the key sets agree), drop the `Row` helper column, drop
    fully-empty rows, and rename internal columns:
    - `Main value` → `Parameter value`
    - `Statistics type` → `Parameter statistic`

14. **Verification + correction** — `prompts/14_verify_and_correct.md`
    Reads `13_final.csv` + `00_markdown_table.md` + `inputs.md` → corrects
    `13_final.csv` in place and writes `14_verification_report.md`.
    Runs the deterministic provenance script + an adversarial semantic review,
    then a **bounded** (≤2 round) correction loop. This is the quality gate;
    see "Verification + correction" below.

### Combine (after every selected table finishes stage 14)
Vertically concatenate each `table_*/13_final.csv` into `combined_final.csv` at
the run root. All per-table CSVs share the identical 19-column schema, so this
is a pure row stack — **no recomputation, no re-curation**:
1. Write the 19-column header once.
2. Append every data row from each `table_*/13_final.csv`, in table order.
3. Drop a row only if it is **byte-identical** to a row already written (an
   exact duplicate across tables); never merge or dedup on partial similarity.

`combined_final.csv` is the deliverable. Present it to the user as a markdown
table, and note which source tables contributed (and any that were excluded in
Stage 0b). With a single input table, `combined_final.csv` is just a copy of
that table's `13_final.csv`.

## Validation
After assembly (stage 13), check the final table yourself against these rules,
row by row, before the verification stage:
- 19 columns, in the order given in "Output schema".
- `Subject N` is a positive integer or `"N/A"`.
- `Parameter value`, `Variation value`, `Lower bound`, `Upper bound`, `P value`,
  and `Time value` are numbers or `"N/A"`.
- `Parameter statistic` is one of the **canonical statistic values** (the same
  list stage 11 emits): `Mean`, `Median`, `Geometric mean`, `Arithmetic mean`,
  `Count`, `N/A`. (An interval like a range is *not* a statistic — it belongs in
  `Interval type` / `Lower bound` / `Upper bound`.)

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 14 (`prompts/14_verify_and_correct.md`) is the quality gate and is **part
of the procedure**, not optional. In summary it:
1. runs `scripts/verify_provenance.py` (with
   `--attribution`) — a deterministic
   check that every numeric value in `13_final.csv` (a) appears in the source
   (catches hallucinated/mistyped numbers) and (b) sits under the source label
   the row best matches (catches values copied to the wrong row/specimen/cohort)
   — all without the model judging itself,
2. does an adversarial semantic pass for what the script cannot prove — the
   right statistic and unit, and attribution the token-overlap heuristic is too
   coarse to confirm,
3. corrects only the flagged rows from the source and re-verifies, looping at
   most **2 rounds**, then stops.

Never silently drop a row to pass verification. Surface every unresolved finding
to the user, and clearly distinguish a verified-clean result from one with
caveats.

## Error handling rules
- **No PK table selected** at stage 0b: do not curate anything. Report that none
  of the input tables met the PK inclusion criteria, list what was provided, and
  stop. (If the user insists a specific table is PK, re-run 0b treating it as
  selected.)
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

## Write out the result (do this last)
When the procedure above finishes, copy its **final deliverable CSV** (the `combined_final.csv` / `NN_final.csv` written by the last stage) to the output location, leaving the scratch copy in place:

```bash
# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory
OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"
cp <final-csv-in-scratch> "$OUT/<pmid>/pk-summary-curation.csv"
```

If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/pk-summary-curation.csv` in the user's current working directory. Then tell the user the exact path you wrote.
