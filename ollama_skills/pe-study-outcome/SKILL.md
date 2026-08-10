---
name: pe-study-outcome
description: Curate pharmacoepidemiology (PE) study-outcome tables from a paper into a normalized 12-column dataset (effect estimates, confidence intervals, p-values per outcome).
---

> **Self-contained skill.** Every `scripts/…` path and every resource file (`verify_and_correct.md`, `refine_population.md`, …) named below lives in **this skill's own directory**. This skill shares nothing with other skills — when run as an installed skill, resolve these paths under this skill's folder.

> **Working-directory base (read this first).** Every `./.…` path this skill uses below — the prepared inputs in `./.paper_assets/<pmid>/` and any `./.…_scratch/<pmid>/` intermediates this skill writes — is relative to one base directory. Resolve it **once, before any file operation**: if the environment variable `SKILL_SCRATCH_FOLDER` is set (run `echo "$SKILL_SCRATCH_FOLDER"` to check), that is the base — e.g. write to `"$SKILL_SCRATCH_FOLDER"/.<name>_scratch/<pmid>/` and read from `"$SKILL_SCRATCH_FOLDER"/.paper_assets/<pmid>/`. Otherwise the base is the user's current working directory (use the paths exactly as written below). Create directories with `mkdir -p` and keep the same base for every read and write.

# PE Study Outcome Curation

Curates a **pharmacoepidemiology outcome table** — each numeric cell becomes one
row tagged with its `Characteristic` / `Exposure` / `Outcome` context and its
parsed statistic (value, unit, variation, interval, p-value). This is a **table**
skill (it reads a table, like `pk-summary-curation`), implementing the active
`pe_study_outcome_ver2` pipeline: *retain numbers → parse each value → tag its
context → clean*. For PE study-level metadata from prose, use `pe-study-info`.

## Inputs you need
1. **One outcome table** — an HTML `<table>...</table>` or a markdown table of PE
   results.
2. **Caption + footnotes** — the table's caption and footnote text (units,
   group labels, and p-value placement are often only resolvable from these).
3. **Paper title** — optional but recommended.

**If the paper was already prepared** by `pk-pe-prepare`, do not ask for a paste:
read the inputs from `./.paper_assets/<pmid>/` (rooted at `$SKILL_SCRATCH_FOLDER`
when that variable is set). Each `table_<n>.md` holds that table's caption,
footnotes, **and the full table as Markdown**; `table_<n>.html` is the same table
as HTML. `manifest.json` lists the tables with their row/column
counts, and the paper title is its `title` field (also the H1 of
`paper_text.md`).

If the user only supplies a PMID or URL, ask them to paste the table HTML and
caption — this skill does not fetch papers.

## Output schema
A CSV (or markdown table) with exactly these 12 columns, in order:

| # | Column | Notes |
|---|--------|-------|
| 1 | Characteristic | demographic/biological feature of the subjects (age, sex, race, …) |
| 2 | Exposure | factor associated with the outcome (a drug, condition, medication) |
| 3 | Outcome | what the value measures (e.g. birth weight, total sleep time) — never the number itself |
| 4 | Parameter unit | unit of the value (e.g. `kg`, `g`, `Count`) — **not** a statistic like SD |
| 5 | Parameter statistic | `Mean`, `Median`, `Sum`, `Proportion`, `%`, … |
| 6 | Parameter value | the main value (not a range) |
| 7 | Variation type | variability measure: `SD`, `%`, … |
| 8 | Variation value | the single value of that variation |
| 9 | Interval type | `95% CI`, `Range`, `IQR`, … |
| 10 | Lower bound | lower end of the interval |
| 11 | Upper bound | upper end of the interval |
| 12 | P value | the value's p-value (often shared across rows — search the whole table) |

Use `"N/A"` (string) for cells that cannot be filled.

## Scratch directory (state between stages)
Persist each stage's output to a file and read it back when the next stage needs
it — do not rely on the conversation alone (a long run can be summarized).

**Create the scratch directory in the user's current project/working directory —
NOT inside this skill's folder.** Concretely, the path is
`./.pe_study_outcome_scratch/<pmid>/` relative to where the user is working. Never
write scratch files under `pipelines/pe-study-outcome/`. If unsure of the working
directory, run `pwd` and create the scratch folder there.

This skill curates **one outcome table per run** (the table the user provides);
there is no multi-table selection. The run is flat:

```
.pe_study_outcome_scratch/<pmid>/
├── 00_markdown_table.md # Stage 0: the source table in markdown
├── inputs.md            # the caption + footnotes + (optional) title, verbatim
├── 01_values.md         # Stage 1: every numeric cell as a [Value] column
├── 02_param_values.md   # Stage 2: [Main value, Main value unit, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound, P value]
├── 03_study_info.md     # Stage 3: [Characteristic, Exposure, Outcome]
├── 04_assembled.csv     # Stage 4: the 12 working-name columns (pre-cleanup)
├── 05_final.csv         # Stage 5: cleanup-script output, the final 12-col schema (corrected in place by stage 6)
└── 06_verification_report.md # Stage 6 output
```

## Procedure
Run the stages **in order**. For each stage: read its input file(s) and its
prompt file, reason explicitly, produce the output, sanity-check it (redo **once**
if a check fails, then carry forward noting any residual issue), and **write** the
output to its scratch file before moving on. Stages 1–3 carry the **same row set
in the same order** — one row per retained numeric value — so stages 2 and 3 have
the same row count as stage 1, which stage 4 relies on.

### Stage 0 — Convert the table (not a prompt file)
If the table is HTML, convert it to markdown with the bundled script — do **not**
parse HTML by hand:

```
python scripts/html_to_markdown_table.py <path-to-html>
```

Write the result to `00_markdown_table.md` and the caption/footnotes/(title) to
`inputs.md`. Skip the conversion if the user already supplied markdown.

### Stages
1. **Numeric retain** — `prompts/01_numeric_retain.md`
   Reads `00_markdown_table.md` → writes `01_values.md`. Emit **every** table cell
   that contains a digit as a one-column `[Value]` table, in row-major order
   (left-to-right, top-to-bottom). One row per numeric cell.

2. **Parameter value** — `prompts/02_param_value.md`
   Reads `01_values.md` + `00_markdown_table.md` + `inputs.md` →
   `02_param_values.md`. Parse each `Value` into `[Main value, Main value unit,
   Statistics type, Variation type, Variation value, Interval type, Lower bound,
   Upper bound, P value]` (9 columns), row-for-row.

3. **Study info** — `prompts/03_study_info.md`
   Reads `01_values.md` + `00_markdown_table.md` + `inputs.md` →
   `03_study_info.md`. For each `Value`, locate it in the table and tag its
   `[Characteristic, Exposure, Outcome]` (3 columns), row-for-row.

4. **Assembly** — `prompts/04_assembly.md`
   Reads `03_study_info.md` + `02_param_values.md` → writes `04_assembled.csv`.
   Positional horizontal join into the **12 working-name columns**:
   `Characteristic, Exposure, Outcome, Main value, Main value unit, Statistics
   type, Variation type, Variation value, Interval type, Lower bound, Upper bound,
   P value`.

5. **Row cleanup** — `prompts/05_row_cleanup.md`
   Runs `scripts/clean_pe_outcome_rows.py` on
   `04_assembled.csv` → `05_final.csv`. Deterministic: applies the interval/
   statistic business rules, normalizes sentinels, drops non-numeric rows, and
   renames/reorders into the final 12-column schema.

6. **Verification + correction** — `prompts/06_verify_and_correct.md`
   Reads `05_final.csv` + `00_markdown_table.md` + `inputs.md` → corrects
   `05_final.csv` in place and writes `06_verification_report.md`. The quality
   gate; see below.

## Validation
After cleanup (stage 5), before verification, check the table row by row:
- 12 columns, in the order given in "Output schema".
- `Parameter value`, `Variation value`, `Lower bound`, `Upper bound`, `P value`
  are numbers or `"N/A"`.
- `Outcome` describes what the value measures — it is **never** the number itself.
- `Parameter unit` is a unit, not a statistic (SD belongs in `Variation type`).

If any row fails, fix it inline (do not silently drop it) and re-check.

## Verification + correction
Stage 6 follows `verify_and_correct.md`. Unlike the
full-text PE skill, the source here **is a table**, so run the provenance script
**with `--attribution`** (it can match each value to the source column/row it best
fits). The stage prompt fills in the exact parameters.

## Error handling rules
- **Empty table / no numeric cells** at stage 1: there is nothing to curate —
  report that and stop.
- **A count + percentage reported together** (e.g. `10 (5%)`): put the count in
  `Parameter value` with unit `Count` and statistic `Sum`, and the percentage in
  `Variation type` / `Variation value` (per the stage-2 prompt).
- **An interval of two numbers**: split into `Lower bound` / `Upper bound`, never
  into `Variation value`.
- **A p-value shared across rows**: search the whole table and fill it into every
  row it applies to.

## What this skill deliberately does NOT do
- It does not fetch papers from PubMed or any URL.
- It does not extract PE study metadata — that is `pe-study-info`.
- It implements `pe_study_outcome_ver2`; the deprecated v1 (header/row
  categorize + mapping) is not ported.
- Its verify → correct loop (stage 6) is **bounded** (≤2 rounds).
- It does not score itself against a gold standard — that's `benchmark/`.

## Write out the result (do this last)
When the procedure above finishes, copy its **final deliverable CSV** (the `combined_final.csv` / `NN_final.csv` written by the last stage) to the output location, leaving the scratch copy in place:

```bash
# output base: $SKILL_OUTPUT_FOLDER if set, else the current directory
OUT="${SKILL_OUTPUT_FOLDER:-.}"; mkdir -p "$OUT/<pmid>"
cp <final-csv-in-scratch> "$OUT/<pmid>/pe-study-outcome.csv"
```

If `SKILL_OUTPUT_FOLDER` is unset this writes `./<pmid>/pe-study-outcome.csv` in the user's current working directory. Then tell the user the exact path you wrote.
