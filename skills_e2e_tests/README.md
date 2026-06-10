# skills_e2e_tests

Regression fixtures and tests for the bundled Claude curation skill
`skills/pk-pe-curation/`. The ten curation pipelines live under
`skills/pk-pe-curation/pipelines/` — the table pipelines `pk-summary-curation` and
`pk-individual-curation`, the full-text pipelines `pk-drug-summary`,
`pk-drug-individual`, `pk-specimen-summary`, `pk-specimen-individual`,
`pk-population-summary`, `pk-population-individual`, `pe-study-info`, the table
pipeline `pe-study-outcome`, plus the `prepare-paper` and `route` procedures — and
the shared tooling sits in `skills/pk-pe-curation/curation-common/` (the bundled
scripts + the generic verify/correct and population-refine prompts the pipelines
reuse). Each *case* bundles a real paper's source table plus the expected output of
the early curation stages, so we can detect drift as the prompts and scripts evolve
— and compare behavior across the models we run them on (Claude, and the open LLMs
served via Ollama on OSC).

The deterministic pieces (HTML→Markdown conversion, the prepare-paper converter,
the provenance checker, the row-cleanup scripts, the route label→procedure map,
selection-case structure, and the bundle structure/schema) are asserted
byte-exactly in CI. The model-driven stages can't be — they are evaluated against
semantic oracles, by hand or in an eval harness.

**Table vs full-text pipelines.** The table pipelines have a fixture `case` (source
HTML + per-stage oracles) and run the Stage-0 conversion test. The full-text
pipelines take the paper's prose, not a table, and their legacy row-cleanup is a
no-op, so their automatable surface is the **structural/schema test**
(`test_skill_structure.py`) plus the shared `verify_provenance.py` existence check;
their model-driven stages are evaluated by the same hand/eval method as the table
pipelines.

## Layout

```
skills_e2e_tests/
├── README.md
├── conftest.py                    # discovers cases/ + selection_cases/; `case` / `selection_case` fixtures
├── test_prepare_paper.py          # deterministic: prepare-paper HTML→input-layout converter (CI-safe)
├── test_stage0_conversion.py      # deterministic: shared HTML→Markdown converter (CI-safe)
├── test_provenance.py             # deterministic: curation-common/verify_provenance.py (CI-safe)
├── test_clean_individual_rows.py  # deterministic: pk-individual cleanup script (CI-safe)
├── test_clean_specimen_rows.py    # deterministic: shared specimen cleanup script (CI-safe)
├── test_clean_population_rows.py   # deterministic: population-individual cleanup script (CI-safe)
├── test_clean_pe_outcome_rows.py   # deterministic: pe-study-outcome cleanup script (CI-safe)
├── test_table_selection.py        # deterministic: Stage 0b selection-case structure (CI-safe)
├── test_skill_structure.py        # deterministic: per-skill front-matter / refs / schema (CI-safe)
├── test_pipeline_skill_map.py     # deterministic: pk-pe-route label→skill map (CI-safe)
├── prepare_cases/
│   ├── sample_paper.html          # synthetic PMC-style paper  ── INPUT
│   ├── expected_paper_text.md     # golden: title H1 + body, refs stripped, [Table N] markers
│   ├── expected_abstract.md       # golden: abstract
│   └── expected_table_1.md … _2.md# golden: per-table caption + footnotes
├── cases/
│   └── 16143486_table_4/          # pk-summary single-table case (source HTML + per-stage oracles)
│       ├── meta.json                     # pmid, table id, oracle map, provenance
│       ├── title.txt                     # paper title  ── INPUT
│       ├── caption.txt                   # caption + footnote  ── INPUT
│       ├── source_table.html             # the source <table>  ── INPUT
│       ├── expected_00_markdown_table.md # Stage 0 golden (deterministic)
│       ├── expected_01_drug_table.md     # Stage 1 oracle (semantic)
│       ├── expected_02_patient_table.md  # Stage 2 oracle (semantic, soft)
│       └── expected_03_patient_refined.md# Stage 3 oracle (semantic)
└── selection_cases/
    └── 16143486_selection/               # multi-table input for Stage 0b
        ├── meta.json                     # tables[], selection oracle, rationale, provenance
        ├── title.txt                     # paper title  ── INPUT
        ├── table_1.html … table_4.html   # the 4 source <table>s  ── INPUT
        └── (selection oracle lives in meta.json: expected.selected / .excluded)
```

The `cases/` and `selection_cases/` fixtures are currently **pk-summary**. The
deterministic script tests (`test_provenance.py`, `test_clean_individual_rows.py`)
are self-contained — they build their own tiny inputs inline and need no case
directory. A pk-individual semantic case (per-subject oracles) is a documented
gap; see "Adding a case".

## Two kinds of oracle

Each skill has a few deterministic pieces and several model-driven stages, so
the suite is split accordingly. **Do not** try to assert the LLM stages
byte-exactly. (The table below is the pk-summary case; pk-individual mirrors it
with a `Patient ID`-keyed patient stage and a 12-column output.)

| Stage | Oracle | How it is checked |
|-------|--------|-------------------|
| 0 — HTML→Markdown | `expected_00_markdown_table.md` | **Deterministic, byte-exact.** Enforced by `test_stage0_conversion.py` in CI. |
| 1 — Drug info | `expected_01_drug_table.md` | **Semantic.** Set-equality of `[Drug name, Analyte, Specimen]` rows (order/whitespace-insensitive). |
| 2 — Patient info | `expected_02_patient_table.md` | **Semantic, soft.** The model may legitimately emit *more* Subject-N rows; the expected set must be **covered** (subset), with no spurious cohorts. |
| 3 — Patient refine | `expected_03_patient_refined.md` | **Semantic.** Row-preserving vs Stage 2; checks Population/Pregnancy-stage normalization and the Pediatric/Gestational-age rule. |

## Running

All of these are deterministic (no model) and safe for CI:

```bash
poetry run pytest skills_e2e_tests            # everything
poetry run pytest skills_e2e_tests/test_prepare_paper.py -v           # prepare-paper HTML→input layout
poetry run pytest skills_e2e_tests/test_stage0_conversion.py -v       # HTML→Markdown
poetry run pytest skills_e2e_tests/test_provenance.py -v              # provenance + attribution
poetry run pytest skills_e2e_tests/test_clean_individual_rows.py -v   # pk-individual cleanup
poetry run pytest skills_e2e_tests/test_clean_specimen_rows.py -v     # specimen cleanup
poetry run pytest skills_e2e_tests/test_clean_population_rows.py -v   # population-individual cleanup
poetry run pytest skills_e2e_tests/test_clean_pe_outcome_rows.py -v   # pe-study-outcome cleanup
poetry run pytest skills_e2e_tests/test_table_selection.py -v         # Stage 0b selection structure
poetry run pytest skills_e2e_tests/test_skill_structure.py -v         # bundle front-matter / refs / schema
poetry run pytest skills_e2e_tests/test_pipeline_skill_map.py -v      # route label→procedure map
```

`test_stage0_conversion.py` parametrizes over every case under `cases/` and
asserts the **shared** converter
(`skills/pk-pe-curation/curation-common/scripts/`) still reproduces each
`expected_00_markdown_table.md`.

## Deterministic script tests (no fixtures needed)

Several of the skills' moving parts are pure scripts, shared or pipeline-specific,
and are tested directly with inline inputs:

| Test | Script under test | What it guards |
|------|-------------------|----------------|
| `test_prepare_paper.py` | `pk-pe-curation/curation-common/scripts/prepare_paper.py` | title→H1 + reference stripping + `[Table N]` marker substitution in `paper_text.md`; abstract extraction; per-table caption+footnotes markdown; table grids stay in `table_<n>.html` (not inlined); `manifest.json` marker↔file index; `--dry-run` writes nothing |
| `test_provenance.py` | `pk-pe-curation/curation-common/scripts/verify_provenance.py` | existence + attribution checks; ranges split; NA/text ignored; `--value/skip/label-columns`; the cord↔maternal **swap** is caught by attribution though existence passes |
| `test_clean_individual_rows.py` | `pk-pe-curation/pipelines/pk-individual-curation/scripts/clean_individual_rows.py` | drop-ERROR / drop-N/A-value rows, long-time-unit blanking + time coupling, Cmax/Tmax/Cavg time-blank, sentinel normalization, dedupe, Patient-ID-first column order |
| `test_clean_specimen_rows.py` | `pk-pe-curation/curation-common/scripts/clean_specimen_rows.py` | remove-half-total row (with the `v!=0` guard), keep-max-`Sample N` dedupe, `Population N`/`Note` excluded from the comparison, distinct specimens/patients kept apart, original order preserved, non-integer `Sample N` → no-op |
| `test_clean_population_rows.py` | `pk-pe-curation/curation-common/scripts/clean_population_individual_rows.py` | drop blank/`N/A` `Characteristic value` rows (incl. slash-normalized `N / A`), keep real values verbatim, preserve order, tolerant when the value column is absent |
| `test_clean_pe_outcome_rows.py` | `pk-pe-curation/curation-common/scripts/clean_pe_outcome_rows.py` | interval/statistic business rules (Main==bound blanking, both-bounds→`Range`, value-contains-bounds blanking, N/A propagation), sentinel normalization (`Standard Deviation (SD)`→`SD`), drop non-numeric rows, working→final rename + reorder, order preserved |
| `test_pipeline_skill_map.py` | `pk-pe-curation/pipelines/route/scripts/pipeline_skill_map.py` | map covers every `PipelineTypeEnum` value (parsed from `extractor/constants.py`), every target `procedure.md` exists, `resolve()` preserves order / de-dupes / ignores blanks / rejects unknown labels, the irregular `pk_summary`→`pipelines/pk-summary-curation` cases, `build_selection` artifact shape |

Because these scripts are model-independent, the assertions are true byte-level
checks — unlike the LLM curation stages.

## Bundle structure / schema test

`test_skill_structure.py` asserts the deterministic, file-level invariants of the
bundled `pk-pe-curation` skill that rot silently as prompts are edited. For the
full-text pipelines, whose row-cleanup is a no-op, this is the main automatable
guard.

| Check | What it guards |
|-------|----------------|
| top front-matter | `pk-pe-curation/SKILL.md` has `---` front-matter with `name: pk-pe-curation` and a non-empty `description:` |
| referenced prompts exist | every `prompts/<file>.md` named in a pipeline's `procedure.md` is present under that pipeline |
| bundle paths resolve | every `curation-common/…` / `pipelines/…` path referenced anywhere in the bundle exists (resolved against the skill root) |
| no stale `skills/…` refs | no `skills/<old-name>/` reference survives bundling (would break once installed under `.claude/skills/pk-pe-curation/`) |
| output schema | the schema table in each pipeline's `procedure.md` matches a known-expected column list (`EXPECTED_SCHEMAS`) — 19-col pk-summary, 12-col pk-individual, 11-col pk-drug-*, … |

When you intentionally change a pipeline's output schema, update `EXPECTED_SCHEMAS`
in `test_skill_structure.py` in the same commit — the diff is the record that the
change was deliberate.

## Evaluating the LLM stages (01–03)

These run a model, so they are evaluated manually / in an eval harness rather
than as a pass/fail unit test:

1. Set up a scratch dir as the skill specifies, e.g.
   `.pk_curation_scratch/16143486_table_4/`.
2. Stage 0: `python skills/pk-pe-curation/curation-common/scripts/html_to_markdown_table.py \
   skills_e2e_tests/cases/16143486_table_4/source_table.html` → `00_markdown_table.md`;
   copy `title.txt` + `caption.txt` into `inputs.md`.
3. Drive the skill (under Claude, or under Claude Code pointed at the OSC Ollama
   server) through stages 01→02→03.
4. Compare the produced `01_drug_table.md` / `02_patient_table.md` /
   `03_patient_refined.md` against this case's `expected_*` files using the
   oracle rules in the table above.

The `expected_*` tables are the regression baseline: when a prompt edit changes
an LLM stage's output, diff against these and decide whether the change is an
improvement or a regression.

## Stage 0b — table selection (multi-table input)

When the input has several tables, both skills first select the PK tables
(Stage 0b, `prompts/00b_select_pk_tables.md` — pk-summary prefers aggregate
tables, pk-individual prefers per-subject tables, but the selection structure is
the same). The `selection_cases/` family covers this. A selection case bundles
every input table plus a **selection oracle** in `meta.json`
(`expected.selected` / `expected.excluded`) and a per-table `rationale`.

`test_table_selection.py` asserts the **deterministic** parts in CI:

| Check | What it guards |
|-------|----------------|
| oracle partitions all tables | `selected ∪ excluded` = every table, disjoint, both non-empty |
| rationale covers every table | each table has a documented include/exclude reason |
| Stage-0a renders every table | the bundled converter turns each table into a labeled `## table_N` block (the `00_all_tables.md` input the selection prompt reads) |

The **selection judgment itself is semantic** — to evaluate it, build the
`00_all_tables.md` input (the `build_all_tables_md` helper in `conftest.py` does
exactly what Stage 0a does), drive the skill through Stage 0b under your model,
and compare the chosen labels against `expected.selected`. For
`16143486_selection`: Table 1 (parturient demographics) should be **excluded**;
Tables 2 (kinetic disposition), 3 (urinary excretion), and 4 (transplacental
distribution) should be **selected**.

## Provenance

The `16143486_table_4` case is lifted verbatim from `tests/conftest.py`
(fixtures `*_16143486_table_4`), which the legacy `pk_summary` system tests
already treat as ground truth. That keeps the skill's expected outputs anchored
to the same references as the existing pipeline.

## Adding a case

Create `cases/<pmid>_<tableid>/` with `meta.json`, `title.txt`, `caption.txt`,
`source_table.html`, and the `expected_*` oracle files. The `case` fixture and
the Stage-0 test pick it up automatically (note: `test_stage0_conversion.py`
runs the byte-exact converter on **every** case, so a case must include
`source_table.html` + `expected_00_markdown_table.md`). Good sources for accurate
inputs + oracles: the `*_table_*` fixtures in `tests/conftest.py` and the
per-PMID `system_tests/conftest_data_*.py` files.

### Known gap — pk-individual semantic case

There is **no** `cases/` entry for `pk-individual-curation` yet. The clean
per-subject fixture (`system_tests/conftest_data_33253437.py`, with
`md_table_patient` → Patient ID, `md_table_aligned`, `col_mapping`, and the
sub-table list as ready-made oracles) ships only **markdown**, not source HTML,
so it can't drive the byte-exact Stage-0 test. To add it: feed that markdown as
the already-converted Stage-0 input (skip conversion), store the conftest
fixtures as semantic oracles for stages 1–3 / 7 / 10, and evaluate them by the
same hand/eval-harness method as the pk-summary LLM stages. The deterministic
cleanup is already covered by `test_clean_individual_rows.py`.
