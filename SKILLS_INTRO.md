# Curation Skills

This repository contains a set of **Claude Skills** that re-implement the PK
table-curation pipelines as model-driven, prose-defined procedures rather than
hard-coded LangGraph workflows. They live under `skills/` and are exercised by
the regression fixtures in `skills_e2e_tests/`.

The goal is to run the same curation logic under **Claude Code pointed at an
Ollama server** (Gemma / Qwen on OSC) as well as under Claude itself, and to be
able to compare model behavior across both. A skill is just a folder the model
reads — no Python runtime, no graph engine — so it ports across hosts as long
as the model can follow instructions and call the few bundled scripts.

## What a skill is here

Each skill is a directory with:
- a `SKILL.md` — YAML front-matter (`name`, `description` used for selection)
  plus a prose description of the inputs, the output schema, and the ordered
  procedure;
- a `prompts/` directory — one markdown file per stage, loaded by `SKILL.md`
  as the model reaches that stage;
- optionally a `scripts/` directory — small, **deterministic** stdlib/`bs4`
  Python helpers for the parts that must not be left to the model.

There is **no templating engine**. The prompt files are prose the model reads
and follows, not `{variable}` templates rendered by code. State between stages
is passed through files on disk, not through a graph's typed state object.

## The three skills

| Skill | Purpose | Output |
|-------|---------|--------|
| `pk-summary-curation` | Curate aggregate PK tables (mean / median / SD / range / CI of AUC, Cmax, t½, CL, Vd, …) into a normalized dataset. | **19 columns** |
| `pk-individual-curation` | Curate per-subject PK tables (one row = one patient's own measured value) into a normalized dataset. | **12 columns** |
| `curation-common` | Shared support library — the bundled scripts and the generic verify/correct procedure both pipelines reuse. **Not invoked directly.** | — |

`pk-summary-curation` and `pk-individual-curation` are the two user-facing
skills; the model selects one based on whether the table reports cohort
aggregates or per-individual values. `curation-common` is referenced by the
other two via file path and should never be selected on its own.

### Key differences between the two pipelines

The two pipelines mirror each other but are not the same:
- **pk-individual** keys on a per-row `Patient ID` (not a `Subject N` count),
  has **no** statistic / variation / interval / P-value block (each row is one
  raw value), and **inverts** the summary-vs-individual row deletion (it keeps
  the per-subject rows).
- **pk-individual Stage 0c** infers and injects a `Patient ID` from the paper's
  **full text** when the table itself has none — essential for single-patient
  case reports, where the subject's identity lives in the prose. Without it such
  a table would be discarded for lacking a Patient ID. This is why
  pk-individual takes the full text as an input and pk-summary does not.
- The parameter-type transpose condition, the header-category set (3 vs 5
  categories), and the per-stage column lists differ accordingly.

## Scratch-state model

A long curation run can have its conversation context summarized, which would
corrupt the exact table text later stages depend on. So **every stage writes its
output to a file and reads its inputs back from disk** — the conversation is
never the source of truth for intermediate tables.

- Summary runs use `./.pk_curation_scratch/<pmid>/`;
  individual runs use `./.pk_individual_scratch/<pmid>/`.
- The scratch directory is created in the **user's current project / working
  directory**, never inside the skill folder (the skill folder is read-only
  content). Both are git-ignored.
- The input may hold several tables. Each selected table gets its own
  `table_<n>/` sub-directory holding its full per-stage run; cross-table files
  (the all-tables listing, the selection note, the combined result) live at the
  top level. The pipeline **always nests, even for a single table** — one code
  path, fewer mistakes.

## Procedure shape

Both pipelines share the same top-level loop:
1. **Stage 0a** — convert every input table (HTML or markdown) to markdown and
   list them, using the bundled `html_to_markdown_table.py` (deterministic
   colspan/rowspan/multi-row-header handling — never parse HTML by hand).
2. **Stage 0b** — if there is more than one table, select the PK tables;
   otherwise skip the judgment.
3. **For each selected table** — run the ordered numbered stages (1–14 for
   summary; a 0c preprocess + 1–14 for individual) in its `table_<n>/`.
4. **Combine** — vertically concatenate each table's final CSV into
   `combined_final.csv`, the deliverable.

The final numbered stage of each pipeline is a **bounded verify → correct
quality gate** (`curation-common/verify_and_correct.md`): a deterministic
provenance + attribution check (`verify_provenance.py`) plus an adversarial
semantic review, then at most **2 correction rounds**. It never silently drops a
row to pass, and surfaces unresolved findings to the user.

## Deterministic vs semantic

The pieces that *can* be made deterministic are isolated into scripts and tested
byte-exactly in CI; everything else is model-driven and evaluated against
semantic oracles by hand or in an eval harness.

- **Deterministic (byte-exact in CI):** HTML→Markdown conversion
  (`html_to_markdown_table.py`), the provenance/attribution checker
  (`verify_provenance.py`), the pk-individual row cleanup
  (`clean_individual_rows.py`), and the Stage-0b selection-case *structure*.
- **Semantic (oracle-checked, not byte-exact):** every LLM stage — drug/patient
  extraction, alignment, splitting, value/time extraction, the selection
  judgment itself, and the verify/correct reasoning.

`verify_provenance.py` is schema-agnostic — it checks **numbers**, not column
names. Its *existence* check catches hallucinated/mistyped values; its
*attribution* check (argmax-Jaccard label matching) catches values copied to the
wrong row / cohort / specimen even when the number does appear somewhere in the
source.

## Testing

All skill regression tests live in `skills_e2e_tests/` and are deterministic
(no model), safe for CI:

```bash
poetry run pytest skills_e2e_tests                                    # everything
poetry run pytest skills_e2e_tests/test_stage0_conversion.py -v       # HTML→Markdown
poetry run pytest skills_e2e_tests/test_provenance.py -v              # provenance + attribution
poetry run pytest skills_e2e_tests/test_clean_individual_rows.py -v   # pk-individual cleanup
poetry run pytest skills_e2e_tests/test_table_selection.py -v         # Stage 0b selection structure
```

Fixtures bundle a real paper's source table plus the expected output of the
early stages, so prompt/script drift is detectable. See
`skills_e2e_tests/README.md` for the case layout, the two kinds of oracle, how
to evaluate the model-driven stages by hand, and the documented gap (a
pk-individual semantic case is not yet bundled).

## Relationship to the legacy pipeline

These skills are a re-expression of the LangGraph pipelines in `extractor/`
(see `CLAUDE.md` and the project memory). The legacy code remains the source of
truth for behavior, and `benchmark/` remains the source of truth for **accuracy**
— the skills deliberately do **not** score themselves against a gold standard.
When porting a stage, the pattern is: read the legacy agent in
`extractor/agents/<pipeline>/`, translate it to a prose prompt + scratch file,
and verify deterministically wherever the stage allows it.
