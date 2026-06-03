# skills_e2e_tests

Regression fixtures and tests for the Claude **skills** (currently
`skills/pk-summary-curation/`). Each *case* bundles a real paper's source table
plus the expected output of the early curation stages, so we can detect drift
as the skill prompts and scripts evolve — and compare the skill across the
models we run it on (Claude, and the open LLMs served via Ollama on OSC).

## Layout

```
skills_e2e_tests/
├── README.md
├── conftest.py                 # discovers cases/, exposes the `case` fixture
├── test_stage0_conversion.py   # deterministic regression test (CI-safe)
└── cases/
    └── 16143486_table_4/
        ├── meta.json                     # pmid, table id, oracle map, provenance
        ├── title.txt                     # paper title  ── INPUT
        ├── caption.txt                   # caption + footnote  ── INPUT
        ├── source_table.html             # the source <table>  ── INPUT
        ├── expected_00_markdown_table.md # Stage 0 golden (deterministic)
        ├── expected_01_drug_table.md     # Stage 1 oracle (semantic)
        ├── expected_02_patient_table.md  # Stage 2 oracle (semantic, soft)
        └── expected_03_patient_refined.md# Stage 3 oracle (semantic)
```

## Two kinds of oracle

The skill has one deterministic stage and several model-driven stages, so the
suite is split accordingly. **Do not** try to assert the LLM stages byte-exactly.

| Stage | Oracle | How it is checked |
|-------|--------|-------------------|
| 0 — HTML→Markdown | `expected_00_markdown_table.md` | **Deterministic, byte-exact.** Enforced by `test_stage0_conversion.py` in CI. |
| 1 — Drug info | `expected_01_drug_table.md` | **Semantic.** Set-equality of `[Drug name, Analyte, Specimen]` rows (order/whitespace-insensitive). |
| 2 — Patient info | `expected_02_patient_table.md` | **Semantic, soft.** The model may legitimately emit *more* Subject-N rows; the expected set must be **covered** (subset), with no spurious cohorts. |
| 3 — Patient refine | `expected_03_patient_refined.md` | **Semantic.** Row-preserving vs Stage 2; checks Population/Pregnancy-stage normalization and the Pediatric/Gestational-age rule. |

## Running

Deterministic regression test (fast, no model, safe for CI):

```bash
poetry run pytest skills_e2e_tests/test_stage0_conversion.py -v
```

This parametrizes over every case under `cases/` and asserts the skill's
bundled converter still reproduces each `expected_00_markdown_table.md`.

## Evaluating the LLM stages (01–03)

These run a model, so they are evaluated manually / in an eval harness rather
than as a pass/fail unit test:

1. Set up a scratch dir as the skill specifies, e.g.
   `.pk_curation_scratch/16143486_table_4/`.
2. Stage 0: `python skills/pk-summary-curation/scripts/html_to_markdown_table.py \
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

## Provenance

The `16143486_table_4` case is lifted verbatim from `tests/conftest.py`
(fixtures `*_16143486_table_4`), which the legacy `pk_summary` system tests
already treat as ground truth. That keeps the skill's expected outputs anchored
to the same references as the existing pipeline.

## Adding a case

Create `cases/<pmid>_<tableid>/` with `meta.json`, `title.txt`, `caption.txt`,
`source_table.html`, and the `expected_*` oracle files. The `case` fixture and
the Stage-0 test pick it up automatically. Good sources for accurate inputs +
oracles: the `*_table_*` fixtures in `tests/conftest.py` and the per-PMID
`system_tests/conftest_data_*.py` files.
