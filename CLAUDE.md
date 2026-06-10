# Repository Guidelines

## Project Structure & Module Organization
- `app.py` and `components/` host the Streamlit UI; `images/` assets are referenced by the app.
- Core extraction logic lives in `extractor/`:
  - `extractor/agents/` — per-pipeline LangGraph workflows (`pk_summary`, `pk_individual`, `pk_population_*`, `pk_specimen_*`, `pk_drug_*`, `pe_study_info`, `pe_study_outcome*`).
  - `extractor/agents/pk_pe_agents/` — high-level orchestration steps (identification, design, execution, verification, correction, correction-code).
  - `extractor/agents_manager/` — top-level `PKPEManager` plus per-pipeline `AgentToolTask` graphs (`pk_summary_task.py`, `pk_individual_task.py`, `pk_populattion_task.py`, `pk_fulltext_tool_task.py`, `pe_study_task.py`, etc.); the shared `pk_pe_agenttool_task.py` builds the `execution → verification → (correction → verification)*` loop.
  - `extractor/request_*`, `extractor/utils*`, and `extractor/database/` provide LLM clients, helpers, and DB access.
- Batch CLIs: `app_script.py` (single-model multi-pipeline) and `app_script_pmids.py` (full PKPEManager orchestration over a PMID CSV).
- Benchmarks live in `benchmark/` with input data under `benchmark/data/<pipeline>/<version>/`, the column/anchor config in `benchmark/configs.py`, shared helpers in `benchmark/common.py` and `benchmark/comm_semantic.py`, PE normalization in `benchmark/pe_preprocess.py`, and results in `benchmark/result/`.
- Tests are split between `tests/` (unit/integration) and `system_tests/` (end-to-end flows with fixtures in `system_tests/data/` and per-PMID `conftest_data_*.py`).
- `scripts/` contains stand-alone utilities (e.g. `prepare_htmls_by_pmids.py`, `add_llm_suffix.py`, `convert_md_table_to_csv.py`).
- Example/fixture data also appears in `data/` and `tests/data/`.
- `skills/pk-pe-curation/` is the **Claude Skills** re-implementation of the PK/PE
  curation suite — a **single bundled skill** (prose `SKILL.md` + `prompts/` +
  deterministic `scripts/`), meant to run under Claude or under Claude Code pointed
  at an Ollama server. It is installed by copying the one folder into
  `<project>/.claude/skills/` (see `skills/pk-pe-curation/INSTALL.md`), so Claude
  Code auto-triggers it from its `description`. Internal layout:
  - `skills/pk-pe-curation/SKILL.md` — the one triggerable **orchestrator**:
    prepare → route (identify + design) → dispatch to the selected pipelines. Opens
    with a path-anchor note (all `curation-common/...` and `pipelines/...` paths are
    relative to the skill's own dir, so they resolve under `.claude/skills/` once
    installed).
  - `skills/pk-pe-curation/pipelines/prepare-paper/` — **front door**
    (deterministic): raw paper HTML → canonical input layout (`paper_text.md` with
    references stripped + tables replaced by `[Table N]` markers, `abstract.md`,
    `table_<n>.md/.html`, `manifest.json`). HTML only (XML is a follow-up).
  - `skills/pk-pe-curation/pipelines/route/` — **selector** (ports
    `PKPEIdentificationStep` + `PKPEDesignStep`): classifies PK/PE/Both/Neither,
    selects the union of applicable pipelines, emits `selected_pipelines.json` via a
    byte-tested label→procedure map (`scripts/pipeline_skill_map.py`).
  - `skills/pk-pe-curation/pipelines/{pk-summary-curation (19 col), pk-individual-curation
    (12 col), pk-drug-summary / pk-drug-individual (11 col), pk-specimen-summary /
    pk-specimen-individual (9 col), pk-population-summary (15 col) /
    pk-population-individual (9 col), pe-study-info (10 col), pe-study-outcome
    (12 col)}/` — the 10 curation pipelines, each `procedure.md` + `prompts/`
    (pk-individual also has its own `scripts/`). `pe-study-outcome` implements
    `pe_study_outcome_ver2`; the deprecated v1 is not ported.
  - `skills/pk-pe-curation/curation-common/` — shared scripts (`prepare_paper.py`,
    `html_to_markdown_table.py`, `verify_provenance.py`, `pipeline_skill_map.py` is
    in `pipelines/route/scripts/`, `clean_specimen_rows.py`,
    `clean_population_individual_rows.py`, `clean_pe_outcome_rows.py`) + the generic
    `verify_and_correct.md` and `refine_population.md`; a support library.
  - `skills_e2e_tests/` — deterministic (CI-safe) regression fixtures and tests
    for the skill. See `SKILLS_INTRO.md` for the full overview.

## Build, Test, and Development Commands
- `poetry install -E semantic -E claude` installs dependencies with optional LLM extras.
- `poetry shell` or `poetry run <command>` activates the environment.
- `poetry run streamlit run app.py` launches the UI.
- `poetry run python app_script_pmids.py -f ./data/pmids.csv -o ./out` runs the full PKPEManager pipeline over a PMID list (see README for the summary CSV format).
- `poetry run pytest tests` runs the main test suite.
- `poetry run pytest system_tests` runs system tests (slower, uses larger fixtures).
- `poetry run pytest skills_e2e_tests` runs the curation-skills regression tests
  (deterministic, CI-safe — converter, provenance/attribution, row cleanup, and
  Stage 0b selection structure). See `SKILLS_INTRO.md` and
  `skills_e2e_tests/README.md`.
- Benchmark runs (see `README.md` for required env vars):
  - Per-pipeline (legacy): `poetry run pytest benchmark/test_pk_summary_benchmark_with_semantic.py` (or `_with_llm.py`, `test_pk_individual_benchmark_with_semantic.py`, `test_pe_benchmark_with_semantic.py`).
  - Combined multi-pipeline: `poetry run pytest benchmark/test_pk_pe_benchmark_with_semantic.py` — drives all PK/PE pipelines from `benchmark/data/pk-pe/<version>/`, scored via `benchmark/configs.py`.

## Coding Style & Naming Conventions
- Python-only codebase; follow PEP 8 with 4-space indentation.
- Prefer `snake_case` for functions/variables and `PascalCase` for classes (see `extractor/agents/`).
- Use `pre-commit` with gitleaks before pushing: `pre-commit run --all-files`.

## Testing Guidelines
- Pytest is the default framework (`tests/`, `system_tests/`, `benchmark/`).
- Name tests `test_*.py` and keep fixtures in `conftest.py` files.
- When adding benchmark tests, mirror existing patterns in `benchmark/test_*` and store inputs under `benchmark/data/...`.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and task-focused (e.g., "adjust pk individual to support qwen3").
- PRs should include a brief summary, the tests run, and any required data/model setup notes.
- If UI changes are made, include screenshots or a short GIF of the Streamlit app.

## Security & Configuration Tips
- Copy `.env.template` to `.env` and set LLM keys/models before running benchmarks or the app.
- Avoid committing API keys; gitleaks will fail pre-commit if secrets are detected.
