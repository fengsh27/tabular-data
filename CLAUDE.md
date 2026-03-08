# Repository Guidelines

## Project Structure & Module Organization
- `app.py` and `components/` host the Streamlit UI; `images/` assets are referenced by the app.
- Core extraction logic lives in `extractor/` (agents, request clients, utils, database helpers).
- Benchmarks live in `benchmark/` with input data under `benchmark/data/` and results in `benchmark/result/`.
- Tests are split between `tests/` (unit/integration) and `system_tests/` (end-to-end flows with fixtures in `system_tests/data/`).
- Example/fixture data also appears in `data/` and `tests/data/`.

## Build, Test, and Development Commands
- `poetry install -E semantic -E claude` installs dependencies with optional LLM extras.
- `poetry shell` or `poetry run <command>` activates the environment.
- `poetry run streamlit run app.py` launches the UI.
- `poetry run pytest tests` runs the main test suite.
- `poetry run pytest system_tests` runs system tests (slower, uses larger fixtures).
- Benchmark runs (see `README.md` for required env vars):
  - `poetry run pytest benchmark/test_pk_summary_benchmark_with_semantic.py`
  - `poetry run pytest benchmark/test_pk_summary_benchmark_with_llm.py`

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
