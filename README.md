# tabular-data

## Benchmark
We use [Poetry](https://python-poetry.org) for dependency management. Please make sure that you have installed Poetry and set up the environment correctly before starting development.

### setup environment
- Install dependencies from the lock file: `poetry install -E semantic -E claude`

- Use the environment: You can either run commands directly with `poetry run
<command>` or open a shell with `poetry shell` and then run commands directly.

### prepare environment variables
- copy `.env.template` and rename to `.env`
- in `.env`, set api key and model for the desired LLM (OpenAI, Gemini or Claude), such as
```
GEMINI_15_API_KEY=AIxxx
GEMINI_15_MODEL=gemini-1.5-pro-latest
```

### prepare assessing LLM
- in `benchmark/conftest.py`, select desired assessing LLM
```
@pytest.fixture
def client():    
    return GeminiClient() # GptClient, GeminiClient and ClaudeClient are available
```

### run benchmark
1. Run PK summary benchmark with **Semantic** assessment
 - Prepare baseline data in `./benchmark/data/pk-summary/baseline` and target data `./benchmark/data/pk-summary/{target}`
 - Set SysVar `TARGET`
```
export TARGET={target}
```
 - run benchmark
```
poetry run pytest benchmark/test_pk_summary_benchmark_with_semantic.py
```
After benchmark completed, we can find the results in `./benchmark/result/pk-summary/{target}/result.log`

2. Run PK summary benchmark with **LLM** assessment
 - Prepare baseline data in `./benchmark/data/pk-summary/baseline` and target data `./benchmark/data/pk-summary/{target}`
 - Set SysVar `TARGET`
```
export TARGET={target}
```
 - run benchmark
```
poetry run pytest benchmark/test_pk_summary_benchmark_with_llm.py
```
After benchmark completed, we can find the results in `./benchmark/result/pk-summary/{target}/result.log`

3. Run PE benchmark with **Semantic** assemssment
 - Prepare baseline data in `./benchmark/data/pe/baseline` and target data `./benchmark/data/pe/{target}`
 - Set SysVar `TARGET`
```
export TARGET={target}
```
 - run benchmark
```
poetry run pytest benchmark/test_pe_benchmark_with_semantic.py
```
After benchmark completed, we can find the results in `./benchmark/result/pe/{target}/result.log`

4. Run PE benchmark with **LLM** assessment
 - Prepare baseline data in `./benchmark/data/pk-summary/baseline` and target data `./benchmark/data/pk-summary/{target}`
 - Set SysVar `TARGET`
```
export TARGET={target}
```
 - run benchmark
```
poetry run pytest benchmark/test_pe_benchmark_with_llm.py
```
After benchmark completed, we can find the results in `./benchmark/result/pe/{target}/result.log`

5. Run PK individual benchmark with **Semantic** assessment
 - Prepare baseline in `./benchmark/data/pk-individual/baseline` and target in `./benchmark/data/pk-individual/{target}`
 - `export TARGET={target}` then:
```
poetry run pytest benchmark/test_pk_individual_benchmark_with_semantic.py
```
Results land in `./benchmark/result/pk-individual/{target}/result.log`.

6. Run the **combined PK-PE benchmark** (drives every PK/PE pipeline in one run)
 - Place per-version curation outputs under `./benchmark/data/pk-pe/{version1}` and `./benchmark/data/pk-pe/{version2}`. Each file is named `{pmid}_{pipeline}.csv` (or `{pmid}_{pipeline}_{llm}.csv`) and is scored against its matching counterpart by `BenchmarkType` derived from the filename.
 - Columns, anchor keys, and numeric/text types per pipeline are defined in `benchmark/configs.py`; PE column aliases (e.g. `Outcome` → `Outcomes`, `Interval Low` → `Lower bound`) are normalized in `benchmark/pe_preprocess.py`.
 - Configure versions and scoring mode:
```
export VERSION1=2026-4-10        # "target" curation version (defaults: 2026-4-10)
export VERSION2=2026-4-16        # "baseline" curation version (defaults: 2026-4-16)
export SCORE_MODE=combined       # or row, column, etc. (defaults: combined)
```
 - Run:
```
poetry run pytest benchmark/test_pk_pe_benchmark_with_semantic.py
```
Results land in `./benchmark/result/pk-pe/{version1}/result.log`. The supported `BenchmarkType` values are: `pk-summary`, `pk-individual`, `pk-population-summary`, `pk-population-individual`, `pk-specimen-summary`, `pk-specimen-individual`, `pk-drug-summary`, `pk-drug-individual`, `pe-study-info`, `pe-study-outcome`.

## Streamlit UI
The Streamlit app provides an interactive workflow to retrieve papers, extract tables, and curate PK/PE/CT outputs.

1. Start the app: `poetry run streamlit run app.py`
2. In the sidebar, use `Access Article` to load content by PMID/PMCID or paste raw HTML. Only PMC-hosted articles are retrievable by PMID/PMCID.
3. In `Curation Settings`, select the article and task (PK/PE/CT), then click `Start Curation`.
4. Optional: use `One Click Curation` to automatically run multiple pipeline types on the selected article.
5. Results appear in the main pane, including `Article Preview`, `Curation Result`, `Follow-up Chat`, and `Manage Records`.

## Extract Data From Papers (CLI)
Use the scripts below for batch extraction without the UI. Outputs are written as CSV files per PMID, with error details saved alongside when applicable. Ensure `.env` is configured before running.

1. Multi-pipeline extraction with a single model:
```
poetry run python app_script.py -i 29943508 -o ./out -m gpt4o
```
2. Batch extraction from a CSV file:
```
poetry run python app_script.py -f ./data/pmids.csv -o ./out -m gemini25flash
```
3. Batch extraction with full pipeline orchestration (identification + design + curation):
```
poetry run python app_script_pmids.py -f ./data/pmids.csv -o ./out
poetry run python app_script_pmids.py -i 29943508 -o ./out
```
  - `-j / --job_id`: optional integer job ID used in the summary filename (defaults to the process PID).

### Summary CSV

`app_script_pmids.py` writes a summary file `summary_{job_id}_{input_stem}.csv` in the output directory after each pipeline completes. Columns:

| Column | Description |
|--------|-------------|
| `pmid` | PubMed ID |
| `pipeline` | Pipeline name (e.g. `PK_SUMMARY`) or `N/A` |
| `final_answer` | Outcome code (see table below) |
| `suggested_fix` | Suggested correction from the verification step, or `N/A` |

`final_answer` values:

| Value | Meaning |
|-------|---------|
| `Correct` | Verification confirmed the curated table is correct |
| `Incorrect` | Verification found errors; correction did not fully resolve them (intermediate state) |
| `MaxStepReached` | Verify/correct cycle hit the step limit while still Incorrect |
| `NoTable` | No relevant tables found in the paper for this pipeline (expected, not an error) |
| `NoIndividualData` | PK tables found but contain only summary-level data, no per-subject rows (PK individual pipeline only) |
| `Neither` | Identification step classified the paper as neither PK nor PE |
| `PipelineError` | Unhandled exception during pipeline tool execution |
| `CorrectionError` | Correction step exhausted all retries without producing a valid fix |
| `VerificationError` | Exception raised inside the verification agent |

## Helper scripts

Stand-alone utilities under `scripts/` support data prep and result management; run them from the repo root with `poetry run python scripts/<name>.py ...`.

- `prepare_htmls_by_pmids.py` — fetch paper full-text HTML for a list of PMIDs.
```
poetry run python scripts/prepare_htmls_by_pmids.py -i ./data/pmids.csv -o ./out/html -n 50 -s 0
```
  Flags: `-i/--input` CSV (PMID column or PMID in first column), `-o/--output` output folder, `-n/--number` max papers, `-s/--offset` start index.

- `add_llm_suffix.py` — append a `_{llm}` suffix to curated files so they can be scored side-by-side in `benchmark/data/pk-pe/{version}/`.
```
poetry run python scripts/add_llm_suffix.py ./benchmark/data/pk-pe/2026-4-16 qwen35 --pattern '*.csv' --recursive --dry-run
```

- `convert_md_table_to_csv.py` — convert Markdown tables in a file into CSV.

## Pipeline orchestration

The top-level orchestration (identification → design → execution → verification → correction loop) is implemented by `PKPEManager` in `extractor/agents_manager/pk_pe_manager.py`. Per-pipeline subgraphs live in `extractor/agents_manager/*_task.py` and reuse the shared `pk_pe_agenttool_task.py` graph
`START → execution_step → verification_step → (correction_step → verification_step)* → END`.
Use `PKPEManager.run(pmid)` / `runAsync(pmid)` programmatically, or `app_script_pmids.py` for batch runs over a PMID CSV.

## bump version
This package employs bump2version to bump version
```
bump2version {major, minor or patch}
```

## Curate data from literature

See `Streamlit UI` for interactive curation and `Extract Data From Papers (CLI)` for batch workflows.
