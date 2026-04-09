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

## bump version
This package employs bump2version to bump version
```
bump2version {major, minor or patch}
```

## Curate data from literature

See `Streamlit UI` for interactive curation and `Extract Data From Papers (CLI)` for batch workflows.
