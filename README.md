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

## bump version
This package employs bump2version to bump version
```
bump2version {major, minor or patch}
```

