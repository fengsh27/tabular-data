# tabula-data

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
- in `benchmark/test_benchmark_with_llm.py`, select desired assessing LLM
```
@pytest.fixture
def client():    
    return GeminiClient() # GptClient, GeminiClient and ClaudeClient are available
```

### run benchmark
We can run benchmark either by execute `poetry run pytest benchmark/test_benchmark_with_llm.py` directly or execute `pytest benchmark/test_benchmark_with_llm.py` in poetry shell.

After benchmark completed, we can find the results in `benchmark-result.log`

## bump version
This package employs bump2version to bupm version
```
bump2version --current-version {current version} {major, minor or patch} version.py pyproject.toml
```

