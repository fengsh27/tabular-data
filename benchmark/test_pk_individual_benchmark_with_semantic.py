import pytest
import os
from dotenv import load_dotenv
import logging

from benchmark.comm_semantic import run_semantic_benchmark
from benchmark.common import (
    ensure_target_result_directory_existed,
    prepare_single_pipeline_dataset_for_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)

load_dotenv()

logger = logging.getLogger(__name__)

baseline = os.environ.get("BASELINE", BASELINE)
target = os.environ.get("TARGET", "2026-6-12")
baseline_dir = os.path.join("./benchmark/data/pk-individual", baseline)
target_dir = os.path.join("./benchmark/data/pk-individual", target)
score_mode = os.environ.get("SCORE_MODE", "combined")

MODELS = [
    LLModelType.GPT4O,
    LLModelType.GPTOSS,
    LLModelType.QWEN3,
    LLModelType.CODEX,
    LLModelType.GPT54,
    LLModelType.GEMMA4,
    LLModelType.QWEN35,
    LLModelType.QWEN36SKILL,
]


@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_single_pipeline_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )


@pytest.fixture(scope="module")
def result_path():
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )
    return os.path.join(result_dir, "result.log")


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.value)
def test_semantic_benchmark(prepared_dataset, result_path, model):
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        model=model,
        result_file=result_path,
        score_mode=score_mode,
    )
