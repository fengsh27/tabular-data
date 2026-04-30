import pytest
import os
from dotenv import load_dotenv
import logging

from benchmark.common import (
    ensure_target_result_directory_existed,
    prepare_multiple_pipeline_dataset_for_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)
from benchmark.comm_semantic import run_semantic_benchmark

load_dotenv()

logger = logging.getLogger(__name__)

version1 = os.environ.get("VERSION1", "2026-4-10")
version2 = os.environ.get("VERSION2", "2026-4-16")
version1_dir = os.path.join("./benchmark/data/pk-pe", version1)
version2_dir = os.path.join("./benchmark/data/pk-pe", version2)
score_mode = os.environ.get("SCORE_MODE", "combined")

MODELS = [
    LLModelType.GPT4O,
    LLModelType.GPTOSS,
    LLModelType.QWEN3,
    LLModelType.CODEX,
    LLModelType.GPT54,
    LLModelType.GEMMA4,
    LLModelType.QWEN35,
]

@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_multiple_pipeline_dataset_for_benchmark(
        version1_dir=version1_dir,
        version2_dir=version2_dir,
    )

@pytest.fixture(scope="module")
def result_path():
    result_dir = ensure_target_result_directory_existed(
        baseline=version2,
        target=version1,
        benchmark_type=BenchmarkType.PK_PE,
    )
    return os.path.join(result_dir, "result.log")

def test_pk_pe_semantic_benchmark(prepared_dataset, result_path):
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_PE,
        model=LLModelType.UNKNOWN,
        result_file=result_path,
        score_mode=score_mode,
    )
        