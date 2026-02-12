import pytest
import os
from dotenv import load_dotenv
import logging

from benchmark.comm_semantic import run_semantic_benchmark
from benchmark.common import (
    ensure_target_result_directory_existed, 
    prepare_dataset_for_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)

load_dotenv()

logger = logging.getLogger(__name__)

baseline = os.environ.get("BASELINE", BASELINE)
target = os.environ.get("TARGET", "2025-10-25-mas")
baseline_dir = os.path.join("./benchmark/data/pk-individual", baseline)
target_dir = os.path.join("./benchmark/data/pk-individual", target)

@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )

def test_gpt_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        model=LLModelType.GPT4O,
        result_file=result_path,
    )

def test_gpt_oss_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        model=LLModelType.GPTOSS,
        result_file=result_path,
    )

def test_qwen3_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        model=LLModelType.QWEN3,
        result_file=result_path,
    )

def test_codex_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_INDIVIDUAL,
        model=LLModelType.CODEX,
        result_file=result_path,
    )





