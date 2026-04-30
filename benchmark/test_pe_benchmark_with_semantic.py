import pytest
import os
from dotenv import load_dotenv
import logging

from benchmark.common import (
    ensure_target_result_directory_existed,
    prepare_single_pipeline_dataset_for_benchmark,
)
from benchmark.comm_semantic import (
    run_semantic_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)

logger = logging.getLogger(__name__)

load_dotenv()

"""
This benchmark is to run semantic benchmark on pe direcotry './benchmark/data/pe/{target}',
the result will be written to './benchmark/result/pe/{target}'

The files in target directory should adhere to the following naming convention:
{pmid}_{model}.csv, such as
12345678_gpt4o.csv
12345678_gemini15.csv
"""

baseline = os.environ.get("BASELINE", BASELINE)
target = os.environ.get("TARGET", "2024-08-12")
baseline_dir = os.path.join("./benchmark/data/pe", baseline)
target_dir = os.path.join("./benchmark/data/pe", target)
score_mode = os.environ.get("SCORE_MODE", "combined")

MODELS = [
    LLModelType.GPT4O,
    LLModelType.GEMINI15,
]


@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_single_pipeline_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PE,
    )


@pytest.fixture(scope="module")
def result_path():
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=BenchmarkType.PE,
    )
    return os.path.join(result_dir, "result.log")


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.value)
def test_semantic_benchmark(prepared_dataset, result_path, model):
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PE,
        model=model,
        result_file=result_path,
        score_mode=score_mode,
    )
