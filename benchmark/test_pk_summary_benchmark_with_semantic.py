
import pytest
import os
from dotenv import load_dotenv
import pandas as pd
import logging

from benchmark.common import (
    run_semantic_benchmark,
    ensure_target_result_directory_existed,
    prepare_dataset_for_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)

logger = logging.getLogger(__name__)

load_dotenv()

"""
This benchmark is to run semantic benchmark on pk summary direcotry './benchmark/data/pk-summary/{target}', 
the result will be written to './benchmark/result/pk-summary/{target}'

The files in target directory should adhere to the following naming convention:
{pmid}_{model}.csv, such as
12345678_gpt4o.csv
12345678_gemini15.csv
"""

target = os.environ.get("TARGET", "yichuan/0213_prompt_chain")
baseline_dir = os.path.join("./benchmark/data/pk-summary", BASELINE)
target_dir = os.path.join("./benchmark/data/pk-summary", target)
result_dir = os.path.join("./benchmark/result/pk-summary", target)

@pytest.fixture(scope="module")
def prepared_dataset():
    dataset = prepare_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    return dataset

def test_gpt4o_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_SUMMARY,
        model=LLModelType.GPT4O,
        result_file=result_path,
    )

def test_gemini_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_SUMMARY,
        model=LLModelType.GEMINI15,
        result_file=result_path,
    )
        
