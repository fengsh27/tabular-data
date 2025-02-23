import pytest
import os
import logging

from benchmark.comm_llm import run_llm_benchmark

from .common import (
    ensure_target_result_directory_existed, 
    prepare_dataset_for_benchmark,
)
from .constant import (
    BASELINE,
    BenchmarkType,
    LLModelType
)

logger = logging.getLogger(__name__)

@pytest.mark.skip("just for test the feasible of claude api")
def test_claude(client):
    msg, useage = client.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system="Respond only in Yoda-speak.",
        messages=[
            {"role": "user", "content": "How are you today?"}
        ]
    )
    
    print(msg)
    assert msg is not None

target = os.environ.get("TARGET", "yichuan/0213_prompt_chain")
baseline_dir = os.path.join("./benchmark/data/pk-summary", BASELINE)
target_dir = os.path.join("./benchmark/data/pk-summary", target)
result_dir = os.path.join("./benchmark/result/pk-summary", target)

@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )

@pytest.fixture(scope="module")
def ensured_result_path():
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    return os.path.join(result_dir, "result.log")

def test_gpt4o_benchmark(client, prepared_dataset, ensured_result_path):
    run_llm_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_SUMMARY,
        model=LLModelType.GPT4O,
        result_file=ensured_result_path,
        client=client,
    )
    
def test_gemini15_benchmark(client, prepared_dataset, ensured_result_path):
    run_llm_benchmark(
        dataset=prepared_dataset,
        benchmark_type=BenchmarkType.PK_SUMMARY,
        model=LLModelType.GEMINI15,
        result_file=ensured_result_path,
        client=client,
    )
    
