
import pytest
import os
from dotenv import load_dotenv
import pandas as pd
import logging

from benchmark.common import (
    walk_benchmark_data_directory,
    ensure_target_result_directory_existed,
    write_semantic_score,
    prepare_dataset_for_benchmark,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)
from benchmark.pk_summary_benchmark_with_semantic import (
    pk_summary_evaluate_dataframe
)
from benchmark.pk_preprocess import (
    preprocess_table
)

logger = logging.getLogger(__name__)

load_dotenv()

"""
This benchmark is to run semantic benchmark on direcotry './benchmark/data/pk-summary/{target}', 
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
    dataset = prepared_dataset
    for id in dataset:
        the_dict = dataset[id]
        if not LLModelType.GPT4O.value in the_dict:
            # no gpt-4o table
            continue
        baseline = the_dict['baseline']
        gpt4o = the_dict[LLModelType.GPT4O.value]
        df_baseline = pd.read_csv(baseline)
        df_target = preprocess_table(gpt4o)
        score = pk_summary_evaluate_dataframe(
            df_baseline=df_baseline,
            df_pk_summary=df_target,
        )        
        write_semantic_score(
            output_fn=result_path,
            model=LLModelType.GPT4O.value,
            pmid=id,
            score=score,
        )

def test_gemini_benchmark(prepared_dataset):
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    result_path = os.path.join(result_dir, "result.log")
    dataset = prepared_dataset
    for id in dataset:
        the_dict = dataset[id]
        if not LLModelType.GEMINI15.value in the_dict:
            # no gpt-4o table
            continue
        baseline = the_dict['baseline']
        gemini15 = the_dict[LLModelType.GEMINI15.value]
        df_baseline = pd.read_csv(baseline)
        df_target = preprocess_table(gemini15)
        score = pk_summary_evaluate_dataframe(
            df_baseline=df_baseline,
            df_pk_summary=df_target,
        )        
        write_semantic_score(
            output_fn=result_path,
            model=LLModelType.GEMINI15.value,
            pmid=id,
            score=score,
        )
        
