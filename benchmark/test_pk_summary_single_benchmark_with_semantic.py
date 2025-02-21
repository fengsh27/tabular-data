
import pytest
import os
from dotenv import load_dotenv
import pandas as pd
import logging

from benchmark.common import (
    walk_benchmark_data_directory,
    ensure_target_result_directory_existed,
    write_semantic_score,
)
from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)
from benchmark.pk_summary_benchmark_with_semantic import (
    pk_summary_evaluate_csvfile,
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

dataset = {}

@pytest.fixture(scope="module")
def setup_module():
    baseline_type, baseline_pmids = walk_benchmark_data_directory(baseline_dir)
    assert baseline_type == BenchmarkType.PK_SUMMARY_BASELINE
    target_type, target_pmids = walk_benchmark_data_directory(target_dir)
    assert target_type == BenchmarkType.PK_SUMMARY

    for pmid in baseline_pmids:
        id, fn, _ = pmid
        dataset[id] = {"baseline": fn}
    for pmid in target_pmids:
        id, fn, model = pmid
        if model == LLModelType.UNKNOWN:
            continue
        if not id in dataset:
            logger.error(f"no baseline for pmid {id}")
            continue
        dataset[id][model.value] = fn

@pytest.fixture
def single_pmid():
    return "29943508"

def test_single_pmid_gpt4o_benchmark(setup_module, single_pmid):
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    result_path = os.path.join(result_dir, "result.log")
    for id in dataset:
        if id != single_pmid:
            continue
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

def test_single_pmid_gemini15_benchmark(setup_module, single_pmid):
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    result_path = os.path.join(result_dir, "result.log")
    for id in dataset:
        if id != single_pmid:
            continue
        the_dict = dataset[id]
        if not LLModelType.GEMINI15.value in the_dict:
            # no gpt-4o table
            continue
        baseline = the_dict['baseline']
        gpt4o = the_dict[LLModelType.GEMINI15.value]
        df_baseline = pd.read_csv(baseline)
        df_target = preprocess_table(gpt4o)
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
        
