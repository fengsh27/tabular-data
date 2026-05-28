from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
from datetime import datetime
import os
from os import path
import logging
from enum import Enum
from dataclasses import dataclass

from .constant import (
    BenchmarkType,
    LLModelType,
)


logger = logging.getLogger(__name__)
    
@dataclass
class MultiplePipelineDatasetItem:
    pmid: str
    version1: dict[str, str]
    version2: dict[str, str]
    version1_model: LLModelType | None
    version2_model: LLModelType | None


def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")

class LLMClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self, systemp_prompts: str, user_prompts: str):
        """query"""

def _check_pipeline_name_in_file_name(file_name: str) -> bool:
    """
    Check if the pipeline name is in the file name.

    file name protocol: {pmid}_{pipeline}_{model}.csv, where pipeline is one of the following:
    - pk_summary
    - pe_study_outcome
    - pe_study_info
    - pk_individual
    - pk_drug_summary
    - pk_drug_individual
    - pk_specimen_individual
    - pk_specimen_summary
    - pk_population_summary
    - pk_population_individual
    
    Args:
        file_name str: file name
    
    Returns:
        bool: True if the pipeline name is in the file name, False otherwise
    """
    arr = file_name.split("_")
    if len(arr) < 3:
        return False

    pipeline = "-".join(arr[1:-1])
    return pipeline in BenchmarkType   
        

def _get_pmid_pipeline_and_model(fn: str) -> tuple[str, str | None, LLModelType | None] | None:
    """
    This function is to identify the pmid and llm model based on file name, which must adhere to the following naming convention:
    {pmid}_{pipeline}_{model}.csv

    Examples of valid file names:
    - 16143486_pk_summary_gpt4o.csv
    - 16143486_pe_study_outcome_gpt4o.csv
    - 16143486_pk_individual_gpt4o.csv

    Args:
        fn str: file name

    Returns:
        tuple[str, str | None, LLModelType | None] | None:
            - pmid
            - pipeline
            - model
    """
    if len(fn) == 0:
        return None
    bn = os.path.splitext(fn)[0]
    arr = bn.split("_")
    if len(arr) < 3:
        logger.error(f"Invalid file name: {fn}")
        return None
    pmid = arr[0]
    pipeline = "_".join(arr[1:-1])
    model = arr[-1]
    try:
        model_type = LLModelType(model)
        return pmid, pipeline, model_type
    except ValueError:
        logger.error(f"Unknown llm: {model} in file: {fn}")
        return pmid, pipeline, LLModelType.UNKNOWN

def _get_pmid_and_llmodel(fn: str) -> tuple[str, LLModelType | None] | None:
    """
    This function is to identify the pmid and llm model based on file name, which must adhere to the following naming convention:
    {pmid}_{model}.csv

    Examples of valid file names:
    - 16143486_gpt40.csv
    - 16143486_baseline.csv

    Args:
    fn str: file name

    Returns:
    a list of tuples,
        - pmid
        - llm type (gpt4o, gemini15 or baseline)
    """
    if len(fn) == 0:
        return None
    bn = os.path.splitext(fn)[0]
    arr = bn.split("_")

    pmid = arr[0]
    try:
        model_type = LLModelType(arr[-1])
        return pmid, model_type
    except ValueError:
        logger.error(f"Unknown llm: {arr[-1]} in file: {fn}")
        return pmid, LLModelType.UNKNOWN


def walk_benchmark_data_directory(
    dir_path: str,
) -> list[tuple[str, str, LLModelType, str | None]]:
    """
    Walk `dir_path` and return (pmid, file_path, model, pipeline) tuples for each
    CSV whose filename matches `{pmid}_{model}.csv`. Files with an
    unknown model suffix are skipped.
    """
    pmids: list[tuple[str, str, LLModelType, str | None]] = []
    for r, _, files in os.walk(dir_path):
        for f in files:
            if _check_pipeline_name_in_file_name(f):
                parsed = _get_pmid_pipeline_and_model(f)
                if parsed is None:
                    continue
                pmid, pipeline, model = parsed
                if model == LLModelType.UNKNOWN:
                    continue
                pmids.append((pmid, path.join(r, f), model, pipeline))
                continue

            parsed = _get_pmid_and_llmodel(f)
            if parsed is None:
                continue
            pmid, model = parsed
            if model == LLModelType.UNKNOWN:
                continue
            pmids.append((pmid, path.join(r, f), model, None))
    return pmids

def prepare_multiple_pipeline_dataset_for_benchmark(
    version1_dir: str,
    version2_dir: str,
) -> dict[str, MultiplePipelineDatasetItem]:
    """
    Walk `version1_dir` and `version2_dir` and build::

        {
            "{pmid}": MultiplePipelineDatasetItem{
                "version1": {
                    "{pipeline}": "{pmid_pipeline_model_path}",
                    ...
                },
                "version2": {
                    "{pipeline}": "{pmid_pipeline_model_path}",
                    ...
                },
                "version1_model": "{model}",
                "version2_model": "{model}",
                ...
            },
            ...
        }
    """
    dataset: dict = {}
    for pmid, fn, model, pipeline in walk_benchmark_data_directory(version1_dir):
        if pmid not in dataset:
            dataset[pmid] = MultiplePipelineDatasetItem(
                pmid=pmid,
                version1={pipeline: fn},
                version2={},
                version1_model=model,
                version2_model=None,
            )
            continue
        dataset[pmid].version1[pipeline] = fn
        dataset[pmid].version1_model = model
    for pmid, fn, model, pipeline in walk_benchmark_data_directory(version2_dir):
        if pmid not in dataset:
            dataset[pmid] = MultiplePipelineDatasetItem(
                pmid=pmid,
                version1={},
                version2={pipeline: fn},
                version1_model=None,
                version2_model=model,
            )
            continue
        dataset[pmid].version2[pipeline] = fn
        dataset[pmid].version2_model = model

    return dataset

def prepare_single_pipeline_dataset_for_benchmark(
    baseline_dir: str,
    target_dir: str,
    benchmark_type: BenchmarkType,
) -> dict[str, dict[str, str]]:
    """
    Walk `baseline_dir` and `target_dir` and build::

        {
            "{pmid}": {
                "baseline": "{pmid_baseline_path}",
                "{model}": "{pmid_model_path}",
                ...
            },
            ...
        }

    `benchmark_type` is kept for API symmetry with callers; the function
    no longer verifies it against the directory path.
    """
    del benchmark_type  # retained for call-site symmetry
    dataset: dict = {}
    for pmid, fn, _ in walk_benchmark_data_directory(baseline_dir):
        dataset[pmid] = {"baseline": fn}
    for pmid, fn, model in walk_benchmark_data_directory(target_dir):
        if pmid not in dataset:
            logger.error(f"no baseline for pmid {pmid}")
            continue
        dataset[pmid][model.value] = fn

    return dataset


def ensure_target_result_directory_existed(
    baseline: str,
    target: str,
    benchmark_type: BenchmarkType,
):
    baseline_name = baseline.replace("/", "_")
    baseline_name = baseline_name.replace("\\", "_")
    target_name = target.replace("/", "_")
    target_name = target_name.replace("\\", "_")
    # result_dir = os.path.join("./benchmark/result/pk-summary", f"{target_name}-{baseline_name}")
    dir_path = path.join(
        "./benchmark/result", benchmark_type.value, f"{target_name}-{baseline_name}"
    )
    if os.path.isdir(dir_path):
        return dir_path
    try:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    except Exception as e:
        logger.error(e)
        raise e

class ColumnType(Enum):
    Text = "text"
    # Integer = "int"
    Numeric = "numeric"
    


