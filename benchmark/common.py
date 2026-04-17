from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
from datetime import datetime
import os
from os import path
import logging
from enum import Enum

from .constant import (
    BenchmarkType,
    LLModelType,
)


logger = logging.getLogger(__name__)


def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")

class LLMClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self, systemp_prompts: str, user_prompts: str):
        """query"""


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
) -> list[tuple[str, str, LLModelType]]:
    """
    Walk `dir_path` and return (pmid, file_path, model) tuples for each
    CSV whose filename matches `{pmid}_{model}.csv`. Files with an
    unknown model suffix are skipped.
    """
    pmids: list[tuple[str, str, LLModelType]] = []
    for r, _, files in os.walk(dir_path):
        for f in files:
            parsed = _get_pmid_and_llmodel(f)
            if parsed is None:
                continue
            pmid, model = parsed
            if model == LLModelType.UNKNOWN:
                continue
            pmids.append((pmid, path.join(r, f), model))
    return pmids


def prepare_dataset_for_benchmark(
    baseline_dir: str,
    target_dir: str,
    benchmark_type: Union[
        BenchmarkType.PK_SUMMARY,
        BenchmarkType.PE,
        BenchmarkType.PK_INDIVIDUAL,
    ],
):
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
    


