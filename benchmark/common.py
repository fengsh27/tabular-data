from typing import Any, Callable, List, Optional
from datetime import datetime
import os
from os import path
import logging

from benchmark.constant import (
    BASELINE,
    BenchmarkType,
    LLModelType,
)

logger = logging.getLogger(__name__)

def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")


class ResponderWithRetries:
    """
    Raise request to LLM with 3 retries
    """

    def __init__(self, runnable_func: Callable, retry: int=3):
        """
        Args:
        runnable_func: function to be executed, if failed, we will retry
        """
        self.runnable = runnable_func
        self.retry = retry

    def respond(self, args: Optional[List[Any]]=None):
        """
        """
        response = []
        for attempt in range(self.retry):
            try:
                response = self.runnable() if args == None else self.runnable(args)
                return response
            except Exception as e:
                print(str(e))
        return response
    
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
    arr = bn.split('_')
    
    pmid = arr[0]
    try:
        model_type = LLModelType(arr[-1])
        return pmid, model_type
    except ValueError:
        logger.error(f"Unknown llm: {arr[-1]} in file: {fn}")
        return pmid, LLModelType.UNKNOWN   

def _get_benchmark_type(dir_path: str) -> BenchmarkType | None:
    """
    This function is to identify benchmark type from directory path, which must adhere to the following naming convention:
    ./benchmark/data/{benchmark_type}/{target}

    Examples of valid directory path:
    - ./benchmark/data/pk-summary/2025-02-20
    - ./benchmark/data/pk-summary/baseline

    Args:
    dir_path str: directory path

    Returns:
    benchmark type
    """
    if len(dir_path) == 0:
        return None
    dir_path = dir_path.replace("\\", "/")
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]
    if "/"+BenchmarkType.PK_SUMMARY.value+"/" in dir_path:
        basename = os.path.basename(dir_path)
        if basename == BASELINE:
            return BenchmarkType.PK_SUMMARY_BASELINE
        else:
            return BenchmarkType.PK_SUMMARY
    if "/"+BenchmarkType.PE.value+"/" in dir_path:
        basename = os.path.basename(dir_path)
        if basename == BASELINE:
            return BenchmarkType.PE_BASELINE
        else:
            return BenchmarkType.PE
        
    return BenchmarkType.UNKNOWN


def walk_benchmark_data_directory(dir_path: str) -> list[str, str, LLModelType]:
    """
    Walks through the directory `dir_path` to identify all PMID table files (.csv) and their associated benchmark type.

    The benchmark type is determined by {dir_path}, which must adhere to the following naming convention:
    ./benchmark/data/{benchmark_type}/{target}
    such as:
    ./benchmark/data/pk-summary/baseline
    ./benchmark/data/pk-summary/2025-02-20

    The pmid and llm model are determined based on the file name, which must adhere to the following naming convention:
    {pmid}_{model}.csv

    Examples of valid file names:
    - 16143486_gpt40.csv
    - 16143486_baseline.csv

    Supported benchmark types:
    - pk-summary
    - pe
    - pk-summary-baseline
    - pe-baseline
    - unknown (if the benchmark type cannot be determined)

    Args:
    dir_path str: folder path

    Returns:
    A list of tuples, where each tuple contains:
        - benchmark type
        - a list of (PMID, file_path, model)

    Raises:
    ValueError: If the directory `dir_path` does not exist or is inaccessible.
    """
    benchmark_type = _get_benchmark_type(dir_path)
    try:
        pmids = []
        for r, _, files in os.walk(dir_path):
            if len(files) == 0:
                return benchmark_type
            for f in files:
                pmid, model = _get_pmid_and_llmodel(f)                
                pmids.append((pmid, path.join(r, f), model))
        
        return benchmark_type, pmids
    except Exception as e:
        print(e)
        raise e

def ensure_target_result_directory_existed(target: str, benchmark_type: BenchmarkType):
    dir_path = path.join("./benchmark/result", benchmark_type.value, target)
    if os.path.isdir(dir_path):
        return dir_path
    try:
        os.mkdir(dir_path)
        return dir_path
    except Exception as e:
        logger.error(e)
        raise e

def write_semantic_score(output_fn: str, pmid: str, score: int):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{pmid}, {score}\n")