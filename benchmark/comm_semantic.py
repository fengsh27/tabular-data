from typing import Literal, Tuple, Union
import pandas as pd
from pathlib import Path
import logging

from .constant import BASELINE, BenchmarkType, LLModelType
from .pk_preprocess import (
    preprocess_pk_individual_table,
    preprocess_pk_summary_table,
)
from .pe_preprocess import (
    preprocess_table as pe_proprocess_table,
)
from .pk_summary_benchmark_with_semantic import (
    pk_summary_evaluate_dataframe,
)
from .pe_benchmark_with_semantic import pe_summary_evaluate_dataframe
from .pk_individual_benchmark_with_semantic import pk_individual_evaluate_dataframe

logger = logging.getLogger(__name__)

def write_semantic_score(output_fn: str, model: str, pmid: str, score: int | Tuple[int, int]):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{model}, {pmid}, {str(score)}\n")

def get_preprocess_and_evaluate_function(benchmark_type: BenchmarkType):
    if benchmark_type == BenchmarkType.PK_SUMMARY:
        return preprocess_pk_summary_table, pk_summary_evaluate_dataframe
    elif benchmark_type == BenchmarkType.PE:
        return pe_proprocess_table, pe_summary_evaluate_dataframe
    elif benchmark_type == BenchmarkType.PK_INDIVIDUAL:
        return preprocess_pk_individual_table, pk_individual_evaluate_dataframe  # pyright: ignore[reportUndefinedVariable]
    else:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

def run_semantic_benchmark(
    dataset: dict,
    benchmark_type: Union[BenchmarkType.PE, BenchmarkType.PK_SUMMARY],
    model: Union[
        LLModelType.GEMINI15, 
        LLModelType.GPT4O, 
        LLModelType.GEMINI25FLASH,
        LLModelType.GEMINI25FLASHLITE,
        LLModelType.METALLAMA4,
        LLModelType.SONNET4,
    ],
    result_file: str,
    score_mode: Literal["combined", "separate"] | None = "combined",
):
    preprocess_table, evaluate_dataframe = get_preprocess_and_evaluate_function(benchmark_type)
    for id in dataset:
        the_dict = dataset[id]
        if model.value not in the_dict:
            continue
        logger.info(f"Processing {id} with model {model.value}")
        baseline = the_dict[BASELINE]
        target = the_dict[model.value]
        # check if target is empty
        target_path = Path(target)
        content = target_path.read_text()
        if content.strip().strip('"') == "":
            score = 0
            write_semantic_score(
                output_fn=result_file,
                model=model.value,
                pmid=id,
                score=score,
            )
            continue

        df_baseline = pd.read_csv(baseline)
        df_target = preprocess_table(target)
        score = evaluate_dataframe(
            df_baseline=df_baseline,
            df=df_target,
            score_mode=score_mode,
        )
        write_semantic_score(
            output_fn=result_file,
            model=model.value,
            pmid=id,
            score=score,
        )
