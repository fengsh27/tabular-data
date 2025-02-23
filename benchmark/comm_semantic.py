
from typing import Union
import pandas as pd
from .constant import BASELINE, BenchmarkType, LLModelType
from .pk_preprocess import (
    preprocess_table as pk_preprocess_table,
)
from .pe_preprocess import (
    preprocess_table as pe_proprocess_table,
)
from .pk_summary_benchmark_with_semantic import (
    pk_summary_evaluate_dataframe
)
from .pe_benchmark_with_semantic import (
    pe_summary_evaluate_dataframe
)

def write_semantic_score(output_fn: str, model: str, pmid: str, score: int):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{model}, {pmid}, {score}\n")

def run_semantic_benchmark(
    dataset: dict,
    benchmark_type: Union[BenchmarkType.PE, BenchmarkType.PK_SUMMARY],
    model: Union[LLModelType.GEMINI15, LLModelType.GPT4O],
    result_file: str,
):
    preprocess_table, evaluate_dataframe = (pk_preprocess_table, pk_summary_evaluate_dataframe) \
        if benchmark_type == BenchmarkType.PK_SUMMARY \
        else (pe_proprocess_table, pe_summary_evaluate_dataframe)
    for id in dataset:
        the_dict = dataset[id]
        if not model.value in the_dict:
            continue
        baseline = the_dict[BASELINE]
        target = the_dict[model.value]
        df_baseline = pd.read_csv(baseline)
        df_target = preprocess_table(target)
        score = evaluate_dataframe(
            df_baseline=df_baseline,
            df_pk_summary=df_target,
        )        
        write_semantic_score(
            output_fn=result_file,
            model=model.value,
            pmid=id,
            score=score,
        )
