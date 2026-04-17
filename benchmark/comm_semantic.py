from typing import Literal, Tuple, Union
import pandas as pd
from pathlib import Path
import logging

from .configs import BenchmarkConfig, get_benchmark_config
from .constant import BASELINE, BenchmarkType, LLModelType
from .evaluate import TablesEvaluator, TablesSeparateEvaluator

logger = logging.getLogger(__name__)


def write_semantic_score(output_fn: str, model: str, pmid: str, score: int | Tuple[int, int]):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{model}, {pmid}, {str(score)}\n")


def _build_evaluator(
    config: BenchmarkConfig,
    score_mode: Literal["combined", "separate"] | None,
) -> TablesEvaluator:
    cls = TablesEvaluator if score_mode == "combined" else TablesSeparateEvaluator
    return cls(
        rating_cols=config.rating_cols,
        anchor_cols=config.anchor_cols,
        columns_type=config.columns_type,
    )


def run_semantic_benchmark(
    dataset: dict,
    benchmark_type: Union[BenchmarkType.PE, BenchmarkType.PK_SUMMARY, BenchmarkType.PK_INDIVIDUAL],
    model: LLModelType,
    result_file: str,
    score_mode: Literal["combined", "separate"] | None = "combined",
):
    config = get_benchmark_config(benchmark_type)
    evaluator = _build_evaluator(config, score_mode)

    for id in dataset:
        the_dict = dataset[id]
        if model.value not in the_dict:
            continue
        logger.info(f"Processing {id} with model {model.value}")
        baseline = the_dict[BASELINE]
        target = the_dict[model.value]

        target_path = Path(target)
        content = target_path.read_text()
        if content.strip().strip('"') == "":
            write_semantic_score(
                output_fn=result_file,
                model=model.value,
                pmid=id,
                score=0,
            )
            continue

        df_baseline = pd.read_csv(baseline)
        df_target = config.preprocess(target)
        score = evaluator.compare_tables(df_baseline, df_target)
        write_semantic_score(
            output_fn=result_file,
            model=model.value,
            pmid=id,
            score=score,
        )
