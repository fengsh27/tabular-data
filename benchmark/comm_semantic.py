from typing import Literal, Tuple, Union
import pandas as pd
from pathlib import Path
import logging

from .configs import BenchmarkConfig, get_benchmark_config
from .constant import BASELINE, BenchmarkType, LLModelType
from .evaluate import TablesEvaluator, TablesSeparateEvaluator
from .common import MultiplePipelineDatasetItem

logger = logging.getLogger(__name__)


def write_semantic_score(output_fn: str, model: str, pmid: str, score: int | Tuple[int, int]):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{model}, {pmid}, {str(score)}\n")

def write_semantic_score_header_for_multiple_pipeline(output_fn: str):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"pmid, pipeline, model1, model2, score\n")

def write_semantic_score_for_multiple_pipeline(
    output_fn: str, 
    pipeline: str,
    model1: str, 
    model2: str, 
    pmid: str, 
    score: int | Tuple[int, int]
):
    with open(output_fn, "a+") as fobj:
        fobj.write(f"{pmid}, {pipeline}, {model1}, {model2}, {str(score)}\n")
        

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

def _run_semantic_benchmark_for_single_pipeline(
    dataset: dict,
    benchmark_type: BenchmarkType,
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

def _get_benchmark_type(pipeline: str) -> BenchmarkType:
    pipeline = pipeline.replace('_', '-')
    try:
        return BenchmarkType(pipeline)
    except ValueError:
        raise ValueError(f"Unknown benchmark type: {pipeline}")

def _run_semantic_benchmark_for_multiple_pipeline(
    dataset: dict[str, MultiplePipelineDatasetItem],
    result_file: str,
    score_mode: Literal["combined", "separate"] | None = "combined",
):
    write_semantic_score_header_for_multiple_pipeline(result_file)
    for id in dataset:
        the_dict = dataset[id]
        version1_dict: dict[str, str] = the_dict.version1
        version2_dict: dict[str, str] = the_dict.version2
        version1_model = the_dict.version1_model
        version2_model = the_dict.version2_model

        score_dict: dict[str, Tuple[bool, bool, float]] = {}
        for pipeline in version1_dict:
            if pipeline not in version2_dict:
                score_dict[pipeline] = (True, False, 0.0)
                write_semantic_score_for_multiple_pipeline(
                    output_fn=result_file,
                    pipeline=pipeline,
                    model1=version1_model,
                    model2=version2_model,
                    pmid=id,
                    score=(True, False, 0.0),
                )
                continue
            benchmark_type = _get_benchmark_type(pipeline)
            config = get_benchmark_config(benchmark_type)
            evaluator = _build_evaluator(config, score_mode)
            df_version1 = config.preprocess(version1_dict[pipeline])
            df_version2 = config.preprocess(version2_dict[pipeline])
            score = evaluator.compare_tables(df_version1, df_version2)
            score_dict[pipeline] = (True, True, score)
            write_semantic_score_for_multiple_pipeline(
                output_fn=result_file,
                pipeline=pipeline,
                model1=version1_model,
                model2=version2_model,
                pmid=id,
                score=score,
            )
        
        for pipeline in version2_dict:
            if pipeline not in version1_dict:
                score_dict[pipeline] = (False, True, 0.0)
                write_semantic_score_for_multiple_pipeline(
                    output_fn=result_file,
                    pipeline=pipeline,
                    model1=version1_model,
                    model2=version2_model,
                    pmid=id,
                    score=(False, True, 0.0),
                )
                continue
            
        return score_dict

def run_semantic_benchmark(
    dataset: dict,
    benchmark_type: BenchmarkType,
    model: LLModelType,
    result_file: str,
    score_mode: Literal["combined", "separate"] | None = "combined",
):
    if benchmark_type != BenchmarkType.PK_PE:
        _run_semantic_benchmark_for_single_pipeline(
            dataset=dataset,
            benchmark_type=benchmark_type,
            model=model,
            result_file=result_file,
            score_mode=score_mode,
        )
    else:
        _run_semantic_benchmark_for_multiple_pipeline(
            dataset=dataset,
            result_file=result_file,
            score_mode=score_mode,
        )
