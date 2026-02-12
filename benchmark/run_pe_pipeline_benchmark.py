import argparse
import logging
import os
from pathlib import Path

from benchmark.comm_semantic import run_semantic_benchmark
from benchmark.common import ensure_target_result_directory_existed
from benchmark.constant import BASELINE, BenchmarkType, LLModelType
from extractor.agents.agent_factory import get_pipeline_llm
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_agent_tools import PEStudyOutcomeCurationTool
from extractor.database.pmid_db import PMIDDB

logger = logging.getLogger(__name__)


def _load_pmids(baseline_dir: str, pmids_arg: str | None) -> list[str]:
    if pmids_arg:
        pmids_path = Path(pmids_arg)
        if pmids_path.exists():
            return [line.strip() for line in pmids_path.read_text().splitlines() if line.strip()]
        return [pmid.strip() for pmid in pmids_arg.split(",") if pmid.strip()]

    baseline_path = Path(baseline_dir)
    pmids = []
    for csv_path in baseline_path.glob("*_baseline.csv"):
        pmids.append(csv_path.stem.split("_")[0])
    return sorted(set(pmids))


def _model_from_name(model_name: str) -> LLModelType:
    try:
        return LLModelType(model_name)
    except ValueError:
        logger.warning("Unknown model name %s, scoring will be skipped.", model_name)
        return LLModelType.UNKNOWN


def run_pipeline(
    pmids: list[str],
    target_dir: str,
    model_name: str,
):
    os.makedirs(target_dir, exist_ok=True)
    pmid_db = PMIDDB()
    llm = get_pipeline_llm()

    for pmid in pmids:
        info = extract_pmid_info_to_db(pmid, pmid_db)
        if info[0] is None:
            logger.warning("Failed to load PMID %s into the database", pmid)
            continue
        tool = PEStudyOutcomeCurationTool(
            pmid=pmid,
            llm=llm,
            pmid_db=pmid_db,
        )
        df, _ = tool.run()
        if df is None:
            logger.warning("No PE outcome data extracted for PMID %s", pmid)
            continue
        output_path = Path(target_dir, f"{pmid}_{model_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info("Wrote %s", output_path)


def run_semantic_scores(
    baseline_dir: str,
    target_dir: str,
    baseline: str,
    model_name: str,
):
    model = _model_from_name(model_name)
    if model == LLModelType.UNKNOWN:
        return

    dataset = {}
    for baseline_path in Path(baseline_dir).glob("*_baseline.csv"):
        pmid = baseline_path.stem.split("_")[0]
        target_path = Path(target_dir, f"{pmid}_{model_name}.csv")
        if not target_path.exists():
            logger.warning("Missing target for PMID %s at %s", pmid, target_path)
            continue
        dataset[pmid] = {
            BASELINE: str(baseline_path),
            model_name: str(target_path),
        }

    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=os.path.basename(target_dir),
        benchmark_type=BenchmarkType.PE,
    )
    result_path = os.path.join(result_dir, "result.log")
    run_semantic_benchmark(
        dataset=dataset,
        benchmark_type=BenchmarkType.PE,
        model=model,
        result_file=result_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PE study outcome pipeline and semantic benchmark.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target folder name under ./benchmark/data/pe",
    )
    parser.add_argument(
        "--baseline-dir",
        default="./benchmark/data/pe/baseline",
        help="Baseline directory with *_baseline.csv files.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("BENCHMARK_MODEL_NAME", "gpt-oss"),
        help="Model tag for output filenames (e.g., gpt4o, gemini15, gpt-oss).",
    )
    parser.add_argument(
        "--pmids",
        default=None,
        help="Comma-separated PMIDs or a file path with one PMID per line.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip semantic scoring and only generate outputs.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    target_dir = os.path.join("./benchmark/data/pe", args.target)
    pmids = _load_pmids(args.baseline_dir, args.pmids)
    if not pmids:
        logger.error("No PMIDs found.")
        return 1

    run_pipeline(pmids, target_dir, args.model_name)
    if not args.no_score:
        run_semantic_scores(
            baseline_dir=args.baseline_dir,
            target_dir=target_dir,
            baseline=BASELINE,
            model_name=args.model_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
