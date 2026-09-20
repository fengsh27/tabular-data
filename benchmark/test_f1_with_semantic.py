"""Row-level precision / recall / F1 for the curation benchmark.

`TablesEvaluator.compare_tables` (benchmark/evaluate.py) returns one blended 0-100
score: each row of the *smaller* table is matched to a row of the larger one through
the anchor columns, rated 0-10 on the weighted rating columns, and the sum is divided
by ``10 * smaller + (larger - smaller)``. That folds recall and precision into one
number, so it cannot say whether a run misses rows or invents them.

This module turns the same machinery into an F1, with the baseline as the gold set and
the curated output as the prediction:

* **Matching** - every baseline row is matched to at most one target row through the
  evaluator's own ``anchor_row_from_rows`` (semantic text similarity via BioLORD for the
  text anchors, numeric tolerance for the numeric ones). Matching is one-to-one: a
  target row that has been claimed is removed, so two identical baseline rows cannot
  both be credited to a single target row.
* **Row rating** - a matched pair is rated with the evaluator's own ``rate_row`` (0-10:
  the weighted share of rating columns that agree, floored).
* **TP** - two flavours are reported for every paper:

  ``strict``  a matched pair counts as one true positive when its rating is at least
              ``ROW_THRESHOLD`` (default 8, i.e. >= 80% of the rating weight agrees). With
              the pk-individual weights that tolerates a wrong specimen or drug name
              (each <= 2/13 of the weight) but not a wrong Parameter value (5/13).
  ``soft``    a matched pair counts as ``rating / 10`` of a true positive, so partially
              correct rows earn partial credit.

  precision = TP / |target rows|, recall = TP / |baseline rows|, F1 = 2PR / (P + R).
  FP = |target| - TP, FN = |baseline| - TP.

Run it like the other benchmark tests::

    TARGET=2026-9-19-collect-pipeline_qwen36_final \\
        python -m pytest benchmark/test_f1_with_semantic.py -k "qwen36 and not skill" -q

Environment variables: ``BASELINE`` (default ``baseline``), ``TARGET`` (default
``2026-6-12``), ``BENCHMARK_TYPE`` (default ``pk-individual``), ``ROW_THRESHOLD``
(default ``8``). Results are written to ``benchmark/result/<type>/<target>-<baseline>/
result_f1.log`` as CSV lines::

    model, pmid, mode, precision, recall, f1, tp, fp, fn

plus one ``MICRO`` (counts pooled over papers) and one ``MACRO`` (mean of the per-paper
scores) line per model and mode. The offline tests at the bottom need no model download.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

from benchmark.common import (
    ensure_target_result_directory_existed,
    prepare_single_pipeline_dataset_for_benchmark,
)
from benchmark.configs import get_benchmark_config
from benchmark.constant import BASELINE, BenchmarkType, LLModelType
from benchmark.evaluate import (
    TablesEvaluator,
    TextComparer,
    is_abbreviation_or_contraction,
    is_values_in_strings_equaled,
)

load_dotenv()

logger = logging.getLogger(__name__)

baseline = os.environ.get("BASELINE", BASELINE)
target = os.environ.get("TARGET", "2026-6-12")
benchmark_type = BenchmarkType(os.environ.get("BENCHMARK_TYPE", "pk-individual"))
ROW_THRESHOLD = int(os.environ.get("ROW_THRESHOLD", "8"))

baseline_dir = os.path.join("./benchmark/data", benchmark_type.value, baseline)
target_dir = os.path.join("./benchmark/data", benchmark_type.value, target)

# every real model label; models with no file in TARGET are skipped, so new labels added to
# LLModelType are picked up without touching this list
MODELS = [m for m in LLModelType if m not in (LLModelType.BASELINE, LLModelType.UNKNOWN)]

RESULT_HEADER = "model, pmid, mode, precision, recall, f1, tp, fp, fn"
MODES = ("strict", "soft")


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------


class CachedTextComparer(TextComparer):
    """`TextComparer` with the embedding and the pair score memoised.

    The anchor search compares the same few strings (drug names, parameter types, ...)
    against many candidate rows, and the stock comparer re-encodes both strings every time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[str, Any] = {}
        self._scores: dict[tuple[str, str], float] = {}

    def _embed(self, text: str):
        if text not in self._embeddings:
            self._embeddings[text] = self.model.encode(text, convert_to_tensor=True)
        return self._embeddings[text]

    def compare(self, a, b) -> float:
        key = (a, b)
        if key not in self._scores:
            if is_abbreviation_or_contraction(a, b):
                score = 1.0
            elif not is_values_in_strings_equaled(a, b):
                score = 0.0
            else:
                from sentence_transformers import util

                score = util.pytorch_cos_sim(self._embed(a), self._embed(b))[0][0].item()
            self._scores[key] = score
        return self._scores[key]


@dataclass(frozen=True)
class F1Result:
    tp: float  # true positives (fractional in soft mode)
    n_pred: int  # rows in the curated output
    n_gold: int  # rows in the baseline

    @property
    def fp(self) -> float:
        return self.n_pred - self.tp

    @property
    def fn(self) -> float:
        return self.n_gold - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        if self.n_pred == 0 and self.n_gold == 0:
            return 1.0  # nothing to find and nothing predicted
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


class TablesF1Evaluator(TablesEvaluator):
    """Row-level P/R/F1 built on `TablesEvaluator`'s anchor matching and row rating."""

    def __init__(
        self,
        rating_cols,
        anchor_cols,
        columns_type,
        text_cmpr: Any | None = None,
        row_threshold: int = ROW_THRESHOLD,
    ):
        # The parent constructor always loads the BioLORD model; set the same attributes
        # here so a stub comparer can be injected (offline tests) and the cached one used
        # otherwise.
        self.rating_cols = rating_cols
        self.anchor_cols = anchor_cols
        self.columns_type = columns_type
        self.text_cmpr = text_cmpr if text_cmpr is not None else CachedTextComparer()
        self.row_threshold = row_threshold

    def match_rows(
        self, gold: pd.DataFrame, pred: pd.DataFrame
    ) -> list[tuple[int, int, int]]:
        """One-to-one matches as ``(gold position, pred position, row rating 0-10)``.

        Gold rows are visited in order and each takes the first still-unclaimed prediction
        row that `anchor_row_from_rows` selects, so the result depends on row order when
        several predictions are equally good anchors (the same rule the blended score has).
        """
        remaining = list(enumerate(pred.to_dict("records")))
        pairs: list[tuple[int, int, int]] = []
        for g_pos, (_, g_row) in enumerate(gold.iterrows()):
            if not remaining:
                break
            hit = self.anchor_row_from_rows(g_row, [row for _, row in remaining])
            if hit is None:
                continue
            idx = next(i for i, (_, row) in enumerate(remaining) if row is hit)
            p_pos, p_row = remaining.pop(idx)
            pairs.append((g_pos, p_pos, self.rate_row(g_row, p_row)))
        return pairs

    def evaluate(self, gold: pd.DataFrame, pred: pd.DataFrame) -> dict[str, F1Result]:
        pairs = self.match_rows(gold, pred)
        n_gold, n_pred = len(gold), len(pred)
        return {
            "strict": F1Result(
                tp=float(sum(1 for *_, rating in pairs if rating >= self.row_threshold)),
                n_pred=n_pred,
                n_gold=n_gold,
            ),
            "soft": F1Result(
                tp=sum(rating for *_, rating in pairs) / 10.0,
                n_pred=n_pred,
                n_gold=n_gold,
            ),
        }


# --------------------------------------------------------------------------
# benchmark run + result file
# --------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _line(model: str, pmid: str, mode: str, r: F1Result) -> str:
    return (
        f"{model}, {pmid}, {mode}, {_fmt(r.precision)}, {_fmt(r.recall)}, {_fmt(r.f1)}, "
        f"{r.tp:g}, {r.fp:g}, {r.fn:g}"
    )


def _write_results(result_file: str, model: str, lines: list[str]) -> None:
    """Replace this model's lines in the result file (re-runs must not duplicate them)."""
    path = Path(result_file)
    kept = []
    if path.exists():
        kept = [
            ln for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith(f"{model},") and ln != RESULT_HEADER
        ]
    path.write_text("\n".join([RESULT_HEADER, *kept, *lines]) + "\n")


def run_f1_benchmark(
    dataset: dict,
    bench_type: BenchmarkType,
    model: LLModelType,
    result_file: str,
    evaluator: TablesF1Evaluator | None = None,
) -> dict[str, dict[str, F1Result]]:
    """Score every paper that has a `model` file; returns ``{pmid: {mode: F1Result}}``."""
    config = get_benchmark_config(bench_type)
    evaluator = evaluator or TablesF1Evaluator(
        rating_cols=config.rating_cols,
        anchor_cols=config.anchor_cols,
        columns_type=config.columns_type,
    )
    results: dict[str, dict[str, F1Result]] = {}
    for pmid, files in dataset.items():
        if model.value not in files:
            continue
        logger.info(f"F1: {pmid} with model {model.value}")
        df_baseline = pd.read_csv(files[BASELINE])
        target_path = files[model.value]
        if Path(target_path).read_text().strip().strip('"') == "":
            df_target = df_baseline.iloc[0:0]  # empty output: every baseline row is a miss
        else:
            df_target = config.preprocess(target_path)
        results[pmid] = evaluator.evaluate(df_baseline, df_target)

    lines: list[str] = []
    for mode in MODES:
        for pmid, res in results.items():
            lines.append(_line(model.value, pmid, mode, res[mode]))
        if results:
            pooled = F1Result(
                tp=sum(r[mode].tp for r in results.values()),
                n_pred=sum(r[mode].n_pred for r in results.values()),
                n_gold=sum(r[mode].n_gold for r in results.values()),
            )
            lines.append(_line(model.value, "MICRO", mode, pooled))
            n = len(results)
            lines.append(
                f"{model.value}, MACRO, {mode}, "
                f"{_fmt(sum(r[mode].precision for r in results.values()) / n)}, "
                f"{_fmt(sum(r[mode].recall for r in results.values()) / n)}, "
                f"{_fmt(sum(r[mode].f1 for r in results.values()) / n)}, , , "
            )
    _write_results(result_file, model.value, lines)
    return results


# --------------------------------------------------------------------------
# benchmark test (same shape as test_pk_individual_benchmark_with_semantic.py)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_single_pipeline_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=benchmark_type,
    )


@pytest.fixture(scope="module")
def result_path():
    result_dir = ensure_target_result_directory_existed(
        baseline=baseline,
        target=target,
        benchmark_type=benchmark_type,
    )
    return os.path.join(result_dir, "result_f1.log")


@pytest.fixture(scope="module")
def evaluator():
    # one evaluator (and one embedding cache) shared by every model in the module
    config = get_benchmark_config(benchmark_type)
    return TablesF1Evaluator(
        rating_cols=config.rating_cols,
        anchor_cols=config.anchor_cols,
        columns_type=config.columns_type,
    )


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.value)
def test_f1_benchmark(prepared_dataset, result_path, evaluator, model):
    if not any(model.value in files for files in prepared_dataset.values()):
        pytest.skip(f"no {model.value} files in {target_dir}")
    results = run_f1_benchmark(
        dataset=prepared_dataset,
        bench_type=benchmark_type,
        model=model,
        result_file=result_path,
        evaluator=evaluator,
    )
    assert results
    for per_mode in results.values():
        for r in per_mode.values():
            assert 0.0 <= r.precision <= 1.0
            assert 0.0 <= r.recall <= 1.0
            assert 0.0 <= r.f1 <= 1.0


# --------------------------------------------------------------------------
# offline tests of the F1 logic (stub text comparer: no model download)
# --------------------------------------------------------------------------


class _ExactText:
    """Stands in for the BioLORD comparer: case-insensitive exact match."""

    def compare(self, a, b) -> float:
        return 1.0 if str(a).strip().lower() == str(b).strip().lower() else 0.0


def _stub_evaluator(row_threshold: int = 8) -> TablesF1Evaluator:
    config = get_benchmark_config(BenchmarkType.PK_INDIVIDUAL)
    return TablesF1Evaluator(
        rating_cols=config.rating_cols,
        anchor_cols=config.anchor_cols,
        columns_type=config.columns_type,
        text_cmpr=_ExactText(),
        row_threshold=row_threshold,
    )


def _row(pid, value, drug="DrugA", analyte="DrugA", specimen="Plasma", ptype="Cmax"):
    return {
        "Patient ID": str(pid),
        "Drug name": drug,
        "Analyte": analyte,
        "Specimen": specimen,
        "Population": "Maternal",
        "Pregnancy stage": "Trimester 3",
        "Pediatric/Gestational age": "N/A",
        "Parameter type": ptype,
        "Parameter unit": "ng/ml",
        "Parameter value": float(value),
        "Time value": float("nan"),
        "Time unit": "N/A",
    }


def _df(*rows):
    return pd.DataFrame(list(rows))


def test_perfect_match():
    gold = _df(_row(1, 10), _row(2, 20), _row(3, 30))
    res = _stub_evaluator().evaluate(gold, gold.copy())
    for mode in MODES:
        assert (res[mode].precision, res[mode].recall, res[mode].f1) == (1.0, 1.0, 1.0)


def test_extra_rows_lower_precision_only():
    gold = _df(_row(1, 10), _row(2, 20))
    pred = _df(_row(1, 10), _row(2, 20), _row(3, 30), _row(4, 40))
    r = _stub_evaluator().evaluate(gold, pred)["strict"]
    assert r.recall == 1.0 and r.precision == 0.5
    assert (r.tp, r.fp, r.fn) == (2, 2, 0)
    assert r.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_missing_rows_lower_recall_only():
    gold = _df(_row(1, 10), _row(2, 20), _row(3, 30), _row(4, 40))
    pred = _df(_row(1, 10), _row(2, 20))
    r = _stub_evaluator().evaluate(gold, pred)["strict"]
    assert r.precision == 1.0 and r.recall == 0.5
    assert (r.tp, r.fp, r.fn) == (2, 0, 2)


def test_matching_is_one_to_one():
    # two identical gold rows must not both be credited to the single prediction
    gold = _df(_row(1, 10), _row(1, 10))
    pred = _df(_row(1, 10))
    r = _stub_evaluator().evaluate(gold, pred)["strict"]
    assert r.tp == 1 and r.precision == 1.0 and r.recall == 0.5


def test_wrong_value_is_not_a_strict_hit_but_earns_soft_credit():
    # same anchor (patient, drug, analyte) but a different value: everything except the
    # value (weight 5 of 13) agrees, so the rating is int(10 * 8 / 13) = 6
    gold = _df(_row(1, 10, ptype="Cmax"))
    pred = _df(_row(1, 99, ptype="Cmax"))
    ev = _stub_evaluator()
    pairs = ev.match_rows(gold, pred)
    assert pairs == [(0, 0, 6)]
    res = ev.evaluate(gold, pred)
    assert res["strict"].tp == 0 and res["strict"].f1 == 0.0
    assert res["soft"].tp == pytest.approx(0.6)


def test_minor_text_mismatch_still_counts_as_strict_hit():
    # a wrong specimen only costs 0.5 of 13 weight: rating 9 >= threshold 8
    gold = _df(_row(1, 10, specimen="Plasma"))
    pred = _df(_row(1, 10, specimen="Serum"))
    res = _stub_evaluator().evaluate(gold, pred)
    assert res["strict"].tp == 1.0 and res["soft"].tp == pytest.approx(0.9)


def test_threshold_is_configurable():
    gold = _df(_row(1, 10, specimen="Plasma"))
    pred = _df(_row(1, 10, specimen="Serum"))
    assert _stub_evaluator(row_threshold=10).evaluate(gold, pred)["strict"].tp == 0.0


def test_empty_prediction_scores_zero_and_empty_both_scores_one():
    gold = _df(_row(1, 10))
    ev = _stub_evaluator()
    empty = gold.iloc[0:0]
    assert ev.evaluate(gold, empty)["strict"].f1 == 0.0
    assert ev.evaluate(empty, empty)["strict"].f1 == 1.0


def test_rerunning_replaces_the_models_lines(tmp_path):
    f = str(tmp_path / "result_f1.log")
    _write_results(f, "a", ["a, 1, strict, 1.0000, 1.0000, 1.0000, 1, 0, 0"])
    _write_results(f, "b", ["b, 1, strict, 0.5000, 0.5000, 0.5000, 1, 1, 1"])
    _write_results(f, "a", ["a, 1, strict, 0.0000, 0.0000, 0.0000, 0, 1, 1"])
    lines = Path(f).read_text().splitlines()
    assert lines[0] == RESULT_HEADER
    assert sum(ln.startswith("a,") for ln in lines) == 1 and sum(ln.startswith("b,") for ln in lines) == 1
    assert any(ln.startswith("a, 1, strict, 0.0000") for ln in lines)
