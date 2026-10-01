"""The semantic benchmark reads the baseline raw; its column names must match the config.

pk-summary PMID 16143486's baseline has "Pregnancy Stage" (config: "Pregnancy stage"), which
made 4 of 12 sets crash with KeyError in the semantic scorer (targets with fewer rows than
the baseline reach the last anchor column).
"""
import pandas as pd

from benchmark.comm_semantic import align_column_case
from benchmark.configs import PK_SUMMARY_ANCHOR_COLUMNS, get_benchmark_config
from benchmark.constant import BenchmarkType
from benchmark.evaluate import TablesEvaluator


def _config():
    return get_benchmark_config(BenchmarkType.PK_SUMMARY)


def test_case_and_space_variants_are_renamed_to_the_config_names():
    df = pd.DataFrame(columns=["Drug name", "Pregnancy Stage", " Value ", "Unit", "Summary Statistics", "Extra Note"])
    out = align_column_case(df, _config())
    assert "Pregnancy stage" in out.columns and "Value" in out.columns
    assert "Summary statistics" in out.columns  # configured (rating/type map), so renamed too
    assert "Extra Note" in out.columns  # not a configured column: left alone


def test_matching_and_unknown_columns_are_untouched():
    df = pd.DataFrame(columns=["Drug name", "Pregnancy stage", "Something else"])
    assert list(align_column_case(df, _config()).columns) == list(df.columns)


def test_real_baseline_no_longer_raises_in_anchor_lookup():
    base = align_column_case(pd.read_csv("benchmark/data/pk-summary/baseline/16143486_baseline.csv"), _config())
    cfg = _config()
    ev = TablesEvaluator(rating_cols=cfg.rating_cols, anchor_cols=cfg.anchor_cols, columns_type=cfg.columns_type)
    assert all(c in base.columns for c in PK_SUMMARY_ANCHOR_COLUMNS)
    row = base.iloc[0]
    # a target row that matches no earlier anchor: the lookup must walk every anchor column
    target_row = pd.Series({c: "zzz-no-match" for c in PK_SUMMARY_ANCHOR_COLUMNS})
    ev.anchor_row_from_rows(target_row, base.to_dict("records"))
    ev.anchor_row_from_rows(row, base.to_dict("records"))
