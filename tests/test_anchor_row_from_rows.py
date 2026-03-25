"""
Unit tests for TablesEvaluator.anchor_row_from_rows.

These tests use a lightweight subclass (_SimpleEvaluator) that replaces the
heavy SentenceTransformer-backed _is_equal with plain string equality, so
they run without any ML dependencies.

Scenarios covered:
  1.  Unique match on the first anchor column → early return
  2.  Progressive narrowing across two columns → unique after second column
  3.  Bug regression (reset-to-all-rows): failed column discards earlier
      progress and returns a row from the wrong group.
      Col A narrows to drug=A rows; Col B (specimen) finds no match and
      RESETS candidate_rows back to all rows; Col C (parameter) then
      finds a drug=B row. Buggy code returns drug=B. Fixed code stays
      within drug=A candidates and returns a drug=A row.
  4.  Second bug regression (single-column reset): one failing column
      should not cause a later column to match an unrelated row.
  5.  Multiple candidates remain after all columns → return first candidate
  6.  All anchor columns produce zero genuine matches → return None
  7.  Empty string and None anchor values are skipped
  8.  All anchor values are empty/None → never narrowed → return None
  9.  Single row in the pool — exact match found → return it
 10.  Single row in the pool — best-effort partial match (only drug differs)
      → function returns the row (intended best-effort behaviour)
 11.  First column skipped (empty value), second column uniquely matches
"""

import pytest

from benchmark.evaluate import TablesEvaluator


# ---------------------------------------------------------------------------
# Lightweight evaluator — no transformer required
# ---------------------------------------------------------------------------

class _SimpleEvaluator(TablesEvaluator):
    """Subclass that skips the SentenceTransformer and uses exact string equality."""

    def __init__(self, anchor_cols: list[str]):
        self.anchor_cols = anchor_cols
        self.rating_cols = []
        self.columns_type = {}
        self.text_cmpr = None  # not used in anchor_row_from_rows

    def _is_equal(self, v1, v2) -> bool:  # type: ignore[override]
        return str(v1).strip().lower() == str(v2).strip().lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> dict:
    """Build a plain dict row (anchor_row_from_rows accepts list[dict])."""
    return kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnchorRowFromRows:
    def setup_method(self):
        self.ev = _SimpleEvaluator(anchor_cols=["drug", "specimen", "parameter"])

    # 1. Unique match on first anchor column → early return
    def test_unique_match_first_column(self):
        pool = [
            _row(drug="A", specimen="plasma", parameter="AUC"),
            _row(drug="B", specimen="plasma", parameter="AUC"),
        ]
        query = _row(drug="A", specimen="plasma", parameter="AUC")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["drug"] == "A"

    # 2. Progressive narrowing — unique match only after second column
    def test_progressive_narrowing_two_columns(self):
        pool = [
            _row(drug="A", specimen="plasma", parameter="AUC"),
            _row(drug="A", specimen="urine",  parameter="Cmax"),
        ]
        query = _row(drug="A", specimen="urine", parameter="Cmax")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["specimen"] == "urine"

    # 3. Bug regression: reset-to-all-rows causes wrong-group return
    #
    #    Pool: [r1=(A,plasma,Cmax), r2=(A,urine,AUC), r3=(B,plasma,T12)]
    #    Query: (A, blood, T12)
    #
    #    Buggy execution:
    #      drug=A    → narrows to [r1, r2]
    #      specimen=blood → 0 matches among [r1,r2] → RESETS to all rows
    #      parameter=T12  → finds [r3] in all rows → returns r3 (drug=B) ← WRONG
    #
    #    Fixed execution:
    #      drug=A    → narrows to [r1, r2]
    #      specimen=blood → 0 matches → keep [r1, r2] (skip column)
    #      parameter=T12  → 0 matches among [r1, r2] → keep [r1, r2]
    #      → returns r1 (drug=A, correct group)
    def test_failed_column_must_not_reset_to_all_rows(self):
        r1 = _row(drug="A", specimen="plasma", parameter="Cmax")
        r2 = _row(drug="A", specimen="urine",  parameter="AUC")
        r3 = _row(drug="B", specimen="plasma", parameter="T12")
        pool = [r1, r2, r3]
        query = _row(drug="A", specimen="blood", parameter="T12")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["drug"] == "A", (
            "Reset-to-all-rows bug: failed column caused a drug=B row to be returned"
        )

    # 4. Bug regression (simpler 2-row case):
    #    Col A uniquely narrows to r1; Col B fails → reset causes Col C to
    #    find r2 from the full pool instead of honouring the Col A constraint.
    #
    #    Pool: [r1=(A,plasma,Cmax), r2=(B,blood,AUC)]
    #    Query: (A, urine, AUC)
    #
    #    Buggy:
    #      drug=A    → [r1]  candidate_rows=[r1]
    #      specimen=urine → 0 among [r1] → RESETS to [r1, r2]
    #      parameter=AUC  → [r2] among [r1, r2] → returns r2 (drug=B) ← WRONG
    #
    #    Fixed:
    #      drug=A    → [r1]  candidate_rows=[r1], narrowed=True
    #      specimen=urine → 0 → keep [r1]
    #      parameter=AUC  → 0 among [r1] → keep [r1]
    #      → returns r1 (drug=A)
    def test_failed_column_single_candidate_preserved(self):
        r1 = _row(drug="A", specimen="plasma", parameter="Cmax")
        r2 = _row(drug="B", specimen="blood",  parameter="AUC")
        pool = [r1, r2]
        query = _row(drug="A", specimen="urine", parameter="AUC")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["drug"] == "A", (
            "Reset bug: failed column caused a drug=B row to be returned"
        )

    # 5. Multiple candidates remain after all columns — return first
    def test_multiple_candidates_returns_first(self):
        r1 = _row(drug="A", specimen="plasma", parameter="AUC")
        r2 = _row(drug="A", specimen="plasma", parameter="AUC")  # identical second row
        pool = [r1, r2]
        query = _row(drug="A", specimen="plasma", parameter="AUC")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result is r1  # first candidate returned

    # 6. No column matches anything → return None (narrowed=False path)
    def test_no_match_returns_none(self):
        pool = [
            _row(drug="B", specimen="urine",  parameter="Cmax"),
            _row(drug="C", specimen="plasma", parameter="AUC"),
        ]
        query = _row(drug="X", specimen="Y", parameter="Z")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is None

    # 7. Empty string and None anchor values are skipped
    def test_empty_and_none_values_skipped(self):
        pool = [
            _row(drug="A", specimen="plasma", parameter="AUC"),
            _row(drug="B", specimen="plasma", parameter="Cmax"),
        ]
        # drug="" and specimen=None are skipped; "parameter" uniquely identifies
        query = _row(drug="", specimen=None, parameter="Cmax")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["parameter"] == "Cmax"

    # 8. All anchor values are empty/None → never narrowed → return None
    def test_all_anchor_values_empty_returns_none(self):
        pool = [_row(drug="A", specimen="plasma", parameter="AUC")]
        query = _row(drug="", specimen=None, parameter="")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is None

    # 9. Single row in pool — exact match
    def test_single_row_exact_match(self):
        pool = [_row(drug="A", specimen="plasma", parameter="AUC")]
        query = _row(drug="A", specimen="plasma", parameter="AUC")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["drug"] == "A"

    # 10. Single row in pool — best-effort partial match when only drug differs.
    #     The function is intentionally a best-effort matcher: if the only row
    #     in the pool matches on specimen and parameter, it is returned even
    #     though drug differs (to allow the caller to penalise the drug column).
    def test_single_row_partial_match_best_effort(self):
        pool = [_row(drug="B", specimen="plasma", parameter="AUC")]
        query = _row(drug="A", specimen="plasma", parameter="AUC")
        result = self.ev.anchor_row_from_rows(query, pool)
        # Best-effort: specimen and parameter match so the row is returned
        assert result is not None
        assert result["specimen"] == "plasma"
        assert result["parameter"] == "AUC"

    # 11. First column value is empty (skipped), second column uniquely matches
    def test_first_column_empty_second_matches(self):
        pool = [
            _row(drug="A", specimen="plasma", parameter="AUC"),
            _row(drug="B", specimen="urine",  parameter="Cmax"),
        ]
        query = _row(drug="", specimen="urine", parameter="Cmax")
        result = self.ev.anchor_row_from_rows(query, pool)
        assert result is not None
        assert result["specimen"] == "urine"
