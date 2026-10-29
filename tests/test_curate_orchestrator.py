"""Phase 2 tests for extractor.curate_agent.curate_orchestrator.

Exercises the cascade's own logic - gate, escalation, interim pick-rule -
with `run_cheap_arm_fn`/`run_expensive_arm_fn` swapped for fakes, since a real
run needs a live Ollama server (Arm A) and a multi-agent gemma4 run (Arm B).
This mirrors Phase 1's approach: test the deterministic orchestration against
reconstructed fixture data, not live models.
"""

import pandas as pd
import pytest

from extractor.curate_agent import curate_orchestrator as co
from extractor.database.pmid_db import PMIDDB

PMID = "99999999"

SOURCE_TABLE = pd.DataFrame({
    "Drug": ["Ketamine"],
    "Parameter": ["Total area under curve"],
    "N": [19],
    "Value": [91],
    "Unit": ["Mg/min/L"],
})

PK_SUMMARY_HEADER = [
    "Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage",
    "Subject N", "Parameter type", "Value", "Unit", "Summary statistics",
    "Variation type", "Variation value", "Interval type", "Lower limit",
    "High limit", "P value",
]

CLEAN_ROW = [
    "Ketamine", "Ketamine", "Plasma", "children", "", "19",
    "Total area under curve", "91", "Mg/min/L", "mean", "SD", "30",
    "", "", "", "",
]

SHIFTED_ROW = [
    "Ketamine", "Ketamine", "", "children", "", "Total area under curve",
    "91", "Mg/min/L", "mean", "SD", "30", "", "", "", "", "",
]


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PK_SUMMARY_HEADER)
        for row in rows:
            w.writerow(row)


def _pmid_db(tmp_path):
    db = PMIDDB(tmp_path / "pmid_info.db")
    tables = [{"caption": "Table 1", "footnote": "", "table": SOURCE_TABLE.copy()}]
    db.insert_pmid_info(PMID, "title", "abstract", "full text", tables, [])
    return db


def test_clean_arm_a_never_escalates(tmp_path):
    calls = {"a": 0, "b": 0}

    def fake_cheap(pmid, schema, scratch_root, out_root, **kw):
        calls["a"] += 1
        import os
        os.makedirs(out_root, exist_ok=True)
        csv_path = os.path.join(out_root, "combined.csv")
        _write_csv(csv_path, [CLEAN_ROW])
        return csv_path, 0  # manifest says nothing dropped

    def fake_expensive(pmid, schema, out_dir, pmid_db):
        calls["b"] += 1
        raise AssertionError("Arm B should not run when Arm A already passes")

    result = co.curate(
        PMID, schema="pk_summary", pmid_db=_pmid_db(tmp_path),
        work_dir=str(tmp_path / "work"),
        run_cheap_arm_fn=fake_cheap, run_expensive_arm_fn=fake_expensive,
    )

    assert calls == {"a": 1, "b": 0}
    assert result["winner"] == "A"
    assert result["escalated"] is False
    assert result["candidate_b"] is None
    assert result["candidate_a"]["report"]["escalate"] is False


def test_bad_arm_a_escalates_and_clean_arm_b_wins(tmp_path):
    def fake_cheap(pmid, schema, scratch_root, out_root, **kw):
        import os
        os.makedirs(out_root, exist_ok=True)
        csv_path = os.path.join(out_root, "combined.csv")
        _write_csv(csv_path, [SHIFTED_ROW])
        return csv_path, 0

    def fake_expensive(pmid, schema, out_dir, pmid_db):
        import os
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{pmid}_{schema}.csv")
        _write_csv(csv_path, [CLEAN_ROW])
        return csv_path

    result = co.curate(
        PMID, schema="pk_summary", pmid_db=_pmid_db(tmp_path),
        work_dir=str(tmp_path / "work"),
        run_cheap_arm_fn=fake_cheap, run_expensive_arm_fn=fake_expensive,
    )

    assert result["escalated"] is True
    assert result["candidate_a"]["report"]["escalate"] is True
    assert result["winner"] == "B"
    assert result["candidate_b"]["report"]["error_rate"] == 0.0


def test_tie_calls_comparison_judge(tmp_path):
    # Both arms produce the same shifted row - error_rate ties, so the gate
    # must hand the pick to Tier 4 (ComparisonJudge) rather than defaulting
    # to either side. The judge itself is faked here; see
    # test_curate_comparison_judge.py for the judge's own logic.
    def fake_cheap(pmid, schema, scratch_root, out_root, **kw):
        import os
        os.makedirs(out_root, exist_ok=True)
        csv_path = os.path.join(out_root, "combined.csv")
        _write_csv(csv_path, [SHIFTED_ROW])
        return csv_path, 0

    def fake_expensive(pmid, schema, out_dir, pmid_db):
        import os
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{pmid}_{schema}.csv")
        _write_csv(csv_path, [SHIFTED_ROW])
        return csv_path

    calls = []

    def fake_compare(**kwargs):
        calls.append(kwargs)
        return {"winner": "B", "explanation": "fake verdict", "used_full_tables": False}

    result = co.curate(
        PMID, schema="pk_summary", pmid_db=_pmid_db(tmp_path),
        work_dir=str(tmp_path / "work"),
        run_cheap_arm_fn=fake_cheap, run_expensive_arm_fn=fake_expensive,
        compare_fn=fake_compare,
    )

    assert result["escalated"] is True
    assert result["candidate_a"]["report"]["error_rate"] == result["candidate_b"]["report"]["error_rate"]
    assert len(calls) == 1
    assert result["winner"] == "B"
    assert result["judge"] == {"winner": "B", "explanation": "fake verdict", "used_full_tables": False}


def test_no_tie_never_calls_comparison_judge(tmp_path):
    def fake_cheap(pmid, schema, scratch_root, out_root, **kw):
        import os
        os.makedirs(out_root, exist_ok=True)
        csv_path = os.path.join(out_root, "combined.csv")
        _write_csv(csv_path, [SHIFTED_ROW])
        return csv_path, 0

    def fake_expensive(pmid, schema, out_dir, pmid_db):
        import os
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{pmid}_{schema}.csv")
        _write_csv(csv_path, [CLEAN_ROW])
        return csv_path

    def fake_compare(**kwargs):
        raise AssertionError("ComparisonJudge should not run when one error_rate is strictly lower")

    result = co.curate(
        PMID, schema="pk_summary", pmid_db=_pmid_db(tmp_path),
        work_dir=str(tmp_path / "work"),
        run_cheap_arm_fn=fake_cheap, run_expensive_arm_fn=fake_expensive,
        compare_fn=fake_compare,
    )

    assert result["winner"] == "B"
    assert result["judge"] is None


def test_unknown_schema_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        co.curate(PMID, schema="pk_individual", pmid_db=_pmid_db(tmp_path),
                   work_dir=str(tmp_path / "work"))
