"""Regression tests for extractor.curate_agent.validity_checker.

Exercises the already-diagnosed `simple_qwen38` failures from
docs/SIMPLE_MODE_QWEN38_ISSUES.md - PMID 30825333's column-shift row and PMID
34183327's zero-row table - as the acceptance test for "does the checker
actually catch what it's meant to." Fixtures are reconstructed from the
issue doc's own quoted data, not copies of the original run (which lived on
scratch and is gone), but they reproduce the exact bug shape.

Tier 3 is an LLM call (`llm_verification.verify`) - every test here injects
a fake `verify_fn` so nothing hits a live LLM, mirroring the DI pattern
`test_curate_comparison_judge.py` and `test_curate_orchestrator.py` already
use for their own injected dependencies.
"""

from extractor.curate_agent import validity_checker as vc

PK_SUMMARY_HEADER = (
    "Drug name,Analyte,Specimen,Population,Pregnancy stage,Subject N,"
    "Parameter type,Value,Unit,Summary statistics,Variation type,"
    "Variation value,Interval type,Lower limit,High limit,P value"
)

SOURCE_MD = """\
| Drug | Parameter | N | Value | Unit | SD |
| --- | --- | --- | --- | --- | --- |
| Ketamine | Total area under curve | 19 | 91 | Mg/min/L | 30 |
"""


def _clean_verify(**kwargs):
    """Fake Tier 3: table matches its source, no errors, no missed rows."""
    return {"correct": True, "error_row_count": 0, "missed_row_count": 0,
            "explanation": "All values match."}


def test_column_shift_row_is_flagged_and_escalates(tmp_path):
    # The exact shifted row quoted in docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 1:
    # the model never wrote "Subject N", so every later field slid one column
    # over. Field count still matches the 16-column schema. Tier 1 alone
    # already catches this (dropped_rows==1) - that's the point of this test -
    # regardless of what Tiers 2-3 (faked clean here) report.
    csv_path = tmp_path / "30825333_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,,children,,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_clean_verify,
    )

    assert report["tier1"]["total_rows"] == 1
    assert report["tier1"]["dropped_rows"] == 1  # row scores <= 0 under _fits()
    assert report["escalate"] is True


def test_zero_row_table_is_flagged_and_escalates_without_calling_tier3(tmp_path):
    # docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 2: table_2, the paper's actual PK
    # data table, produced zero rows (the drug-as-subheader nesting confused
    # the model). Neither Tier 2 nor Tier 3 must run on a zero-row table -
    # there is nothing to check, and no LLM call should be spent on it.
    csv_path = tmp_path / "34183327_combined.csv"
    csv_path.write_text(PK_SUMMARY_HEADER + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    def fail_verify(**kwargs):
        raise AssertionError("Tier 3 should not run on a zero-row table")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=fail_verify,
    )

    assert report["tier1"]["total_rows"] == 0
    assert report["tier2"] is None
    assert report["tier3"] is None
    assert report["error_rate"] == 1.0
    assert report["escalate"] is True


def test_clean_table_does_not_escalate(tmp_path):
    # A well-formed table with no structural, existence, or LLM-flagged
    # problems should not trip the gate - this is the "stay quiet on
    # known-good papers" half of the acceptance bar, not just "flags
    # known-bad ones."
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        extra_dropped_rows=0,  # simulating Arm A's manifest.json, which already says 0
        verify_fn=_clean_verify,
    )

    assert report["tier1"]["total_rows"] == 1
    assert report["tier1"]["dropped_rows"] == 0
    assert report["tier2"]["unsupported"] == 0
    assert report["tier3"]["error_row_count"] == 0
    assert report["tier3"]["missed_row_count"] == 0
    assert report["error_rate"] == 0.0
    assert report["escalate"] is False


def test_tier2_existence_flags_a_hallucinated_number(tmp_path):
    # A value that never appears anywhere in the source at all - existence
    # mode's one job - even though Tier 1 and the (faked clean) Tier 3 see
    # nothing wrong structurally.
    csv_path = tmp_path / "hallucinated_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,999,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_clean_verify, threshold=0.05,
    )

    assert report["tier2"]["unsupported"] == 1  # 999 is nowhere in SOURCE_MD
    assert report["error_rate"] > 0.0
    assert report["escalate"] is True


def test_tier3_llm_errors_feed_into_error_rate(tmp_path):
    # Tier 3 is the only tier that can catch misattribution (a value that
    # exists in the source but is on the wrong row) - fake it reporting one
    # error row and one missed row, and confirm both land in error_rate via
    # (error_row_count + missed_row_count) / total_rows.
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    def flawed_verify(**kwargs):
        return {"correct": False, "error_row_count": 1, "missed_row_count": 1,
                "explanation": 'idx 0, Col "Value": change "91" to "90"\nMissing row: ...'}

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=flawed_verify,
    )

    assert report["tier3"]["error_row_count"] == 1
    assert report["tier3"]["missed_row_count"] == 1
    assert report["escalate"] is True


def test_tier3_rate_denominator_includes_missed_rows(tmp_path):
    # missed_row_count counts source rows the model never even attempted to
    # extract - those were never part of tier1's total_rows (rows that made
    # it to the CSV, plus rows the parser attempted and dropped). If the
    # denominator stayed plain total_rows while missed rows piled into the
    # numerator, the rate could exceed 1.0; it must include missed_row_count
    # in the denominator too, same as the numerator, so it stays <= 1.0.
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    def very_incomplete_verify(**kwargs):
        # 1 curated row, but the model reports 5 missing rows on top of it.
        return {"correct": False, "error_row_count": 0, "missed_row_count": 5,
                "explanation": "Missing row: ...\n" * 5}

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=very_incomplete_verify,
    )

    # tier3 rate = (0 + 5) / (1 + 5) = 0.833..., not (0 + 5) / 1 = 5.0.
    tier1_total = report["tier1"]["total_rows"]
    expected_tier3_rate = 5 / (tier1_total + 5)
    assert expected_tier3_rate < 1.0
    assert report["error_rate"] <= 1.0
    assert report["escalate"] is True


def test_include_tier1_rate_false_excludes_a_real_tier1_hit(tmp_path):
    # Same column-shift row as test_column_shift_row_is_flagged_and_escalates
    # - Tier 1 genuinely does flag it (dropped_rows=1) - but with
    # include_tier1_rate=False, that hit must not land in error_rate at all,
    # per curate_orchestrator.py's own Arm B call (see validity_checker.py's
    # check_validity() docstring for why: Tier 1 was never validated against
    # the multi-agent arm's output conventions).
    csv_path = tmp_path / "30825333_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,,children,,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_clean_verify, include_tier1_rate=False,
    )

    assert report["tier1"]["dropped_rows"] == 1  # Tier 1 still computed, still flags it...
    assert report["error_rate"] == 0.0  # ...but excluded from error_rate entirely.
    assert report["escalate"] is False


def test_pk_summary_na_and_synonym_values_do_not_trip_tier1(tmp_path):
    # Per the real PMID 16143486 investigation (docs/cascade_mode_results.md):
    # "N/A" in a numeric column and a correct-but-unlisted synonym
    # ("Parturition" for "delivery") must not be treated as structural
    # errors - this exercises the fix end to end through check_validity(),
    # not just scripts/simple_prompt/validity.py's scorer directly.
    csv_path = tmp_path / "na_and_synonym_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,Parturition,N/A,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_clean_verify,
    )

    assert report["tier1"]["dropped_rows"] == 0
    assert report["escalate"] is False
