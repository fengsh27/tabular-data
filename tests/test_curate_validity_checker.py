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


def _fail_verify(**kwargs):
    raise AssertionError("Tier 3 verify must not run on a zero-row table")


def test_zero_row_table_with_curatable_source_escalates(tmp_path):
    # docs/SIMPLE_MODE_QWEN38_ISSUES.md Issue 2: table_2, the paper's actual PK
    # data table, produced zero rows (the drug-as-subheader nesting confused
    # the model). Tier 2 and the row-level Tier 3 verify never run on a
    # zero-row table; the empty-table assessment runs instead and, since the
    # source does hold curatable data, the table is a failure with the rows
    # it should have produced recorded for the judge.
    csv_path = tmp_path / "34183327_combined.csv"
    csv_path.write_text(PK_SUMMARY_HEADER + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")
    seen = {}

    def has_data(**kwargs):
        seen.update(kwargs)
        return {"has_curatable_data": True, "expected_rows": ["Ketamine AUC 91 Mg/min/L"],
                "explanation": "one PK parameter row with n, value, unit, SD"}

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_fail_verify, empty_fn=has_data,
    )

    assert report["tier1"]["total_rows"] == 0
    assert report["tier2"] is None
    assert report["tier3"]["empty_table_justified"] is False
    assert report["tier3"]["missed_row_count"] == 1
    assert report["tier3"]["expected_rows"] == ["Ketamine AUC 91 Mg/min/L"]
    assert "Missing row: Ketamine AUC 91 Mg/min/L" in report["tier3"]["explanation"]
    assert report["error_rate"] == 1.0
    assert report["escalate"] is True
    # the assessment saw the schema and the source table text, not a path
    assert seen["schema"] == "pk_summary"
    assert "Total area under curve" in seen["source_tables"]


def test_zero_row_table_is_accepted_when_source_has_no_curatable_data(tmp_path):
    # PMID 18426260 table 2 / 34746508 table 3 (Tier 4 audit): Arm A returned
    # nothing for a means-only / literature table. The empty table is the
    # correct output - no escalation, error_rate 0, flagged as justified.
    csv_path = tmp_path / "18426260_table_2.csv"
    csv_path.write_text(PK_SUMMARY_HEADER + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    def no_data(**kwargs):
        return {"has_curatable_data": False, "expected_rows": [],
                "explanation": "rows are group means with no patient identifiers"}

    report = vc.check_validity(
        str(csv_path), "pk_individual", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_fail_verify, empty_fn=no_data,
    )

    assert report["tier1"]["total_rows"] == 0
    assert report["tier2"] is None
    assert report["tier3"]["empty_table_justified"] is True
    assert report["tier3"]["correct"] is True
    assert report["tier3"]["missed_row_count"] == 0
    assert report["error_rate"] == 0.0
    assert report["escalate"] is False


def test_zero_row_table_without_empty_fn_keeps_the_old_behaviour(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(PK_SUMMARY_HEADER + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_fail_verify, empty_fn=None,
    )

    assert report["tier2"] is None and report["tier3"] is None
    assert report["error_rate"] == 1.0 and report["escalate"] is True


def test_empty_fn_is_not_called_on_a_table_with_rows(tmp_path):
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    def fail_empty(**kwargs):
        raise AssertionError("empty-table assessment must not run when the table has rows")

    report = vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=_clean_verify, empty_fn=fail_empty,
    )
    assert report["escalate"] is False


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


def test_schema_reaches_verify_fn(tmp_path):
    # Tier 3's completeness rules depend on the schema (llm_verification.
    # completeness_rules_for) - check_validity must hand its own `schema` on.
    csv_path = tmp_path / "clean_combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,"
        "Mg/min/L,mean,SD,30,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")
    seen = {}

    def capture_verify(**kwargs):
        seen.update(kwargs)
        return _clean_verify()

    vc.check_validity(
        str(csv_path), "pk_summary", [str(source_path)],
        paper_title="A PK study", paper_abstract="An abstract.",
        verify_fn=capture_verify,
    )

    assert seen["schema"] == "pk_summary"


def test_a_table_missing_half_its_rows_escalates_even_when_tier1_and_tier2_are_clean(tmp_path):
    """PMID 32153014 table 2 (pk_individual, rounds 4-5): Arm A returned 15 correct rows and
    left out the 13 placenta / placenta-to-maternal rows; Tier 1 and Tier 2 saw nothing wrong
    (they only judge rows that exist) and Tier 3 reported 13 missed rows. With the tiers
    averaged, error_rate was (0 + 0 + 13/28) / 3 = 0.155 and nothing escalated. The gate now
    takes the worst tier: 13/28 = 0.46 > 0.2 - escalate."""
    rows = "\n".join(f"P{i},Infliximab,Infliximab,serum,maternal,delivery,maternal,concentration,µg/mL,{v},,"
                     for i, v in enumerate([25.3, 29.8, 12.0, 24.0, 17.0, 29.0, 3.0, 0.1, 0.2, 1.18, 2.0, 1.71, 0.04, 5.5, 6.6], 1))
    csv_path = tmp_path / "arm_a.csv"
    csv_path.write_text("Patient ID,Drug name,Analyte,Specimen,Population,Pregnancy stage,Pediatric/Gestational age,"
                        "Parameter type,Parameter unit,Parameter value,Time value,Time unit\n" + rows + "\n", encoding="utf-8")
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text("| Patient | Maternal | Cord |\n| --- | --- | --- |\n" + "\n".join(
        f"| {i} | {v} | {v} |" for i, v in enumerate([25.3, 29.8, 12.0, 24.0, 17.0, 29.0, 3.0, 0.1, 0.2, 1.18, 2.0, 1.71, 0.04, 5.5, 6.6], 1)),
        encoding="utf-8")

    def half_missing_verify(**kwargs):
        return {"correct": False, "error_row_count": 0, "missed_row_count": 13,
                "explanation": "\n".join(f"Missing row: Patient ID=\"{i}\", Specimen=\"placenta\"" for i in range(1, 14))}

    report = vc.check_validity(str(csv_path), "pk_individual", [str(source_path)],
                               paper_title="Placental transfer", paper_abstract="An abstract.",
                               verify_fn=half_missing_verify)

    assert report["tier1"]["dropped_rows"] == 0 and report["tier2"]["unsupported"] == 0
    assert abs(report["error_rate"] - 13 / 28) < 1e-9  # the Tier 3 rate itself, not a third of it
    assert report["escalate"] is True


def test_error_rate_is_the_worst_tier_not_the_mean(tmp_path):
    # Tier 1 flags the shifted row (1 of 2 dropped -> 0.5); Tier 3 is told the table is fine.
    # Mean of (0.5, 0.0, 0.0) would be 0.17 and pass the 0.2 gate; the worst tier is 0.5.
    csv_path = tmp_path / "combined.csv"
    csv_path.write_text(
        PK_SUMMARY_HEADER + "\n"
        "Ketamine,Ketamine,Plasma,children,,19,Total area under curve,91,Mg/min/L,mean,SD,30,,,,\n"
        "Ketamine,Ketamine,,children,,Clearance,55,L/h,mean,SD,7,,,,,\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "00_markdown_table.md"
    source_path.write_text(SOURCE_MD, encoding="utf-8")

    report = vc.check_validity(str(csv_path), "pk_summary", [str(source_path)],
                               paper_title="A PK study", paper_abstract="An abstract.", verify_fn=_clean_verify)

    assert report["tier1"]["dropped_rows"] == 1
    assert report["error_rate"] == max(report["tier1"]["dropped_rows"] / report["tier1"]["total_rows"], 0.0)
    assert report["escalate"] is True


def test_generic_schema_tier2_checks_only_numeric_cells(tmp_path):
    # "Trimester 3" and the Note excerpt carry digits that are not source values;
    # the dose and the count are, and the dose 999 is not in the text.
    src = tmp_path / "full_text.md"
    src.write_text("Ten (10) mothers received buprenorphine 8 mg once daily in the third trimester.",
                   encoding="utf-8")
    csv_path = tmp_path / "arm_a.csv"
    csv_path.write_text(
        "Drug/Metabolite name,Dose amount,Dose unit,Dose frequency,Dose schedule,Dose route,"
        "Population,Pregnancy stage,Pediatric/Gestational age,Subject N,Note\n"
        "Buprenorphine,8,mg,Multiple,once daily,Sublingual,Maternal,Trimester 3,,10,day 1 of 3\n"
        "Buprenorphine,999,mg,Multiple,once daily,Sublingual,Maternal,Trimester 3,,10,x\n",
        encoding="utf-8")
    report = vc.check_validity(str(csv_path), "pk_drug_summary", [str(src)], "t", "a",
                            verify_fn=_clean_verify, empty_fn=None)
    assert report["tier2"]["checked_numbers"] == 4  # 2 rows x (Dose amount, Subject N)
    assert [f["cell"] for f in report["tier2"]["findings"]] == ["999"]
    assert report["escalate"] is True and report["error_rate"] == 0.25
