"""Tests for the simple-prompt parsers' row repairs (scripts/simple_prompt/
_common.py and _common_summary.py): `_realign` for a field-count mismatch and
`_reshift` for a count-correct row whose cells sit in the wrong columns.

The pk_summary case is the real Arm A reply for PMID 35465728 table 2: the
model dropped the empty `Pregnancy stage` cell and wrote an extra empty after
`median`, so every row had 16 fields with `60 / Initial Cavg / 12.9 / mg/L /
median` one column left of home. Gold has those 8 rows; the parsed table
scored F1 0.000.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "simple_prompt"))

summary = importlib.import_module("_common_summary")
individual = importlib.import_module("_common")

SUMMARY_HEADER = ",".join(summary.COLS)
INDIVIDUAL_HEADER = ",".join(individual.COLS)

# --- pk_summary ---------------------------------------------------------------

SHIFTED_35465728 = (
    "Vancomycin,Vancomycin,,,60,Initial Cavg,12.9,mg/L,median,,,,25th-75th percentiles,11.3,17.0,<0.001"
)


def test_reshift_repairs_the_35465728_row():
    row = SHIFTED_35465728.split(",")
    fixed = summary._reshift(row, summary.COLS)
    rec = dict(zip(summary.COLS, fixed))
    assert rec["Pregnancy stage"] == ""
    assert rec["Subject N"] == "60"
    assert rec["Parameter type"] == "Initial Cavg"
    assert rec["Value"] == "12.9"
    assert rec["Unit"] == "mg/L"
    assert rec["Summary statistics"] == "median"
    assert rec["Variation type"] == "" and rec["Variation value"] == ""
    assert rec["Interval type"] == "25th-75th percentiles"
    assert (rec["Lower limit"], rec["High limit"], rec["P value"]) == ("11.3", "17.0", "<0.001")


def test_parse_csv_applies_reshift_to_every_shifted_row():
    raw = "\n".join([
        SUMMARY_HEADER,
        SHIFTED_35465728,
        "Vancomycin,Vancomycin,,,59,Initial Cavg,20.3,mg/L,median,,,,25th-75th percentiles,17.0,22.2,<0.001",
    ])
    rows = summary.parse_csv(raw)
    assert [r["Subject N"] for r in rows] == ["60", "59"]
    assert [r["Value"] for r in rows] == ["12.9", "20.3"]
    assert all(r["Summary statistics"] == "median" for r in rows)


def test_reshift_leaves_a_correct_row_alone():
    row = ["Ketamine", "Ketamine", "Plasma", "children", "", "19", "AUC", "91",
           "Mg/min/L", "mean", "SD", "30", "", "", "", ""]
    assert summary._reshift(row, summary.COLS) is None
    assert summary.parse_csv(f"{SUMMARY_HEADER}\n{','.join(row)}")[0]["Value"] == "91"


def test_reshift_needs_a_hard_misfit_not_just_an_odd_label():
    # "Parturition" is not in the Pregnancy-stage vocabulary (-1), but nothing
    # is structurally wrong - no repair should be attempted.
    row = ["Ketamine", "Ketamine", "Plasma", "maternal", "Parturition", "19", "AUC", "91",
           "mg/L", "mean", "", "", "", "", "", ""]
    assert summary._reshift(row, summary.COLS) is None


def test_reshift_refuses_when_no_candidate_clears_the_misfit():
    # Text in Value with no empty cell anywhere to move: nothing to try.
    row = ["Ketamine", "Ketamine", "Plasma", "children", "delivery", "19", "AUC", "high",
           "mg/L", "mean", "SD", "30", "range", "1", "2", "0.05"]
    assert summary._reshift(row, summary.COLS) is None


def test_reshift_refuses_a_tie():
    # Text in Value, an empty on either side: moving it left or right scores
    # the same and displaces the same single cell, so neither wins outright.
    names = ["Analyte", "Value", "Specimen"]
    row = ["", "plasma", ""]
    assert summary._reshift(row, names) is None


def test_reshift_prefers_the_smallest_move_among_equal_fits():
    # The misplaced empty can be re-inserted anywhere among the free-text
    # columns with the same fit; the repair that moves fewest cells wins.
    row = SHIFTED_35465728.split(",")
    fixed = summary._reshift(row, summary.COLS)
    assert fixed[:4] == ["Vancomycin", "Vancomycin", "", ""]


def test_reshift_repairs_both_segments_of_the_16143486_row():
    # Real Arm A row for PMID 16143486 table 2: the head (Population..Unit)
    # and the tail (Interval type..P value) are both one column left - two
    # slips, so two moves. "95% CI" left in Lower limit counts as a misfit
    # (it starts with a digit but is text), and a one-move candidate that
    # leaves Lower limit empty with High limit filled loses to the full
    # repair through _row_consistency.
    row = "lorazepam,Lorazepam,,,parturients,delivery,,Cmax,12.96,ng/ml,mean,,,95% CI,9.42,16.49".split(",")
    assert summary._fits_strict("Lower limit", "95% CI") == -3
    rec = dict(zip(summary.COLS, summary._reshift(row, summary.COLS)))
    assert rec["Population"] == "parturients" and rec["Pregnancy stage"] == "delivery"
    assert rec["Subject N"] == ""
    assert rec["Parameter type"] == "Cmax" and rec["Value"] == "12.96" and rec["Unit"] == "ng/ml"
    assert rec["Summary statistics"] == "mean"
    assert rec["Interval type"] == "95% CI"
    assert (rec["Lower limit"], rec["High limit"], rec["P value"]) == ("9.42", "16.49", "")


def test_row_consistency_penalises_a_one_ended_interval():
    names = summary.COLS
    one_ended = dict.fromkeys(names, "")
    one_ended["High limit"] = "9.42"
    assert summary._row_consistency([one_ended[n] for n in names], names) == -2
    both = dict(one_ended, **{"Lower limit": "1.0"})
    assert summary._row_consistency([both[n] for n in names], names) == 0


def test_reshift_does_not_fire_on_an_age_group_in_a_text_column():
    # Real Arm A row for PMID 22050870 table 3: "3 Month to < 3 Years" starts
    # with a digit, but it is a label, not a misplaced number - nothing to fix.
    row = ["", "", "3 Month to < 3 Years", "", "", "17", "Free Fraction", "0.11", "",
           "mean", "SD", "0.10", "range", "0.07", "0.48", ""]
    assert summary._fits_strict("Specimen", "3 Month to < 3 Years") == 0
    assert summary._reshift(row, summary.COLS) is None


def test_realign_breaks_a_free_text_tie_by_extending_the_run_of_empties():
    # Second live round, PMID 35465728 table 2: 15 fields, the empty
    # `Pregnancy stage` dropped. Inserting the empty into the existing run
    # (",,," before 60) or between the two "Vancomycin" cells scores the same
    # (Analyte/Specimen are 0-voting); the row used to be dropped as a tie,
    # 4 of 8 rows, and the table escalated. The run-of-empties candidate wins.
    short = "Vancomycin,Vancomycin,,,60,Initial Cavg,12.9,mg/L,median,,,25th-75th percentiles,11.3,17.0,<0.001"
    fixed = summary._realign(short.split(","), summary.COLS)
    rec = dict(zip(summary.COLS, fixed))
    assert rec["Drug name"] == "Vancomycin" and rec["Analyte"] == "Vancomycin"
    assert rec["Specimen"] == "" and rec["Population"] == "" and rec["Pregnancy stage"] == ""
    assert rec["Subject N"] == "60" and rec["Value"] == "12.9" and rec["P value"] == "<0.001"
    dropped = []
    rows = summary.parse_csv(f"{SUMMARY_HEADER}\n{short}", dropped=dropped)
    assert len(rows) == 1 and dropped == []


def test_individual_realign_breaks_a_free_text_tie_the_same_way():
    short = "1,Venlafaxine,Venlafaxine,,maternal/pediatric,,M/PAUC,,2.40,,".split(",")  # 11 fields
    fixed = individual._realign(short, [(c, None) for c in individual.COLS])
    rec = dict(zip(individual.COLS, fixed))
    assert rec["Drug name"] == "Venlafaxine" and rec["Analyte"] == "Venlafaxine"
    assert rec["Population"] == "maternal/pediatric"
    assert rec["Parameter type"] == "M/PAUC" and rec["Parameter value"] == "2.40"


def test_realign_still_handles_a_short_row():
    # One interior empty dropped from the `median,,,IQR` run: 15 fields.
    short = "Vancomycin,Vancomycin,,,,60,Initial Cavg,12.9,mg/L,median,,IQR,11.3,17.0,0.05"
    rows = summary.parse_csv(f"{SUMMARY_HEADER}\n{short}")
    assert len(rows) == 1 and rows[0]["Value"] == "12.9" and rows[0]["P value"] == "0.05"
    assert rows[0]["Interval type"] == "IQR" and rows[0]["High limit"] == "17.0"


def test_headerless_rows_are_reshifted_too():
    raw = "\n".join([SHIFTED_35465728,
                     "Vancomycin,Vancomycin,,,59,Initial Cavg,20.3,mg/L,median,,,,25th-75th percentiles,17.0,22.2,<0.001"])
    rows = summary.parse_csv(raw)
    assert [r["Subject N"] for r in rows] == ["60", "59"]


# --- pk_individual ------------------------------------------------------------

SHIFTED_11849190 = "1,Venlafaxine,Venlafaxine,,,maternal/pediatric,,,M/PAUC,,2.40,"


def test_individual_reshift_repairs_the_11849190_row():
    # Real Arm A row for PMID 11849190 table 1 (12 of 48 rows had this shape):
    # the Population cell slid into Pregnancy stage, the parameter name into
    # Parameter unit and the value into Time value; Parameter value was blank.
    # Population's vocabulary (+2 for "maternal/pediatric") and the empty
    # Parameter type penalty decide the placement.
    fixed = individual._reshift(SHIFTED_11849190.split(","), individual.COLS)
    rec = dict(zip(individual.COLS, fixed))
    assert rec["Patient ID"] == "1"
    assert rec["Population"] == "maternal/pediatric" and rec["Pregnancy stage"] == ""
    assert rec["Parameter type"] == "M/PAUC"
    assert rec["Parameter unit"] == ""
    assert rec["Parameter value"] == "2.40"
    assert rec["Time value"] == "" and rec["Time unit"] == ""


def test_individual_reshift_without_vocabulary_evidence_still_recovers_the_value():
    # A shift that starts at Pediatric/Gestational age displaces no vocabulary
    # column, so the scorer cannot tell "Concentration" (a type) from "ng/mL"
    # (a unit) - both are free text. The repair is still accepted because it
    # puts the number back under Parameter value; type/unit placement is not
    # asserted because the evidence genuinely does not decide it.
    row = ["Case 1", "Lamotrigine", "Lamotrigine", "Plasma", "maternal", "3rd trimester",
           "Concentration", "ng/mL", "4.2", "", "", "h"]
    fixed = individual._reshift(row, individual.COLS)
    rec = dict(zip(individual.COLS, fixed))
    assert rec["Pregnancy stage"] == "3rd trimester"
    assert rec["Parameter value"] == "4.2"
    assert rec["Time value"] == "" and rec["Time unit"] == "h"
    assert {rec["Parameter type"], rec["Parameter unit"], rec["Pediatric/Gestational age"]} \
        >= {"Concentration", "ng/mL"}


def test_individual_parse_csv_reshifts_but_keeps_correct_rows():
    good = "2,Venlafaxine,Venlafaxine,,maternal/pediatric,,,M/PAUC,,2.05,,"
    rows = individual.parse_csv(f"{INDIVIDUAL_HEADER}\n{good}\n{SHIFTED_11849190}")
    assert [r["Parameter value"] for r in rows] == ["2.05", "2.40"]
    assert [r["Parameter type"] for r in rows] == ["M/PAUC", "M/PAUC"]
    assert [r["Population"] for r in rows] == ["maternal/pediatric", "maternal/pediatric"]
    assert [r["Patient ID"] for r in rows] == ["2", "1"]


def test_individual_wide_layout_is_not_reshifted():
    header = "Patient ID,Drug name,Analyte,Specimen,Population,Pregnancy stage,Pediatric/Gestational age,parameter1,unit1,value1,time1,timeunit1"
    row = "1,Drug,Drug,Plasma,maternal,,,Cmax,ng/mL,3.0,,"
    rows = individual.parse_csv(f"{header}\n{row}")
    assert len(rows) == 1 and rows[0]["Parameter value"] == "3.0"


# --- 2026-10-31, pk_summary round 6 (qwen3.8 Arm A): three ways a reply was lost ---

_KET = [  # 30825333 table 3: headerless, every row one trailing empty too wide (17 fields)
    "Ketamine,Ketamine,,children,,,Total area under curve,91,Mg/min/L,mean,SD,30,,,,,",
    "Ketamine,Ketamine,,children,,,Maximum concentration,1.6,mg/L,mean,SD,0.68,,,,,",
    "Ketamine,Ketamine,,children,,,Total body clearance,0.52,L/min,mean,SD,0.26,,,,,",
]
_INH_HEADER = ("Drug name,Analyte,Specimen,Pregnancy stage,Subject N,Parameter type,Value,Unit,"
               "Summary statistics,Variation type,Variation value,Interval type,Lower limit,High limit,P value")
_INH = [  # 34183327 table 2: a 15-name header (no Population) over 17-field rows
    "Isoniazid,Isoniazid,plasma,patients with tuberculous meningitis,,20,AUC0–24,18.5,h∙mg/L,geometric mean,,,,range,5.1,47.4,0.888",
    "Isoniazid,Isoniazid,plasma,patients with tuberculous meningitis,,12,AUC0–24,14.5,h∙mg/L,geometric mean,,,,range,5.9,44.2,0.888",
    "Isoniazid,Isoniazid,CSF,patients with tuberculous meningitis,,20,C_CSF0–2,1.4,mg/L,geometric mean,,,,range,0.5,6.1,",
    "Isoniazid,Isoniazid,CSF,patients with tuberculous meningitis,,12,C_CSF0–2,1.6,mg/L,geometric mean,,,,range,1.2,2.5,",
]


def test_summary_headerless_rows_with_a_trailing_empty_are_parsed():
    dropped = []
    rows = summary.parse_csv("\n".join(_KET), dropped=dropped)
    assert len(rows) == 3 and dropped == []
    assert rows[0]["Parameter type"] == "Total area under curve" and rows[0]["Value"] == "91"
    assert rows[0]["Unit"] == "Mg/min/L" and rows[0]["Variation value"] == "30" and rows[0]["P value"] == ""


def test_summary_headerless_non_csv_is_reported_as_dropped_not_lost():
    dropped = []
    rows = summary.parse_csv("The table lists AUC, Cmax, and t1/2 for ketamine.\nNo CSV here, really.", dropped=dropped)
    assert rows == [] and len(dropped) == 2  # both comma-bearing lines surface in Tier 1's count


def test_summary_short_header_over_full_width_rows_uses_the_canonical_schema():
    dropped = []
    rows = summary.parse_csv("\n".join([_INH_HEADER] + _INH), dropped=dropped)
    assert len(rows) == 4 and dropped == []
    for r in rows:
        assert r["Population"] == "patients with tuberculous meningitis" and r["Pregnancy stage"] == ""
        assert r["Interval type"] == "range" and r["Variation type"] == "" and r["Variation value"] == ""
    assert rows[0]["Subject N"] == "20" and rows[0]["Lower limit"] == "5.1" and rows[0]["High limit"] == "47.4" and rows[0]["P value"] == "0.888"
    assert rows[2]["Lower limit"] == "0.5" and rows[2]["High limit"] == "6.1" and rows[2]["P value"] == ""


def test_summary_header_only_reply_parses_to_nothing_and_nothing_dropped():
    dropped = []
    assert summary.parse_csv(",".join(summary.COLS), dropped=dropped) == [] and dropped == []


def test_individual_headerless_trailing_empty_and_non_csv_reporting():
    good = ",".join(["P1", "Drug", "Drug", "plasma", "adult", "", "", "Cmax", "ng/mL", "12.5", "2", "h"])
    dropped = []
    rows = individual.parse_csv("\n".join([good + ",", good + ","]), pmid="1", dropped=dropped)
    assert len(rows) == 2 and dropped == []
    dropped = []
    assert individual.parse_csv("prose, with commas, only\nand more, prose", pmid="1", dropped=dropped) == []
    assert len(dropped) == 2
