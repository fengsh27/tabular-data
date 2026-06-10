"""Deterministic regression test for the route label->procedure map.

The route stage emits pipeline labels (PipelineTypeEnum values); turning them into
the sub-procedure to follow is mechanical but the naming is irregular (`pk_summary`
-> `pk-summary-curation`, but `pk_drug_summary` -> `pk-drug-summary`), so it must be
a fixed, tested table — not a string transform. This guards:
  1. the map covers EVERY PipelineTypeEnum value (no pipeline silently unroutable);
  2. every target procedure.md exists in the bundle (no dangling path);
  3. resolve() preserves order, de-dupes, and rejects unknown labels;
  4. the known irregular cases map correctly.

Run:  poetry run pytest skills_e2e_tests/test_pipeline_skill_map.py
"""
import importlib.util
import os
import re

import pytest

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
BUNDLE = os.path.join(REPO_ROOT, "skills", "pk-pe-curation")
SCRIPT = os.path.join(BUNDLE, "pipelines", "route", "scripts", "pipeline_skill_map.py")


def _pipeline_enum_values():
    """Parse the PipelineTypeEnum string values from extractor/constants.py source.

    Parsing the source (rather than importing extractor, which is not on pytest's
    path and pulls heavy deps) keeps this test dep-free and CI-safe while still
    cross-checking the real source of truth.
    """
    src = open(os.path.join(REPO_ROOT, "extractor", "constants.py"), encoding="utf-8").read()
    block = re.search(r"class\s+PipelineTypeEnum\b.*?:\n(.*?)(?:\n\S|\nclass |\Z)", src, re.DOTALL)
    assert block, "could not locate PipelineTypeEnum in extractor/constants.py"
    return set(re.findall(r'^\s+[A-Z_]+\s*=\s*"([^"]+)"', block.group(1), re.MULTILINE))


def load_script():
    spec = importlib.util.spec_from_file_location("pipeline_skill_map", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load_script()


def test_map_covers_every_pipeline_enum():
    """Every PipelineTypeEnum value must be routable."""
    enum_values = _pipeline_enum_values()
    assert len(enum_values) == 10, f"expected 10 pipelines, parsed {enum_values}"
    assert set(m.PIPELINE_TO_PROCEDURE) == enum_values, (
        "label->procedure map is out of sync with PipelineTypeEnum.\n"
        f"  missing from map: {enum_values - set(m.PIPELINE_TO_PROCEDURE)}\n"
        f"  extra in map:     {set(m.PIPELINE_TO_PROCEDURE) - enum_values}"
    )


def test_every_target_procedure_exists():
    for label, proc in m.PIPELINE_TO_PROCEDURE.items():
        path = os.path.join(BUNDLE, proc, "procedure.md")
        assert os.path.isfile(path), f"{label} -> {proc}: procedure.md not found"


def test_irregular_names_map_correctly():
    # the two table PK pipelines carry a -curation suffix; the full-text ones do not
    assert m.PIPELINE_TO_PROCEDURE["pk_summary"] == "pipelines/pk-summary-curation"
    assert m.PIPELINE_TO_PROCEDURE["pk_individual"] == "pipelines/pk-individual-curation"
    assert m.PIPELINE_TO_PROCEDURE["pk_drug_summary"] == "pipelines/pk-drug-summary"
    assert m.PIPELINE_TO_PROCEDURE["pe_study_outcome"] == "pipelines/pe-study-outcome"


def test_resolve_preserves_order_and_dedupes():
    out = m.resolve(["pe_study_outcome", "pk_summary", "pk_summary"])
    assert out == [
        {"pipeline": "pe_study_outcome", "procedure": "pipelines/pe-study-outcome"},
        {"pipeline": "pk_summary", "procedure": "pipelines/pk-summary-curation"},
    ]


def test_resolve_ignores_blanks():
    assert m.resolve(["", "  ", "pk_summary"]) == [
        {"pipeline": "pk_summary", "procedure": "pipelines/pk-summary-curation"},
    ]


def test_resolve_rejects_unknown_label():
    with pytest.raises(KeyError):
        m.resolve(["pk_summary", "pk_bogus"])


def test_build_selection_structure():
    sel = m.build_selection("12345678", "Both", ["pk_summary", "pe_study_outcome"])
    assert sel == {
        "pmid": "12345678",
        "paper_type": "Both",
        "selected": [
            {"pipeline": "pk_summary", "procedure": "pipelines/pk-summary-curation"},
            {"pipeline": "pe_study_outcome", "procedure": "pipelines/pe-study-outcome"},
        ],
    }


def test_empty_selection():
    assert m.build_selection("x", "Neither", []) == {
        "pmid": "x", "paper_type": "Neither", "selected": [],
    }
