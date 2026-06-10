"""Deterministic structural tests for the bundled pk-pe-curation skill (CI-safe).

The model-driven stages can't be unit-tested, but the skill is also a set of
*files* with internal references and declared output schemas — and those CAN rot
silently as prompts are edited. After bundling, there is ONE top-level skill
(`pk-pe-curation/SKILL.md`) and ten pipeline sub-procedures under `pipelines/`,
plus the `route`/`prepare-paper` procedures and the shared `curation-common/`.

Guards:
  1. The top SKILL.md has valid front-matter (`name` == pk-pe-curation, non-empty
     `description`).
  2. Every `prompts/NN_*.md` a pipeline's procedure.md references exists on disk.
  3. Every bundle-relative path (`curation-common/...`, `pipelines/...`) referenced
     anywhere in the bundle resolves to a real file — and NO stale repo-relative
     `skills/<old-name>/` reference survives the bundling.
  4. Each pipeline's declared output-schema table matches a known column list.

Run:  poetry run pytest skills_e2e_tests/test_skill_structure.py
"""
import os
import re

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLE = os.path.join(REPO_ROOT, "skills", "pk-pe-curation")
PIPELINES_DIR = os.path.join(BUNDLE, "pipelines")

# The output schema each pipeline must declare, verbatim and in order. route and
# prepare-paper produce files (no column schema) and are intentionally absent.
EXPECTED_SCHEMAS = {
    "pk-summary-curation": [
        "Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage",
        "Pediatric/Gestational age", "Subject N", "Parameter type",
        "Parameter unit", "Parameter value", "Parameter statistic",
        "Variation type", "Variation value", "Interval type", "Lower bound",
        "Upper bound", "P value", "Time value", "Time unit",
    ],
    "pk-individual-curation": [
        "Patient ID", "Drug name", "Analyte", "Specimen", "Population",
        "Pregnancy stage", "Pediatric/Gestational age", "Parameter type",
        "Parameter unit", "Parameter value", "Time value", "Time unit",
    ],
    "pk-drug-summary": [
        "Drug/Metabolite name", "Dose amount", "Dose unit", "Dose frequency",
        "Dose schedule", "Dose route", "Population", "Pregnancy stage",
        "Pediatric/Gestational age", "Population N", "Note",
    ],
    "pk-drug-individual": [
        "Patient ID", "Drug/Metabolite name", "Dose amount", "Dose unit",
        "Dose frequency", "Dose schedule", "Dose route", "Population",
        "Pregnancy stage", "Pediatric/Gestational age", "Source text",
    ],
    "pk-specimen-summary": [
        "Specimen", "Sample N", "Population", "Pregnancy stage",
        "Pediatric/Gestational age", "Population N", "Sample time", "Time unit",
        "Note",
    ],
    "pk-specimen-individual": [
        "Patient ID", "Specimen", "Sample N", "Population", "Pregnancy stage",
        "Pediatric/Gestational age", "Sample time", "Time unit", "Note",
    ],
    "pk-population-summary": [
        "Characteristic", "Characteristic subcategory", "Characteristic unit",
        "Characteristic value", "Statistics type", "Variation type",
        "Variation value", "Interval type", "Lower bound", "Upper bound",
        "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N",
        "Note",
    ],
    "pk-population-individual": [
        "Patient ID", "Characteristic", "Characteristic subcategory",
        "Characteristic unit", "Characteristic value", "Population",
        "Pregnancy stage", "Pediatric/Gestational age", "Note",
    ],
    "pe-study-info": [
        "Study type", "Population", "Study design", "Pregnancy stage",
        "Drug name", "Data source", "Inclusion criteria", "Exclusion criteria",
        "Outcomes", "Subject N",
    ],
    "pe-study-outcome": [
        "Characteristic", "Exposure", "Outcome", "Parameter unit",
        "Parameter statistic", "Parameter value", "Variation type",
        "Variation value", "Interval type", "Lower bound", "Upper bound",
        "P value",
    ],
}

# old top-level names that must NOT survive as `skills/<name>/` after bundling
_STALE_NAMES = sorted(set(EXPECTED_SCHEMAS) | {
    "curation-common", "pk-pe-route", "prepare-paper",
})


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def discover_pipelines():
    if not os.path.isdir(PIPELINES_DIR):
        return []
    return sorted(
        name for name in os.listdir(PIPELINES_DIR)
        if os.path.isfile(os.path.join(PIPELINES_DIR, name, "procedure.md"))
    )


def _all_bundle_md():
    """Every .md file under the bundle, as (path, text)."""
    out = []
    for root, _dirs, names in os.walk(BUNDLE):
        for n in names:
            if n.endswith(".md"):
                p = os.path.join(root, n)
                out.append((p, _read(p)))
    return out


def _parse_schema_columns(text):
    """Extract ordered column names from an output-schema table.

    Rows look like `| 1 | Column name | notes |`; the integer first cell marks a
    schema row, and we take the second cell.
    """
    cols = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            cols.append(m.group(2).strip())
    return cols


PIPELINES = discover_pipelines()


def test_top_skill_has_valid_frontmatter():
    text = _read(os.path.join(BUNDLE, "SKILL.md"))
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "pk-pe-curation/SKILL.md is missing '---' front-matter"
    fm = m.group(1)
    name_m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
    assert name_m and name_m.group(1).strip() == "pk-pe-curation", (
        "top SKILL.md front-matter name must be 'pk-pe-curation'"
    )
    desc_m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
    assert desc_m and desc_m.group(1).strip(), "top SKILL.md has empty/missing description"


def test_all_ten_pipelines_present():
    for p in EXPECTED_SCHEMAS:
        assert p in PIPELINES, f"expected pipeline {p} not found under pipelines/"


@pytest.mark.parametrize("pipeline", sorted(EXPECTED_SCHEMAS))
def test_referenced_prompt_files_exist(pipeline):
    """Every `prompts/<file>.md` named in a pipeline's procedure.md must exist."""
    proc_dir = os.path.join(PIPELINES_DIR, pipeline)
    refs = set(re.findall(r"prompts/([\w./-]+\.md)", _read(os.path.join(proc_dir, "procedure.md"))))
    if not refs:
        pytest.skip(f"{pipeline} references no prompts/ files")
    for ref in sorted(refs):
        assert os.path.isfile(os.path.join(proc_dir, "prompts", ref)), (
            f"{pipeline}/procedure.md references prompts/{ref} which does not exist"
        )


def test_bundle_relative_paths_resolve():
    """Every `curation-common/...` / `pipelines/...` path in the bundle resolves."""
    pat = re.compile(r"(?:curation-common|pipelines)/[\w./-]+\.(?:py|md|txt)")
    missing = []
    for path, text in _all_bundle_md():
        for ref in set(pat.findall(text)):
            if not os.path.isfile(os.path.join(BUNDLE, ref)):
                missing.append(f"{os.path.relpath(path, REPO_ROOT)} -> {ref}")
    assert not missing, "unresolved bundle-relative references:\n" + "\n".join(sorted(missing))


def test_no_stale_skills_prefix_references():
    """No `skills/<old-name>/` reference should survive bundling (would break once
    installed under .claude/skills/pk-pe-curation/)."""
    stale_pat = re.compile(r"skills/(?:" + "|".join(map(re.escape, _STALE_NAMES)) + r")/")
    offenders = []
    for path, text in _all_bundle_md():
        if stale_pat.search(text):
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, "stale 'skills/<old-name>/' references remain in:\n" + "\n".join(sorted(offenders))


@pytest.mark.parametrize("pipeline", sorted(EXPECTED_SCHEMAS))
def test_output_schema_matches_expected(pipeline):
    cols = _parse_schema_columns(_read(os.path.join(PIPELINES_DIR, pipeline, "procedure.md")))
    assert cols == EXPECTED_SCHEMAS[pipeline], (
        f"{pipeline}: declared output schema drifted.\n"
        f"  declared: {cols}\n  expected: {EXPECTED_SCHEMAS[pipeline]}"
    )
