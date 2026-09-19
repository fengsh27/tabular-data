"""Guards for the "reply envelope" class of qwen3.6-via-Ollama failures.

Ollama only enforces `format=<schema>` in some think modes. When it does not, a
model copies the output example from the prompt. If the example is a bare list
(or uses another key than the schema's field), the reply fails to parse and the
identical retries can never recover (temperature 0).

Two layers of defence, both tested here:
1. prompts show the schema's object envelope (static scan of every agent prompt);
2. `fix_reply_shape_for_single_field_schema` repairs a wrong envelope at parse time.
"""
import ast
import importlib
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from extractor.agents.common_agent.common_agent_ollama import (
    fix_reply_shape_for_single_field_schema,
)

AGENTS_ROOT = Path(__file__).resolve().parent.parent / "extractor" / "agents"

# pe_study_outcome (v1) is deprecated and intentionally not maintained.
SKIP_DIRS = {"pk_pe_agents", "pe_study_outcome", "common_agent", "__pycache__"}

AGENT_FILES = sorted(
    p
    for p in AGENTS_ROOT.rglob("*_agent.py")
    if not (set(p.relative_to(AGENTS_ROOT).parts) & SKIP_DIRS)
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(AGENTS_ROOT.parent.parent).with_suffix("")
    return ".".join(rel.parts)


def _prompt_texts(path: Path) -> list[str]:
    """Long string constants of a module, with `{{`/`}}` template escapes undone."""
    tree = ast.parse(path.read_text())
    return [
        n.value.replace("{{", "{").replace("}}", "}")
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 300
    ]


def _top_level_schema_fields(module) -> set[str]:
    """Field names of the module's result schemas, excluding nested sub-models."""
    models = {
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == module.__name__
    }
    nested = {
        ann
        for m in models
        for f in m.model_fields.values()
        for ann in [f.annotation]
        if ann in models
    }
    return {name for m in models - nested for name in m.model_fields}


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.name)
def test_prompt_examples_use_the_schema_envelope(path):
    texts = _prompt_texts(path)
    if not texts:
        pytest.skip("no prompt in module")

    # 1. no example line may be a bare list (`[["a", ...], ...]`, `[0, 1, ...]`)
    for text in texts:
        for line in text.splitlines():
            # `[[...` (list of lists) or `[matched_index_row_0, ...` / `[0, 1, ...`
            # (list of non-strings); a line starting `["` is a row inside an object.
            assert not re.match(r"^\s*`?\[(\[|[^\"\s])", line), (
                f"{path.name}: bare-list output example {line.strip()[:80]!r}; "
                'show it as {"<schema field>": [[...]]}'
            )
            assert not re.search(r"output format:\s*\[", line, re.I), (
                f"{path.name}: bare-list output example {line.strip()[:80]!r}"
            )

    # 2. an object example's top-level key must be a field of the module's schema
    module = importlib.import_module(_module_name(path))
    fields = _top_level_schema_fields(module)
    if not fields:
        return
    for text in texts:
        for line in text.splitlines():
            m = re.match(r'^\s*`?\{\s*"(\w+)"\s*:', line)
            if m and m.group(1) not in fields and "{" not in line[m.end():m.end() + 1]:
                pytest.fail(
                    f"{path.name}: example key {m.group(1)!r} is not a schema field "
                    f"{sorted(fields)}: {line.strip()[:90]!r}"
                )


# --------------------------------------------------------------------------
# parse-time fallback
# --------------------------------------------------------------------------


def _schemas():
    from extractor.agents.pk_individual.pk_ind_drug_info_agent import (
        DrugInfoResult,
    )
    from extractor.agents.pk_individual.pk_ind_time_unit_agent import (
        TimeAndUnitResult,
    )
    from extractor.agents.pk_individual.pk_ind_patient_matching_agent import (
        MatchedPatientResult,
    )
    from extractor.agents.pk_summary.pk_sum_drug_matching_agent import (
        MatchedDrugResult,
    )
    from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_agent import (
        ParamTypeUnitExtractionResult,
    )
    from extractor.agents.pk_specimen_summary.pk_spec_sum_time_unit_agent import (
        TimeAndUnitResult as SpecSumTimeAndUnitResult,
    )

    return dict(
        drug=DrugInfoResult,
        time=TimeAndUnitResult,
        patient_match=MatchedPatientResult,
        drug_match=MatchedDrugResult,
        unit=ParamTypeUnitExtractionResult,
        spec_time=SpecSumTimeAndUnitResult,
    )


def test_bare_list_is_wrapped():
    s = _schemas()
    res = fix_reply_shape_for_single_field_schema(
        '[["Lorazepam", "Lorazepam", "Plasma"]]', s["drug"]
    )
    assert res.drug_combinations == [["Lorazepam", "Lorazepam", "Plasma"]]

    res = fix_reply_shape_for_single_field_schema("[0, 1, 1, 2]", s["patient_match"])
    assert res.matched_row_indices == [0, 1, 1, 2]


def test_fenced_and_indented_bare_list_is_wrapped():
    s = _schemas()
    reply = '```json\n[\n  ["0-1", "Hour"],\n  ["N/A", "N/A"]\n]\n```'
    res = fix_reply_shape_for_single_field_schema(reply, s["time"])
    assert res.times_and_units == [["0-1", "Hour"], ["N/A", "N/A"]]


def test_wrong_key_is_remapped():
    s = _schemas()
    res = fix_reply_shape_for_single_field_schema(
        '{"matching_row_indices": [0, 0, 1]}', s["drug_match"]
    )
    assert res.matched_row_indices == [0, 0, 1]

    res = fix_reply_shape_for_single_field_schema(
        '{"time_and_units": [["0", "Hour", "src"]]}', s["spec_time"]
    )
    assert res.times_and_units == [["0", "Hour", "src"]]


def test_missing_nested_wrapper_is_added():
    s = _schemas()
    res = fix_reply_shape_for_single_field_schema(
        '{"parameter_types": ["Cmax"], "parameter_units": ["ng/mL"]}', s["unit"]
    )
    assert res.extracted_param_units.parameter_types == ["Cmax"]
    assert res.extracted_param_units.parameter_units == ["ng/mL"]


@pytest.mark.parametrize(
    "schema_key, reply",
    [
        ("drug", "not json at all"),
        ("drug", "[1, 2, 3]"),  # payload does not validate as list[list[str]]
        ("patient_match", '["a", "b"]'),  # not integers
        ("patient_match", '{"matched_row_indices": ["x"]}'),  # right envelope, bad data: don't mask
        ("drug", '{"a": 1, "b": 2}'),  # unrelated object
        ("drug", ""),
    ],
)
def test_unrepairable_replies_return_none(schema_key, reply):
    assert fix_reply_shape_for_single_field_schema(reply, _schemas()[schema_key]) is None


def test_multi_field_and_non_model_schemas_are_left_alone():
    from extractor.agents.pk_individual.pk_ind_summary_data_del_agent import (
        SummaryDataDelResult,
    )

    assert fix_reply_shape_for_single_field_schema("[1, 2]", SummaryDataDelResult) is None
    assert fix_reply_shape_for_single_field_schema("[1, 2]", {"type": "object"}) is None
