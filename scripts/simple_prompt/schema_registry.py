"""Which parser module handles which schema.

    load_schema_module("pk_individual")   -> the hand-written _common module
    load_schema_module("pk_summary")      -> _common_summary
    load_schema_module("pk_drug_summary") -> a SchemaParser built from schema_specs.SPECS

All three expose COLS / NUMERIC_COLS / _TEXT_COLS / _VALUE_RE / _fits /
parse_csv / write_csv, which is the whole surface run_simple_prompt.py,
validity.py and extractor/curate_agent/arms.py rely on.
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_specs import ALL_SCHEMAS, SPECS, UNIT_KINDS  # noqa: E402,F401

LEGACY_MODULES = {"pk_individual": "_common", "pk_summary": "_common_summary"}

_PARSERS: dict[str, object] = {}


def load_schema_module(schema: str):
    if schema in LEGACY_MODULES:
        return importlib.import_module(LEGACY_MODULES[schema])
    if schema not in SPECS:
        raise ValueError(f"unknown schema {schema!r}; expected one of {sorted(ALL_SCHEMAS)}")
    parser = _PARSERS.get(schema)
    if parser is None:
        from _common_generic import SchemaParser
        parser = _PARSERS[schema] = SchemaParser(SPECS[schema])
    return parser


def is_generic(schema: str) -> bool:
    return schema in SPECS
