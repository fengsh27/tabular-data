"""Tier 3 rule table: parameter-type <-> unit/magnitude sanity.

Deliberately narrow and keyword-based, per docs/VERIFICATION_AGENT_DESIGN.md's
Tier 3 ("cheap to encode as a lookup table"): each rule matches a PK
parameter's common *abbreviation* (AUC, Cmax, Tmax, CL, Vd, t1/2), not every
spelled-out synonym a model might produce. This keeps false positives rare -
a parameter type the table doesn't recognize is simply left unchecked (same
"only flag what we have grounds to flag" principle Tier 2 already uses for
caption-only values) - at the cost of not catching every row. Widening the
match patterns is a deliberate case-by-case decision for whoever tunes this
table against real data, not something to do speculatively here.

Ranges are intentionally generous (order-of-magnitude sanity, not precise
pharmacology) - this tier exists to catch gross errors (sign flips, unit
mix-ups, stray extra digits), not to second-guess a real outlier.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

NA_TOKENS = {"", "n/a", "na", "nd", "none", "-", "–", "—"}


def _parse_number(cell: str):
    m = _NUMBER_RE.search(cell or "")
    return float(m.group()) if m else None


# Each rule: `match` finds the parameter type; `unit_ok` (if set) the unit
# must match; `value_range` (if set) the value must fall inside, inclusive.
# `unit_hint` is just for the finding's message.
RULES = [
    {
        "name": "half-life",
        "match": re.compile(r"half[\s-]?life|\bt1\s*/\s*2\b", re.I),
        "unit_ok": re.compile(r"\b(h|hr|hrs|hour|hours|min|mins|minute|minutes|day|days)\b", re.I),
        "unit_hint": "a time unit (h, min, day)",
        "value_range": (0, 500),
    },
    {
        "name": "clearance",
        "match": re.compile(r"\bclearance\b|\bCL(?:/F)?\b", re.I),
        "unit_ok": re.compile(r"\b(l|ml)\s*/\s*(h|hr|hrs|hour|hours|min|kg)\b", re.I),
        "unit_hint": "volume/time (e.g. L/h, mL/min/kg)",
        "value_range": (0, 10000),
    },
    {
        "name": "volume of distribution",
        "match": re.compile(r"\bV(?:d|ss|z)(?:/F)?\b", re.I),
        "unit_ok": re.compile(r"\b(l|ml)\b", re.I),
        "unit_hint": "a volume unit (L, mL)",
        "value_range": (0, 10000),
    },
    {
        "name": "AUC",
        "match": re.compile(r"\bAUC\b", re.I),
        "unit_ok": re.compile(
            r"(h|hr|hrs|hour|hours|min)\s*[*x·.]?\s*(ng|ug|mcg|mg|pg)|"
            r"(ng|ug|mcg|mg|pg)\W{0,3}(h|hr|hrs|hour|hours|min)\b", re.I,
        ),
        "unit_hint": "concentration * time (e.g. ng*h/mL, mg.h/L)",
        "value_range": (0, 1_000_000),
    },
    {
        "name": "Cmax",
        "match": re.compile(r"\bCmax\b", re.I),
        "unit_ok": re.compile(r"\b(ng|ug|mcg|mg|pg)\s*/\s*(ml|l|dl)\b", re.I),
        "unit_hint": "a concentration unit (ng/mL, mg/L)",
        "value_range": (0, 100_000),
    },
    {
        "name": "Tmax",
        "match": re.compile(r"\bTmax\b", re.I),
        "unit_ok": re.compile(r"\b(h|hr|hrs|hour|hours|min|mins|minute|minutes)\b", re.I),
        "unit_hint": "a time unit (h, min)",
        "value_range": (0, 72),
    },
]


def _find_rule(parameter_type: str):
    for rule in RULES:
        if rule["match"].search(parameter_type):
            return rule
    return None


def check_row(parameter_type: str, unit: str, value: str) -> list[str] | None:
    """Return a list of plausibility issues for one row, or None if either no
    rule applies or the row checks out clean."""
    rule = _find_rule((parameter_type or "").strip())
    if rule is None:
        return None

    issues = []
    unit = (unit or "").strip()
    if rule["unit_ok"] is not None and unit and unit.lower() not in NA_TOKENS:
        if not rule["unit_ok"].search(unit):
            issues.append(f"unit '{unit}' is not typical for {rule['name']} "
                           f"(expected {rule['unit_hint']})")

    value = (value or "").strip()
    if rule["value_range"] is not None and value and value.lower() not in NA_TOKENS:
        num = _parse_number(value)
        if num is not None:
            lo, hi = rule["value_range"]
            if not (lo <= num <= hi):
                issues.append(f"value {num:g} is outside the plausible range "
                               f"for {rule['name']} ({lo:g}-{hi:g})")

    return issues or None
