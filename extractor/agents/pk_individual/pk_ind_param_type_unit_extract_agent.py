import re
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

logger = logging.getLogger(__name__)

UNIT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""

You are an expert in pharmacokinetics (PK) data interpretation and table normalization.

You are given:

**Main PK Table (reference only):**
{processed_md_table_aligned}

**Table Caption (reference only):**
{caption}

**Subtable 1 (PRIMARY input):**
{processed_md_sub_table}

The column **"{key_with_parameter_type}"** in Subtable 1 *approximately* represents the pharmacokinetic **parameter type**, but it may be incomplete, abbreviated, or ambiguous.

---

## Task

For **each row in Subtable 1 (rows 0 to {row_max_index}, inclusive)**, extract and construct **three aligned outputs**:

1. **Parameter type**
2. **Parameter unit**
3. **Parameter value**

You must return these as **three parallel lists**, preserving row order exactly.

---

## Extraction Rules (Follow in Order)

### Step 1 — Row Scope (STRICT)

* Process **only** rows `0 … {row_max_index}` from Subtable 1.
* The number of output entries **must exactly equal** the number of processed rows.
* Do not skip, add, reorder, or merge rows.

---

### Step 2 — Parameter Type Construction (PRIMARY FOCUS)

Use the column **"{key_with_parameter_type}"** as the starting point.

#### 2a. No-Loss Normalization Rule (MANDATORY)

* **Do NOT drop, simplify away, or merge informative components.**
* Preserve **all meaningful tokens**, including:

  * biological matrix (e.g., plasma, serum, milk)
  * subject/context (e.g., maternal, fetal, cord)
  * timing or condition (e.g., trough, peak)
  * method or sampling source

#### 2b. Composite Values

If the parameter type appears as a tuple, list, or nested structure:

* **Flatten and concatenate all elements in order**
* Use `"-"` as the delimiter
* Remove brackets and quotes only
* Preserve original wording and capitalization

**Examples:**

* `('Cordocentesis', 'Serum')` → `Cordocentesis-Serum`
* `('Maternal', 'Plasma', 'Trough')` → `Maternal-Plasma-Trough`
* `(('Umbilical', 'Vein'), 'Plasma')` → `Umbilical-Vein-Plasma`

If the parameter type is already a single string:

* Use it **as-is**, trimming surrounding whitespace only.

---

### Step 3 — Parameter Type Semantics

* Keep the **core PK concept clear and simple**, such as:

  * `Concentration`
  * `Cmax`
  * `Tmax`
  * `AUC`
* While doing so, **do NOT remove contextual qualifiers** required by the No-Loss Rule.

Correct example:

* `Cmax-Maternal-Plasma`
* `Concentration-Cordocentesis-Serum`

Incorrect example:

* Reducing everything to just `Concentration`

---

### Step 4 — Parameter Unit and Value Extraction

* Prefer values and units **explicitly present in Subtable 1**.
* If unit or value is ambiguous or incomplete:

  * Refer to the **main table and caption** ONLY to clarify or refine.
* Do **not infer or invent** values or units that are not supported by the provided tables.

---

### Step 5 — Missing or Unextractable Rows

* If **any** of the three fields (type, unit, value) cannot be confidently extracted for a row:

  * Output `"N/A"` for **all three fields** for that row.

---

## Output Format (STRICT)

Return **exactly one tuple** containing **three lists**:

```json
{{
  "extracted_param_units": {{
    "parameter_types": ["Parameter type 1", "Parameter type 2", ...],
    "parameter_units": ["Parameter unit 1", "Parameter unit 2", ...],
    "parameter_values": ["Parameter value 1", "Parameter value 2", ...]
  }}
}}
```

Constraints:

* All three lists must have **identical length**
* List indices must correspond to the same row in Subtable 1
* Output **only** the tuple — no explanations, no markdown, no extra text

---

## Priority Rules (Highest to Lowest)

1. Row count and alignment correctness
2. No-Loss Normalization Rule
3. Fidelity to Subtable 1
4. Clarification via main table/caption
5. Simplicity of PK terminology without information loss

---


""")


class ExtractedParamTypeUnits(BaseModel):
    # Defaulting to [] rather than requiring the key: a reply that omits
    # "parameter_units"/"parameter_values" entirely (33253437: 629 repeated
    # "parameter_types" entries and nothing else) used to fail Pydantic
    # validation before a result object even existed, so post_process and
    # try_fix_error never ran and the table was dropped with no retry at all.
    # An empty list still fails post_process_validate_matched_tuple's length
    # check (0 != expected_rows), so it is sent through the normal
    # retry-then-fallback path instead of skipping it.
    parameter_types: List[str] = Field(
        default_factory=list, description="Extracted 'Parameter type' values."
    )
    parameter_units: List[str] = Field(
        default_factory=list, description="Corresponding 'Parameter unit' values."
    )
    parameter_values: List[str] = Field(
        default_factory=list, description="Corresponding 'Parameter value' values."
    )


class ParamTypeUnitExtractionResult(PKIndCommonAgentResult):
    """Unit Extraction Result"""

    extracted_param_units: ExtractedParamTypeUnits = Field(
        description="An object with lists of extracted parameter types and their corresponding units and values."
    )


def get_param_type_unit_extraction_prompt(
    md_table_aligned: str, md_sub_table: str, col_mapping: dict, caption: str
) -> str | None:
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    # parameter_unit_count = list(match_dict.values()).count("Parameter unit")
    if parameter_type_count == 1:
        key_with_parameter_type = [
            key for key, value in col_mapping.items() if value == "Parameter type"
        ][0]
        return UNIT_EXTRACTION_PROMPT.format(
            processed_md_table_aligned=display_md_table(md_table_aligned),
            caption=caption,
            processed_md_sub_table=display_md_table(md_sub_table),
            key_with_parameter_type=key_with_parameter_type,
            row_max_index=markdown_to_dataframe(md_sub_table).shape[0] - 1,
        )
    return None


# def pre_process_param_type_unit(md_table: str, col_mapping: dict):
#     parameter_type_count = list(col_mapping.values()).count("Parameter type")
#     parameter_unit_count = list(col_mapping.values()).count("Parameter unit")
#     if parameter_type_count == 1 and parameter_unit_count == 0:
#         return True
#     return False


def post_process_validate_matched_tuple(
    res: ParamTypeUnitExtractionResult,
    md_table: str,
    col_mapping: dict,
):
    matched_tuple = (
        res.extracted_param_units.parameter_types,
        res.extracted_param_units.parameter_units,
        res.extracted_param_units.parameter_values,
    )
    expected_rows = markdown_to_dataframe(md_table).shape[0]
    if len(matched_tuple[0]) != expected_rows or len(matched_tuple[1]) != expected_rows or len(matched_tuple[2]) != expected_rows:
        error_msg = (
            f"Mismatch: Expected {expected_rows} rows, but got {len(matched_tuple[0])} (types) and {len(matched_tuple[1])} (units), and "
            f"{len(matched_tuple[2])} (values)"
        )
        logger.error(error_msg)
        raise RetryException(error_msg)

    return matched_tuple

# A "Name (unit)" parameter-type string, e.g. "Mother's PL III trimester (ng/ml)"
# -> unit "ng/ml". Matched against the end of the string so a type that is itself
# a parenthesised tuple repr (an unresolved multi-level header) still yields
# something rather than crashing; it just won't look like a real unit.
_TRAILING_UNIT = re.compile(r"\(([^()]*)\)\s*$")


def _unit_from_type(type_str: str) -> str:
    match = _TRAILING_UNIT.search(str(type_str))
    return match.group(1).strip() if match else "N/A"


def try_fix_error_param_type_unit(
    res: ParamTypeUnitExtractionResult,
    md_table: str,
    col_mapping: dict,
) -> Tuple[List[str], List[str], List[str]]:
    """Last-attempt fallback (retries exhausted): read the three lists from the
    sub-table itself, not from the model's reply.

    33253437: the sub-table's "Parameter type" column holds the same string on
    every one of its 55 rows (the split step put the original column name
    there - see pk_ind_split_by_col_step.py), and the model got stuck
    repeating that string ~630 times while never writing "Parameter unit" or
    "Parameter value" at all, identically on every retry (temperature 0).
    There is nothing in a reply like that worth trimming or padding.

    Every value needed is already in the sub-table: "Parameter type" and
    "Parameter value" are its own columns (always present and in that order -
    the split step's contract), and the unit is the trailing "(...)" in the
    type text, the same thing the model is asked to extract. This loses
    whatever normalization the model would otherwise do (e.g. flattening a
    tuple-style type into "A-B-C") - a deliberate last resort, not a
    substitute for the model doing that job when it can.
    """
    df = markdown_to_dataframe(md_table)
    types = df["Parameter type"].astype(str).tolist()
    values = df["Parameter value"].astype(str).tolist()
    units = [_unit_from_type(t) for t in types]
    return (types, units, values)