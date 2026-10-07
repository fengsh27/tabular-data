import json
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, ValidationError
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_summary.pk_sum_common_agent import PKSumCommonAgentResult

logger = logging.getLogger(__name__)

HEADER_CATEGORIZE_PROMPT = ChatPromptTemplate.from_template("""
The following table contains pharmacokinetics (PK) data:  
{processed_md_table_aligned}
{column_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all column headers and categorize each one into one of the following groups:  
   - **"Parameter type"**: Columns that describe the type of pharmacokinetic parameter.  
   - **"Parameter unit"**: Columns that **only** specify the unit of the parameter type. e.g. "fentanyl (ng/ml)" is not Parameter unit.  
   - **"Parameter value"**: Columns that contain numerical parameter values.  
   - **"P value"**: Columns that represent statistical P values.  
   - **"Uncategorized"**: Columns that do not fit into any of the above categories.  
(2) if a column is only about the subject number, it is considered as "Uncategorized"
(3) Return a categorized headers dictionary where each key is a column header, and the corresponding value is its assigned category, e.g.
{categorized_headers_example}

### **Output Format**
The output **must** exactly match the following format:
{{
  "categorized_headers": {{ "column_header_1": "category_1", "column_header_2": "category_2", ... }}
}}

Example:
{{
  "categorized_headers": {{ "Parameter type": "Parameter type","N": "Uncategorized","Range": "Parameter value","Mean ± s.d.": "Parameter value","Median": "Parameter value"}}
}}

""")


def get_header_categorize_prompt(md_table_aligned: str):
    df_table = markdown_to_dataframe(md_table_aligned)
    processed_md_table_aligned = display_md_table(md_table_aligned)
    column_headers_str = "These are all its column headers: " + ", ".join(
        f'"{col}"' for col in df_table.columns
    )
    categorized_headers_example = """```json\n{{"Parameter type": "Parameter type","N": "Uncategorized","Range": "Parameter value","Mean ± s.d.": "Parameter value","Median": "Parameter value"}}```"""
    return HEADER_CATEGORIZE_PROMPT.format(
        processed_md_table_aligned=processed_md_table_aligned,
        column_headers_str=column_headers_str,
        categorized_headers_example=categorized_headers_example,
    )


class HeaderCategorizeResult(PKSumCommonAgentResult):
    """Categorized results for headers"""

    categorized_headers: dict[str, str] = Field(
        description="""the dictionary represents the categorized result for headers. Each key is a column header, and the corresponding value is its assigned category (one of the values: "Parameter type", "Parameter unit", "Parameter value", "P value" and "Uncategorized")"""
    )


## It seems it's LangChain's bug when trying to convert HeaderCategorizeResult into a JSON schema. It throws error:
## Error code 400 - Invalid schema for response_format 'HeaderCategorizeResult': In context=(), 'required' is required to be
##     supplied and to be an array including every key in properties. Extra required key 'categorized_headers' supplied.
## So, here we introduce json schema
HeaderCategorizeJsonSchema = {
    "title": "HeaderCategorizeResult",
    "description": "Categorized results for headers",
    "type": "object",
    "properties": {
        "categorized_headers": {
            "type": "object",
            "description": 'the dictionary represents the categorized result for headers. Each key is a column header name, and the corresponding value is its assigned category string (one of the values: "Parameter type", "Parameter unit", "Parameter value", "P value" and "Uncategorized")',
            "title": "Categorized Headers",
        },
    },
    "required": ["categorized_headers"],
}


def _coerce_result(result: HeaderCategorizeResult | dict) -> HeaderCategorizeResult:
    if not isinstance(result, dict):
        return result
    try:
        # try parse the result
        if result.get("categorized_headers") is not None and isinstance(result.get("categorized_headers"), str):
            result["categorized_headers"] = json.loads(result["categorized_headers"])

        return HeaderCategorizeResult(**result)
    except json.JSONDecodeError as e:
        logger.error(e)
        raise RetryException(f"Invalid categorized headers: {result.get('categorized_headers')}")
    except ValidationError as e:
        logger.error(e)
        raise e


# A cell that reads as a number, a range, "mean (SD)", "x +/- y" or a percentage: digits
# and numeric punctuation only, no words. An equation such as "t1/2 = 0.693/k" or a unit
# does not match, so a table of abbreviations or formulas is not taken for data.
_NUMERIC_CELL = re.compile(r"^[\s<>≤≥~≈±+\-–−]*\d[\d\s.,()\[\]±/%–−\-+<>≤≥~≈:;]*$")
# rule (2) of the prompt: a column that is only about the subject number stays Uncategorized
_COUNT_HEADER = re.compile(
    r"^\W*(n|no|number(\s+of\s+\w+)*|subjects?|patients?|participants?|count)\W*$", re.IGNORECASE
)
_NO_VALUE_CELLS = ("", "N/A", "nan", "None")


def find_unlabeled_value_columns(match_dict: dict[str, str], md_table_aligned: str) -> list[str]:
    """Columns categorized "Uncategorized" whose cells are mostly numeric.

    A subject-count column (header "N", "Number of patients", ...) is not one of them.
    """
    df = markdown_to_dataframe(md_table_aligned)
    found = []
    for idx, col in enumerate(df.columns):
        if match_dict.get(col) != "Uncategorized" or _COUNT_HEADER.match(str(col).strip()):
            continue
        cells = [str(v).strip() for v in df.iloc[:, idx] if str(v).strip() not in _NO_VALUE_CELLS]
        if cells and sum(1 for c in cells if _NUMERIC_CELL.match(c)) / len(cells) >= 0.5:
            found.append(col)
    return found


def post_process_validate_categorized_result(
    result: HeaderCategorizeResult | dict,
    md_table_aligned: str,
) -> HeaderCategorizeResult:
    res = _coerce_result(result)
    # Ensure column count matches the table
    expected_columns = markdown_to_dataframe(md_table_aligned).shape[1]
    match_dict = res.categorized_headers
    if len(match_dict.keys()) != expected_columns:
        error_msg = f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict."
        logger.error(error_msg)
        raise RetryException(error_msg)

    # Ensure exactly one "Parameter type" column exists
    parameter_type_count = list(match_dict.values()).count("Parameter type")
    if parameter_type_count != 1:
        error_msg = f"Invalid mapping: Expected 1 'Parameter type' column, but found {parameter_type_count}."
        logger.error(error_msg)
        raise RetryException(error_msg)

    # SplitByColumnsStep builds one sub-table per "Parameter value" column, so a mapping
    # with none makes it return an empty list with no error and the table is silently lost
    # (6 whole papers in the pk-summary benchmark: gpt-4o and qwen3.6 labelled the numeric
    # data columns "Uncategorized"). Only ask again when the table does hold numeric
    # columns - a table of abbreviations or equations legitimately has no value column.
    if "Parameter value" not in match_dict.values():
        unlabeled = find_unlabeled_value_columns(match_dict, md_table_aligned)
        if unlabeled:
            error_msg = (
                f"No column was categorized as \"Parameter value\", but these columns hold "
                f"numerical values: {unlabeled}. A column of numerical results (means, medians, "
                "ranges, SD or CI, percentages) is \"Parameter value\" even when its header "
                "names a group, a dose or a time point; only a column that is just the subject "
                "number, or has no numbers, is \"Uncategorized\". Categorize the headers again."
            )
            logger.error(error_msg)
            raise RetryException(error_msg)

    return res


def try_fix_error_header_categories(
    res: HeaderCategorizeResult | dict,
    md_table_aligned: str,
) -> HeaderCategorizeResult | None:
    """Last-attempt fallback (retries exhausted) for a mapping with no "Parameter value".

    Label the numeric "Uncategorized" columns "Parameter value" - the same detection the
    validator asks the model to act on - so the table is not lost. Returns None when
    there is no such column, so any other failure still fails.
    """
    res = _coerce_result(res)
    unlabeled = find_unlabeled_value_columns(res.categorized_headers, md_table_aligned)
    if not unlabeled:
        return None
    fixed = {**res.categorized_headers, **{col: "Parameter value" for col in unlabeled}}
    return HeaderCategorizeResult(categorized_headers=fixed)
