from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, ValidationError
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
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
        "reasoning_process": {
            "type": "string",
            "description": "A detailed explanation of the thought process or reasoning steps taken to reach a conclusion.",
            "title": "Reasoning Process",
        },
        "categorized_headers": {
            "type": "object",
            "description": 'the dictionary represents the categorized result for headers. Each key is a column header name, and the corresponding value is its assigned category string (one of the values: "Parameter type", "Parameter unit", "Parameter value", "P value" and "Uncategorized")',
            "title": "Categorized Headers",
        },
    },
    "required": ["categorized_headers"],
}


def post_process_validate_categorized_result(
    result: HeaderCategorizeResult | dict,
    md_table_aligned: str,
) -> HeaderCategorizeResult:
    if isinstance(result, dict):
        try:
            res = HeaderCategorizeResult(**result)
        except ValidationError as e:
            logger.error(e)
            raise e
    # Ensure column count matches the table
    expected_columns = markdown_to_dataframe(md_table_aligned).shape[1]
    match_dict = res.categorized_headers
    if len(match_dict.keys()) != expected_columns:
        error_msg = f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Ensure exactly one "Parameter type" column exists
    parameter_type_count = list(match_dict.values()).count("Parameter type")
    if parameter_type_count != 1:
        error_msg = f"Invalid mapping: Expected 1 'Parameter type' column, but found {parameter_type_count}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return res
