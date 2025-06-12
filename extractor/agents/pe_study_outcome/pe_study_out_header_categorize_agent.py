from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, ValidationError
import logging

from TabFuncFlow.utils.table_utils import markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pe_study_outcome.pe_study_out_common_agent import PEStudyOutCommonAgentResult

logger = logging.getLogger(__name__)


HEADER_CATEGORIZE_PROMPT = ChatPromptTemplate.from_template("""
The following table contains Pharmacoepidemiology data:  
{processed_md_table}
{column_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all column headers and categorize each one into one of the following groups:  
    - **Characteristic**: Any geographic, demographic or biological feature of the subjects in the study (e.g., age, sex, race, weight, genetic markers).
    - **Exposure**: Any factor that might be associated with an outcome of interest (e.g., drugs, medical conditions, medications, etc.). If the column header name includes a drug name, it is very likely to be classified as "Exposure".
    - **Outcome**: A measure or set of measures that reflect the effects or consequences of the exposure (e.g., drug treatment, condition, or intervention). These are typically endpoints used to assess the impact of the exposure on the subjects, such as birth weight, symptom reduction, or lab results. In the context of the study outcomes sheet, this category should include all such relevant measures and their associated statistics.    
    - **P value**: Columns that represent statistical P values.  
    - **Row headers**: If the column contains row labels.
    - **Uncategorized**: Any column that does not clearly fit into the above categories. 
(2) Please note that, when encountering terms like "Overall", check which category the next column of "Overall" belong to, and categorize it accordingly. Because "Overall" and the column next to it always have the same category.
    -  This rule is critically important. Misclassifying these entries will directly impact the accuracy of subsequent processing, so please handle such cases with the utmost care and precision.
(3) Return a categorized headers dictionary where each key is a column header, and the corresponding value is its assigned category, e.g.
{categorized_headers_example}
""")


def get_header_categorize_prompt(md_table: str):
    df_table = markdown_to_dataframe(md_table)
    processed_md_table = display_md_table(md_table)
    column_headers_str = "These are all its column headers: " + ", ".join(
        f'"{col}"' for col in df_table.columns
    )
    categorized_headers_example = """```json\n{{"Unnamed:0": "Row headers", "Race:Overall": "Characteristic", "Race:White": "Characteristic", "Race:Native American": "Characteristic", "Unexposed": "Exposure", "Fentanyl Exposed": "Exposure", "Lorazepam Exposed": "Exposure", "Preterm delivery": "Outcome", "Birth weight": "Outcome", "Pain score": "Outcome", "P-value": "P value"}}```"""
    return HEADER_CATEGORIZE_PROMPT.format(
        processed_md_table=processed_md_table,
        column_headers_str=column_headers_str,
        categorized_headers_example=categorized_headers_example,
    )


class HeaderCategorizeResult(PEStudyOutCommonAgentResult):
    """Categorized results for headers"""

    categorized_headers: dict[str, str] = Field(
        description="""the dictionary represents the categorized result for headers. Each key is a column header, and the corresponding value is its assigned category (one of the values: "Characteristic", "Exposure", "Outcome", "P value", "Row headers" and "Uncategorized")"""
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
            "description": 'the dictionary represents the categorized result for headers. Each key is a column header name, and the corresponding value is its assigned category string (one of the values: "Characteristic", "Exposure", "Outcome", "P value", "Row headers" and "Uncategorized")',
            "title": "Categorized Headers",
        },
    },
    "required": ["categorized_headers"],
}


def post_process_validate_categorized_result(
    result: HeaderCategorizeResult | dict,
    md_table: str,
) -> HeaderCategorizeResult:
    if isinstance(result, dict):
        try:
            res = HeaderCategorizeResult(**result)
        except ValidationError as e:
            logger.error(e)
            raise e
    # Ensure column count matches the table
    expected_columns = markdown_to_dataframe(md_table).shape[1]
    match_dict = res.categorized_headers
    if len(match_dict.keys()) != expected_columns:
        error_msg = f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return res
