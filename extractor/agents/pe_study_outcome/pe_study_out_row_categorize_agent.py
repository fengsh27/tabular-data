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
{row_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all row headers and categorize each one into one of the following groups:  
    - **Characteristic**: Any geographic, demographic or biological feature of the subjects in the study (e.g., age, sex, race, weight, genetic markers).
    - **Exposure**: Any factor that might be associated with an outcome of interest (e.g., drugs, medical conditions, medications, etc.).
    - **Outcome**: A measure or set of measures that reflect the effects or consequences of the exposure (e.g., drug treatment, condition, or intervention). These are typically endpoints used to assess the impact of the exposure on the subjects, such as birth weight, symptom reduction, or lab results. In the context of the study outcomes sheet, this category should include all such relevant measures and their associated statistics.    
    - **Subheader**: A label that introduces or groups related rows under a common theme, but is not itself a data variable (e.g., "Pruritus", "Nausea", "Pain scores"). Subheaders serve as visual or structural separators within the table.
    - **Uncategorized**: Any row that does not clearly fit into the above categories. 
(2) You must first identify the Subheaders.
    For example, consider the following excerpt from a table:
    ...
    |Fentanyl| | |
    | Used | 10(10%) | |
    | NOT Used | 5(5%) | |
    ...
    In this case, the row "Fentanyl" should be labeled as a Subheader, while "Used" and "NOT Used" should be categorized as Exposure.
    Do not directly classify "Fentanyl" as "Exposure", even though it appears to refer to a drug.
    Instead, base your labeling on the role the row header plays within the overall table structure. Subheaders are typically used to group related variables and should be labeled accordingly.
(3) Return a categorized headers dictionary where each key is a row header, and the corresponding value is its assigned category, e.g.
{{
"categorized_headers": {{"Race:Overall": "Characteristic", "Race": "Subtitle", "White": "Characteristic",
    "Native American": "Characteristic", "Exposure": "Exposure", "Unexposed": "Exposure",
    "Fentanyl Exposed": "Exposure", "Lorazepam Exposed": "Exposure", "Preterm delivery": "Outcome",
    "Birth weight": "Outcome", "Pain score": "Outcome", "Education": "Uncategorized", "Income": "Uncategorized"}} 
}} (example)
""")


def get_row_categorize_prompt(md_table: str, row_header_name: str):
    df_table = markdown_to_dataframe(md_table)
    processed_md_table = display_md_table(md_table)

    if row_header_name not in df_table.columns:
        raise ValueError(f'"{row_header_name}" not found in table columns.')

    row_headers_str = "These are all its row headers: " + ", ".join(
        f'row {i}: "{val}"' for i, val in enumerate(df_table[row_header_name])
    )

    return HEADER_CATEGORIZE_PROMPT.format(
        processed_md_table=processed_md_table,
        row_headers_str=row_headers_str,
    )


class RowCategorizeResult(PEStudyOutCommonAgentResult):
    """Categorized results for headers"""

    categorized_headers: dict[str, str] = Field(
        description="""the dictionary represents the categorized result for headers. Each key is a row header, and the corresponding value is its assigned category (one of the values: "Characteristic", "Exposure", "Outcome" and "Uncategorized")"""
    )


## It seems it's LangChain's bug when trying to convert RowCategorizeResult into a JSON schema. It throws error:
## Error code 400 - Invalid schema for response_format 'RowCategorizeResult': In context=(), 'required' is required to be
##     supplied and to be an array including every key in properties. Extra required key 'categorized_headers' supplied.
## So, here we introduce json schema
RowCategorizeJsonSchema = {
    "title": "RowCategorizeResult",
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
            "description": 'the dictionary represents the categorized result for headers. Each key is a row header name, and the corresponding value is its assigned category string (one of the values: "Characteristic", "Exposure", "Outcome", "P value", "Row headers" and "Uncategorized")',
            "title": "Categorized Rows",
        },
    },
    "required": ["categorized_headers"],
}


def post_process_validate_categorized_result(
    result: RowCategorizeResult | dict,
    md_table: str,
) -> RowCategorizeResult:
    if isinstance(result, dict):
        try:
            res = RowCategorizeResult(**result)
        except ValidationError as e:
            logger.error(e)
            raise e
    # # Ensure column count matches the table
    # expected_columns = markdown_to_dataframe(md_table).shape[1]
    # match_dict = res.categorized_headers
    # if len(match_dict.keys()) != expected_columns:
    #     error_msg = f"Mismatch: Expected {expected_columns} rows, but got {len(match_dict.keys())} in match_dict."
    #     logger.error(error_msg)
    #     raise ValueError(error_msg)

    return res
