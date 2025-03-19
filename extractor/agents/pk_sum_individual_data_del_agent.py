
from typing import Union
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI

from TabFuncFlow.operations import f_select_row_col
from TabFuncFlow.utils.table_utils import (
    dataframe_to_markdown, 
    fix_col_name, 
    markdown_to_dataframe,
)
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult

from .agent_utils import display_md_table

from .agent_prompt_utils import INSTRUCTION_PROMPT

INDIVIDUAL_DATA_DEL_PROMPT = ChatPromptTemplate.from_template("""
There is now a table related to pharmacokinetics (PK). 
{processed_md_table}
Carefully examine the table and follow these steps:
(1) Remove any information that pertains to **specific individuals**, such as individual-level results or personally identifiable data.
(2) **Do not remove** summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
Please return the result with the following format:
processed: boolean value, False represents the table have already meets the requirement, don't need to be processed. Otherwise, it will be True
row_list: an array of row indices that satisfy the requirement.
col_list: an array of column indices that satisfy the requirement.
""")

class IndividualDataDelResult(PKSumCommonAgentResult):
    """ Individual data deletion result """
    processed: bool = Field(description="""A boolean flag indicating whether the table requires processing. 
- `False`: The table already meets the requirements and does not need further processing.
- `True`: The table has been processed.""")
    row_list: Union[list[int], None] = Field(description="An array of row indices that satisfy the requirement. If the table does not require processing, this value will be `None`")
    col_list: Union[list[int], None] = Field(description="An array of column indices that satisfy the requirement. If the table does not require processing, this value will be `None`")

def post_process_individual_del_result(
    res: IndividualDataDelResult,
    md_table: str,
):
    if res.processed is False:
        return md_table
    
    row_list = res.row_list
    col_list = res.col_list
    if col_list is not None:
        col_list = [fix_col_name(item, md_table) for item in col_list]
    
    df_table = f_select_row_col(row_list, col_list, markdown_to_dataframe(md_table))
    return dataframe_to_markdown(df_table)
    


