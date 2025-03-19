
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from TabFuncFlow.operations.f_split_by_cols import f_split_by_cols
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, fix_col_name, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_sum_common_agent import PKSumCommonAgentResult

SPLIT_BY_COLUMNS_PROMPT = ChatPromptTemplate.from_template("""
There is a table related to pharmacokinetics (PK):
{processed_md_table}

This table contains multiple columns, categorized as follows:
{mapping_str}

This table can be split into multiple sub-tables {situation_str}.
Please follow these steps:
  (1) Carefully review all columns and analyze their relationships to determine logical groupings.
  (2) Ensure that each group contains exactly one 'Parameter type' column and at most one 'P value' column.

Return the results as a list of lists, where each inner list represents a sub-table with its included columns, like this:
[["ColumnA", "ColumnB", "ColumnC", "ColumnG"], ["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]
""")

def get_split_by_columns_prompt(md_table: str, col_mapping: dict) -> str:
    """
    get system prompt for splitting by columns

    Args:
    md_table str: aligned table in markdown
    col_mapping dict: mapped columns, like {'Parameter type': 'Parameter type',
        'N': 'Uncategorized', 'Range': 'Parameter value', 
        'Mean ± s.d.': 'Parameter value', 'Median': 'Parameter value'}
    """
    mapping_str = "\n".join(f'"{k}" is categorized as "{v},"' for k, v in col_mapping.items())

    # Count occurrences of specific categories
    parameter_type_count = sum(1 for v in col_mapping.values() if v == "Parameter type")
    parameter_pvalue_count = sum(1 for v in col_mapping.values() if v == "P value")

    # Identify the situation based on category counts
    if parameter_pvalue_count > 1 and parameter_type_count <= 1:
        situation_str = "because there are multiple columns categorized as \"P value\","
    elif parameter_type_count > 1 and parameter_pvalue_count <= 1:
        situation_str = "because there are multiple columns categorized as \"Parameter type\","
    elif parameter_type_count > 1 and parameter_pvalue_count > 1:
        situation_str = "because there are multiple columns categorized as both \"Parameter type\" and \"P value\","
    else:
        situation_str = ""
    
    return SPLIT_BY_COLUMNS_PROMPT.format(
        processed_md_table=display_md_table(md_table),
        mapping_str=mapping_str,
        situation_str=situation_str,
    )

class SplitByColumnsResult(PKSumCommonAgentResult):
    """ 
    The splitted sub-tables for a table
    """
    sub_tables_columns: list[list[str]] = Field(description="""a list of lists, where each inner list represents a sub-table with its included columns, like this:
[["ColumnA", "ColumnB", "ColumnC", "ColumnG"], ["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]""")

def post_process_split_by_columns(
    res: SplitByColumnsResult,
    md_table: str,
) -> list[str]:
    """
    split table to sub-tables according res.sub_tables_columns

    Return:
    a list of sub-tables in markdown string
    """
    col_groups = res.sub_tables_columns
    # Fix column names before using them
    col_groups = [[fix_col_name(item, md_table) for item in group] for group in col_groups]
    # Perform the actual column splitting
    df_table = f_split_by_cols(col_groups, markdown_to_dataframe(md_table))

    # Convert the resulting DataFrames to markdown
    return_md_table_list = [dataframe_to_markdown(d) for d in df_table]

    return return_md_table_list



