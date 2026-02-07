from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate

from TabFuncFlow.operations.f_select_row_col import f_select_row_col
from TabFuncFlow.utils.table_utils import (
    dataframe_to_markdown,
    fix_col_name,
    markdown_to_dataframe,
)
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult


SUMMARY_DATA_DEL_PROMPT = ChatPromptTemplate.from_template("""
There is now a table related to pharmacokinetics (PK). 

---

### **Task**
Your task is to delete the summary-level results from the PK table.

You can follow the following steps to complete the task:
(1) Remove any information that pertains to summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
    - Delete data entries that do not have an associated Patient ID (e.g., Patient 1, Case 1).
(2) **Do not remove** any information that pertains to specific individuals, such as individual-level results or personally identifiable data.
    - That is, if a row contains information referring to a specific individual, it must be retained — even if it's not a numeric result — because it's part of that individual's record.

---

### **Instructions**
 - When a row's classification is ambiguous, check its **adjacent rows** to infer grouping structure. 

        | Subject    | Parameter Type and Value | Parameter Type and Value |
        |------------|--------------------------|--------------------------|
 row 0: | 1          | 1                        | 1                        |
 row 1: | drug 1     | 40                       | 50                       |
 row 2: | drug 2     | 5                        | 8                        |
 row 3: | 2          | 2                        | 2                        |
 row 4: | drug 1     | 40                       | 50                       |
 row 5: | drug 2     | 5                        | 8                        |
 row 6: | 3          | 3                        | 3                        |
 row 7: | drug 1     | 40                       | 50                       |
 row 8: | drug 2     | 5                        | 8                        |
 ...

 In the above table, the row 1 and 2 are related to the subject 1, the row 3 and 4 are related to the subject 2, the row 5 and 6 are related to the subject 3, ...
 All the rows are related to individual-level results, so we should not delete them.

---

### **Input**
The input is a markdown table.
{processed_md_table}

---

### **Output**
Please return the result with the following format:
processed: boolean value, False represents the table have already meets the requirement, don't need to be processed. Otherwise, it will be True.
row_list: an array of row indices that satisfy the requirement, that is the rows have no summary-level results or personally identifiable data.
col_list: an array of column names that satisfy the requirement, that is the columns in the above rows have no summary-level results or personally identifiable data.

---

### **Example**
If the input is the above table, the output should be:
processed: False
row_list: [0, 1, 2, 3, 4, 5, 6, 7, 8]
col_list: ["Subject", "Parameter Type and Value", "Parameter Type and Value"]
""")


class SummaryDataDelResult(PKIndCommonAgentResult):
    """Summary data deletion result"""

    processed: bool = Field(
        description="""A boolean flag indicating whether the table requires processing. 
- `False`: The table already meets the requirements and does not need further processing.
- `True`: The table has been processed."""
    )
    row_list: list[int] | None = Field(
        description="""An array of row indices that satisfy the requirement, that is the rows no summary-level results or personally identifiable data. 
If the table does not require processing, this value will be `None`"""
    )
    col_list: list[str] | None = Field(
        description="""An array of column names that satisfy the requirement, that is the columns no summary-level results or personally identifiable data. 
If the table does not require processing, this value will be `None`"""
    )


def post_process_summary_del_result(
    res: SummaryDataDelResult,
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
