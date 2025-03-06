import ast
from TabFuncFlow.utils.table_utils import *
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_select_row_col import *


# def s_pk_delete_individual_prompt(md_table):
#     return f"""
# There is now a table related to pharmacokinetics (PK).
# {display_md_table(md_table)}
# Carefully examine the table and follow these steps:
# (1) Remove any information that is specific to an individual.
# If the table already meets this requirement, return [[END]].
# If not, please use the following function to create a new table: f_select_row_col(row_list, col_list)
# Replace row_list with the row indices that satisfy the requirement, and col_list with the column names that satisfy the requirement.
# When returning this, enclose the function call in double angle brackets.
# """
"""The table presents pharmacokinetic data comparing two groups of patients: those without and those with "ARC->GM".  The table provides the median concentration (Cmid) and trough concentration (Ctrough) along with their 95% confidence intervals for each group.  It also provides the number of patients (N) in each group for two different treatments (EI and IB).  Since the prompt asks to remove individual-specific information, we need to remove the 'N=' values associated with each group.  This means rows 0 and 3 should be removed."""


def s_pk_delete_individual_prompt(md_table):
    return f"""
There is now a table related to pharmacokinetics (PK). 
{display_md_table(md_table)}
Carefully examine the table and follow these steps:
(1) Remove any information that pertains to **specific individuals**, such as individual-level results or personally identifiable data.
(2) **Do not remove** summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
If the table already meets this requirement, return [[END]].
If not, please use the following function to create a new table: f_select_row_col(row_list, col_list)
Replace row_list with the row indices that satisfy the requirement, and col_list with the column names that satisfy the requirement. 
When returning this, enclose the function call in double angle brackets.
"""


def s_pk_delete_individual_parse(content, usage):
    content = content.replace('\n', '')
    match_end = re.search(r'\[\[END\]\]', content)
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_end:
        return None, None

    elif match_angle:
        inner_content = match_angle[2:-2]
        match_func = re.match(r'\w+\s*\(\s*(?:\w+\s*=\s*)?(\[[^\]]*\])\s*,\s*(?:\w+\s*=\s*)?(\[[^\]]*\])\s*\)', inner_content)

        if match_func:
            try:
                arg1 = ast.literal_eval(match_func.group(1))
                arg2 = ast.literal_eval(match_func.group(2))
                return arg1, arg2
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse row/col data: {e}", f"\n{content}", f"\n<<{usage}>>") from e
        else:
            raise ValueError(f"Invalid format in extracted content: {inner_content}", f"\n{content}", f"\n<<{usage}>>")

    else:
        raise ValueError("No valid deletion parameters found in content.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_delete_individual(md_table, model_name="gemini_15_pro"):
    msg = s_pk_delete_individual_prompt(md_table)
    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
    # print(display_md_table(md_table))
    # print(usage, content)

    try:
        row_list, col_list = s_pk_delete_individual_parse(content, usage)
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_delete_individual_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    if col_list:
        col_list = [fix_col_name(item, md_table) for item in col_list]

    df_table = f_select_row_col(row_list, col_list, markdown_to_dataframe(md_table))
    return_md_table = dataframe_to_markdown(df_table)
    # print(display_md_table(return_md_table))

    return return_md_table, res, content, usage, truncated
