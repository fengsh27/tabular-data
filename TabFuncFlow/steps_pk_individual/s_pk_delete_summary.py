from TabFuncFlow.utils.table_utils import *
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_select_row_col import *
import time
import ast
import re


def s_pk_delete_summary_prompt(md_table):
    return f"""
There is now a table related to pharmacokinetics (PK). 
{display_md_table(md_table)}
Carefully examine the table and follow these steps:
(1) Remove any information that pertains to summary statistics, aggregated values, or group-level information such as 'N=' values, as these are not individual-specific.
(2) **Do not remove** any information that pertains to specific individuals, such as individual-level results or personally identifiable data.
If the table already meets this requirement, return [[END]].
If not, please return the following list of lists to assist in creating a new table: [row_list, col_list].  
Replace row_list with the row indices that meet the requirement and col_list with the column names that satisfy the condition.  
When returning this, enclose the list in double angle brackets, like this:
<<[[0, 1, 2, 3], ["Column 1", "Column 2"]]>>
"""


def s_pk_delete_summary(md_table, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_delete_summary_prompt(md_table)
    messages = [msg]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    retries = 0
    wait_time = initial_wait
    total_usage = 0
    all_content = []

    while retries < max_retries:
        try:
            res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
            content = fix_angle_brackets(content)

            total_usage += usage
            all_content.append(f"Attempt {retries + 1}:\n{content}")

            content = content.replace('\n', '')
            matches = re.findall(r'<<.*?>>', content)
            match_angle = matches[-1] if matches else None
            match_end = re.search(r'\[\[END\]\]', content)

            if match_end:
                row_list, col_list = None, None
            elif match_angle:
                extracted_data = fix_trailing_brackets(match_angle[2:-2])
                try:
                    row_list, col_list = ast.literal_eval(extracted_data)
                    if not row_list:
                        row_list = None
                    if not col_list:
                        col_list = None
                except Exception as e:
                    raise ValueError(f"Failed to parse row/column data: {e}") from e

                if not isinstance(row_list, list) or not isinstance(col_list, list):
                    raise ValueError(f"Extracted row/column data is not a list: {extracted_data}")
            else:
                raise ValueError(f"No valid deletion parameters found.")

            if col_list:
                col_list = [fix_col_name(col, md_table) for col in col_list]

            df_table = f_select_row_col(row_list, col_list, markdown_to_dataframe(md_table))
            return_md_table = dataframe_to_markdown(df_table)

            return return_md_table, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to delete summary rows/columns.")
