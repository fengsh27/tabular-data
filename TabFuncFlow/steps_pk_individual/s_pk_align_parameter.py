from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import time
import ast
import re


def s_pk_align_parameter_prompt(md_table):
    return f"""
There is now a table related to pharmacokinetics (PK). 
{display_md_table(md_table)}
Carefully examine the pharmacokinetics (PK) table and follow these steps to determine how the PK parameter type is represented:
(1) Identify how the PK parameter type (e.g., Cmax, tmax, t1/2, etc.) is structured in the table:
If the PK parameter type serves as the row header or is listed under a specific column, return <<the_col_name>>, replacing the_col_name with the actual column name, and enclose the name in double angle brackets.
If the PK parameter type is represented as column headers, return [[COL]].
(2) Ensure a thorough analysis of the table structure before selecting your answer.
"""


def s_pk_align_parameter(md_table, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_align_parameter_prompt(md_table)
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
            match_col = re.search(r'\[\[COL\]\]', content)

            if match_col:
                col_name = None
            elif matches:
                col_name = matches[-1][2:-2]
            else:
                raise ValueError(f"No valid alignment parameter found.")

            df_table = markdown_to_dataframe(md_table)

            # if col_name:
            #     # col_name = fix_col_name(col_name, md_table)
            #     # df_table = df_table.rename(columns={col_name: "Parameter type"})
            #     return_md_table = dataframe_to_markdown(df_table)
            # else:
            #     df_table = f_transpose(df_table)
            #     # df_table.columns = ["Parameter type"] + list(df_table.columns[1:])
            #     return_md_table = deduplicate_headers(fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(df_table))))
            if col_name:
                df_table = f_transpose(df_table)
                return_md_table = deduplicate_headers(fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(df_table))))
            else:
                return_md_table = dataframe_to_markdown(df_table)

            return return_md_table, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to align parameter column.")

