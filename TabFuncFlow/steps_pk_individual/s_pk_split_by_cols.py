import ast
from TabFuncFlow.utils.table_utils import *
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_split_by_cols import *
import re
import time


def s_pk_split_by_cols_prompt(md_table, col_mapping):
    """
    Generates a structured prompt for splitting a pharmacokinetics (PK) table into sub-tables
    based on column classifications.

    Args:
        md_table (str): Markdown representation of the table.
        col_mapping (dict): Dictionary mapping column names to their respective categories.

    Returns:
        str: A formatted prompt guiding the splitting process.
    """
    mapping_str = "\n".join(f'"{k}" is categorized as "{v},"' for k, v in col_mapping.items())

    # Count occurrences of specific categories
    parameter_count = sum(1 for v in col_mapping.values() if v == "Parameter")
    patient_id_count = sum(1 for v in col_mapping.values() if v == "Patient ID")

    # Identify the situation based on category counts
    if patient_id_count > 1:
        situation_str = "because there are multiple columns categorized as \"Patient ID\","
    else:
        situation_str = ""

    return f"""
There is a table related to pharmacokinetics (PK):
{display_md_table(md_table)}

This table contains multiple columns, categorized as follows:
{mapping_str}

This table can be split into multiple sub-tables {situation_str}.
Please follow these steps:
  (1) Carefully review all columns and analyze their relationships to determine logical groupings.
  (2) Ensure that each group contains exactly one 'Patient ID'.

Return the results as a list of lists, where each inner list represents a sub-table with its included columns.
Enclose the final list within double angle brackets (<< >>) like this:
<<[["ColumnA", "ColumnB", "ColumnC", "ColumnG"], ["ColumnA", "ColumnD", "ColumnE", "ColumnF", "ColumnG"]]>>
"""


def s_pk_split_by_cols(md_table, col_mapping, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_split_by_cols_prompt(md_table, col_mapping)
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

            if not matches:
                raise ValueError(f"No valid column groups found.")

            extracted_data = matches[-1][2:-2]

            try:
                col_groups = ast.literal_eval(fix_trailing_brackets(extracted_data))
            except Exception as e:
                raise ValueError(f"Failed to parse column groups: {e}") from e

            if not isinstance(col_groups, list) or not all(isinstance(group, list) for group in col_groups):
                raise ValueError(
                    f"Parsed content is not a valid list of column groups: {col_groups}"
                )

            if not col_groups:
                raise ValueError(f"Column splitting failed: No valid column groups found.")

            col_groups = [[fix_col_name(item, md_table) for item in group] for group in col_groups]

            df_table = f_split_by_cols(col_groups, markdown_to_dataframe(md_table))

            return_md_table_list = [dataframe_to_markdown(d) for d in df_table]

            return return_md_table_list, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")

            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to split columns.")

