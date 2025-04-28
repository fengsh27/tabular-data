import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import re
import time

# The following main table contains pharmacokinetics (PK) data:
# {display_md_table(md_table_aligned)}
# Here is the table caption:
# {caption}
# From the main table above, I have extracted the following columns to create Subtable 1:
# {extracted_param_types}
# Below is Subtable 1:
# {display_md_table(md_table_aligned_with_1_param_type_and_value)}
# Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Drug name" - "Analyte" - "Specimen," as follows:
# {display_md_table(drug_md_table)}
# Carefully analyze the tables and follow these steps:
# (1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.
# (2) Strictly ensure that you process only rows 0 to {markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0] - 1} from the Subtable 1 (which has {markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0]} rows in total).
#     - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.
# (3) Format the final list within double angle brackets without removing duplicates or sorting, like this:
#     <<[1,1,2,2,3,3]>>
# (4) In rare cases where a row in Subtable 1 cannot be matched, return -1 for that row. This should only be used when absolutely necessary.
# Analyze the following pharmacokinetics (PK) data tables and perform the specified matching operation:


def s_pk_match_drug_info_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table):
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    return f"""
MAIN TABLE (PK Data):
{display_md_table(md_table_aligned)}

Caption: {caption}

SUBTABLE 1 (Extracted from Main Table):
{display_md_table(md_table_aligned_with_1_param_type_and_value)}

SUBTABLE 2 (Drug-Analyte-Specimen Combinations):
{display_md_table(drug_md_table)}

TASK:
1. For each of rows 0-{markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0] - 1} in Subtable 1, find the BEST matching row in Subtable 2 based on:
   - For each row in Subtable 1, find **the best matching one** row in Subtable 2
   - Context from the table caption about the drug (e.g. lorazepam)

2. Processing Rules:
   - Only process rows 0-{markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0] - 1} from Subtable 1 (exactly {markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0]} rows total)
   - Return indices of matching Subtable 2 rows as a Python list of integers
   - If no clear best match is identified for a given row, default to using -1. Important: This default should only be applied when no legitimate match exists after thorough evaluation of all available data.
   - Format the final list within double angle brackets without removing duplicates or sorting, like this:
#     <<[1,1,2,2,3,3]>>

"""
# (3) If a row in Subtable 1 cannot be matched, return -1 for that row.


def s_pk_match_drug_info(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_match_drug_info_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table)
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
                raise ValueError(f"No valid matched drug information found.")

            extracted_data = matches[-1][2:-2]

            try:
                match_list = ast.literal_eval(fix_trailing_brackets(extracted_data))
            except Exception as e:
                raise ValueError(f"Failed to parse matched drug info: {e}") from e

            if not isinstance(match_list, list):
                raise ValueError(
                    f"Parsed content is not a valid list: {match_list}"
                )

            if not match_list:
                raise ValueError(
                    f"Drug information matching failed: No valid matches found."
                )

            expected_rows = markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0]
            if len(match_list) != expected_rows:
                messages.append("Wrong answer example:\n" + content + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches. Think about why this happened, correct your approach, and try again with the right answer.")
                raise ValueError(
                    f"Mismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches."
                )

            return match_list, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to match drug information.")
