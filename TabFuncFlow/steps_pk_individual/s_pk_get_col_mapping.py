import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
from difflib import get_close_matches
import re
import time


def s_pk_get_col_mapping_prompt(md_table):
    df = markdown_to_dataframe(md_table)  # Assuming this is your DataFrame
    column_headers_str = 'These are all its column headers: ' + ", ".join(f'"{col}"' for col in df.columns)
    return f"""
The following table contains pharmacokinetics (PK) data:  
{display_md_table(md_table)}
{column_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all column headers and categorize each one into one of the following groups:  
   - **"Patient ID"**: Columns that describe the identifier assigned to each patient.
   - **"Parameter"**: (Must be Dependent Variable) Columns that describe pharmacokinetics concentration or ratio parameter data.  
        - Examples: Drug concentration (e.g., "Plasma Conc ng/mL"), Area under the curve (e.g., "AUC 0-∞"), Maximum concentration (e.g., "Cmax"), Time to Maximum Concentration (e.g., "Tmax"), Half-life (e.g., "T1/2"), Clearance rate (e.g., "CL/F").
   - **"Uncategorized"**: Columns that do not fit into the above categories, such as those representing time-related information, or dose amount information.
        - Examples: Dose amount (e.g. "1 mg/kg", "infant dose", "maternal dose"), Sampling time (e.g., "Time postdose_hr"), Dosing interval (e.g., "Tau hr"), Collection date (e.g., "Sample Date"), Study period (e.g., "Period").
(3) if a column is only about the subject number, it is considered as "Uncategorized"
(2) Return a dictionary where each key is a column header, and the corresponding value is its assigned category. Your dictionary should be enclosed in double angle brackets <<>>.  
"""


def s_pk_get_col_mapping(md_table, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_get_col_mapping_prompt(md_table)
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
                raise ValueError(f"No valid column mapping found.")

            extracted_data = matches[-1][2:-2]

            try:
                match_dict = ast.literal_eval(fix_trailing_brackets(extracted_data))
            except Exception as e:
                raise ValueError(f"Failed to parse column mapping: {e}") from e

            if not isinstance(match_dict, dict):
                raise ValueError(f"Parsed content is not a dictionary")

            predefined_categories = ["Patient ID", "Parameter", "Uncategorized"]

            match_dict = {
                fix_col_name(k, md_table): (
                    get_close_matches(v, predefined_categories, n=1)[0] if get_close_matches(v, predefined_categories, n=1) else "Uncategorized"
                ) for k, v in match_dict.items()
            }

            if not match_dict:
                raise ValueError(f"Column mapping extraction failed: No mappings found.")

            expected_columns = markdown_to_dataframe(md_table).shape[1]
            if len(match_dict.keys()) != expected_columns:
                raise ValueError(
                    f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict."
                )

            # parameter_type_count = list(match_dict.values()).count("Parameter type")
            # if parameter_type_count != 1:
            #     raise ValueError(
            #         f"Invalid mapping: Expected 1 'Parameter type' column, but found {parameter_type_count}."
            #     )

            return match_dict, res, "\n\n".join(all_content), total_usage, truncated

        except (RuntimeError, ValueError) as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")

            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to extract column mapping.")

