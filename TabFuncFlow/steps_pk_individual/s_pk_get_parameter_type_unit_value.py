import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import re
import time


def s_pk_get_parameter_type_unit_value_prompt(md_table_aligned, parameter_type_list, caption):
    return f"""
The following main table contains pharmacokinetics (PK) data:  
{display_md_table(md_table_aligned)}
Here is the table caption:  
{caption}
From the main table above, I have extracted some column names which represent parameter types, as a list:  
{parameter_type_list}

Carefully analyze the table and follow these steps:  
(1) Refer to the list {parameter_type_list} to construct two separate lists: one for new "Parameter type" and another for "Parameter unit". If the information in the original list is too coarse or ambiguous, you may need to refer to the main table and its caption to refine and clarify your summarized "Parameter type" and "Parameter unit".
(2) Return a tuple containing two lists:  
    - The first list should contain the extracted "Parameter type" values.  
    - The second list should contain the corresponding "Parameter unit" values.  
(3) Strictly ensure that you process only index 0 to {len(parameter_type_list) - 1} from the original list {parameter_type_list} (which has {len(parameter_type_list)} elements in total). 
    - The number of processed rows must **exactly match** the number of elements in the original list—no more, no less.  
(4) For elements in the list that can not be extracted, enter "N/A" for the entire row.
(5) The returned list should be enclosed within double angle brackets, like this:  
    <<(["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...])>>  
(6) ** Keep the parameter type simple and informative, e.g. "Concentration", "Cmax", "Tmax" etc. **
"""


def s_pk_get_parameter_type_unit_value(md_table_aligned, parameter_type_list, caption, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_get_parameter_type_unit_value_prompt(md_table_aligned, parameter_type_list, caption)
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
                raise ValueError(f"No valid parameter type and unit found.")

            extracted_data = matches[-1][2:-2]

            try:
                match_tuple = ast.literal_eval(fix_trailing_brackets(extracted_data))
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse parameter type and unit: {e}") from e

            if not isinstance(match_tuple, tuple) or len(match_tuple) != 2:
                raise ValueError(
                    f"Parsed content is not a valid (type, unit) tuple: {match_tuple}"
                )

            if match_tuple is None:
                raise ValueError(f"Parameter type and unit extraction failed: No valid tuple found.")

            expected_elements = len(parameter_type_list)
            if len(match_tuple[0]) != expected_elements or len(match_tuple[1]) != expected_elements:
                messages.append("Wrong answer example:\n"+content+f"\nWhy it's wrong:\nMismatch: Expected {expected_elements} parameter types/units, but got {len(match_tuple[0])} (types) and {len(match_tuple[1])} (units). Think about why this happened, correct your approach, and try again with the right answer.")
                raise ValueError(
                    f"Mismatch: Expected {expected_elements} parameter types/units, but got {len(match_tuple[0])} (types) and {len(match_tuple[1])} (units)."
                )

            return match_tuple, res, "\n\n".join(all_content), total_usage, truncated

        except (RuntimeError, ValueError) as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")

            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to extract parameter type and unit.")


