import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd
import time
import re


def s_pk_extract_time_and_unit_prompt(md_table, caption, md_data_lines):
    return f"""
The following table contains pharmacokinetics (PK) data:  
{display_md_table(md_table)}
Here is the table caption:  
{caption}
From the main table above, I have extracted some information to create Subtable 1:  
Below is Subtable 1:
{display_md_table(md_data_lines)}  
Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Time value, Time unit].  
- **Time Value:** A specific moment (numerical or time range) when the row of data is recorded, or a drug dose is administered.  
  - Examples: Sampling times, dosing times, or reported observation times.  
  - **The duration of pregnancy (e.g. 20 weeks) must not be recorded as a time value.**
- **Time Unit:** The unit corresponding to the recorded time point (e.g., "Hour", "Min").  
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   <<[["1", "Hour"], ["1", "Hour"], ["1", "Hour"], ["10", "Min"], ["N/A", "N/A"]]>> (example)  
(3) Strictly ensure that you process only rows 0 to {markdown_to_dataframe(md_data_lines).shape[0] - 1} from the Subtable 1 (which has {markdown_to_dataframe(md_data_lines).shape[0]} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
    - For duplicate timestamps and units, merging is strictly prohibited; each occurrence must be recorded as many times as it appears.
(4) Verify the source of each [Time value, Time unit] combination before including it in your answer.  
    **Important: The following parameter types **must not** include time or time units and must be directly entered as ["N/A", "N/A"]:**
        Tmax – Maximum Time (Time to reach the maximum concentration)
        Cmax – Maximum Concentration (You MUST NOT use Tmax value as its time, You MUST use "N/A")
        Cavg – Average Concentration
        AUC Ratio – Area Under the Curve Ratio
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Time value, Time unit] combinations are found, return the default output:**  
    **<<[["N/A", "N/A"]]>>**  

"""


def s_pk_extract_time_and_unit(md_table, caption, md_data_lines, model_name="gemini_15_pro",
                               max_retries=5, initial_wait=1):
    msg = s_pk_extract_time_and_unit_prompt(md_table, caption, md_data_lines)
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
            print(usage, content)

            total_usage += usage
            all_content.append(f"Attempt {retries + 1}:\n{content}")

            content = content.replace('\n', '')
            matches = re.findall(r'<<.*?>>', content)
            match_angle = matches[-1] if matches else None

            if match_angle:
                try:
                    match_list = ast.literal_eval(fix_trailing_brackets(match_angle[2:-2]))
                    # match_list = [list(t) for t in dict.fromkeys(map(tuple, match_list))]
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted time information. {e}") from e
            else:
                raise ValueError("No time information found in the extracted content.")

            if not match_list:
                raise ValueError("Time information extraction failed: No valid entries found!")

            expected_rows = markdown_to_dataframe(md_data_lines).shape[0]

            # if all the same
            if all(x == match_list[0] for x in match_list):
                # expand to expect_rows
                match_list = [match_list[0]] * expected_rows

            df_table = pd.DataFrame(match_list, columns=["Time value", "Time unit"])

            if df_table.shape[0] != expected_rows:
                messages.append("Wrong answer example:\n" + content + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted matches. Think about why this happened, correct your approach, and try again with the right answer.")
                raise ValueError(
                    f"Mismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values."
                )

            return_md_table = dataframe_to_markdown(df_table)

            return return_md_table, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to extract drug information.")
