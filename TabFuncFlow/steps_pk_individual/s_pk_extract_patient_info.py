import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd
import time
import re


def s_pk_extract_patient_info_prompt(md_table, caption):
    # int_list = extract_integers(md_table+caption)
    # print("==== Automatically Extracted Integers ====")
    # print(int_list)
    return f"""
The following table contains pharmacokinetics (PK) data:
{display_md_table(md_table)}
Here is the table caption:
{caption}
Carefully analyze the table, **row by row and column by column**, and follow these steps:
(1) Identify how many unique [Patient ID, Population, Pregnancy stage] combinations are present in the table.
Patient ID refers to the identifier assigned to each patient.
Population is the patient age group.
Pregnancy stage is the pregnancy stages of patients mentioned in the study.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>.
(3) Ensure that all elements in the list of lists are **strings**, especially Patient ID, which must be enclosed in double quotes (`""`).
(4) Verify the source of each [Patient ID, Population, Pregnancy stage] combination before including it in your answer.
(5) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
"""


def s_pk_extract_patient_info(md_table, caption, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_extract_patient_info_prompt(md_table, caption)
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

            if match_angle:
                try:
                    match_list = ast.literal_eval(fix_trailing_brackets(match_angle[2:-2]))
                    match_list = [list(t) for t in dict.fromkeys(map(tuple, match_list))]
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted population information. {e}") from e
            else:
                raise ValueError(f"No population information found in the extracted content.")

            if not match_list:
                raise ValueError(f"Population information extraction failed: No valid entries found!")

            df_table = pd.DataFrame(match_list, columns=["Patient ID", "Population", "Pregnancy stage"])
            return_md_table = dataframe_to_markdown(df_table)

            return return_md_table, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to extract population information.")
