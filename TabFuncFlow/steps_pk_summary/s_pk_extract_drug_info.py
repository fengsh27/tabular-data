import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd
import time
import re


def s_pk_extract_drug_info_prompt(md_table, caption):
    return f"""
The following table contains pharmacokinetics (PK) data:  
{display_md_table(md_table)}
Here is the table caption:  
{caption}
Carefully analyze the table and follow these steps:  
(1) Identify how many unique [Drug name, Analyte, Specimen] combinations are present in the table.  
Drug name is the name of the drug mentioned in the study.
Analyte is the substance measured in the study, which can be the primary drug, its metabolite, or another drug it affects, etc. When filling in "Analyte," only enter the name of the substance.
Specimen is the type of sample.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be enclosed in double angle brackets, like this:  
   <<[["Lorazepam", "Lorazepam", "Plasma"], ["Lorazepam", "Lorazepam", "Urine"]]>> (example)  
(3) Verify the source of each [Drug Name, Analyte, Specimen] combination before including it in your answer.  
(4) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred.
"""


def s_pk_extract_drug_info(md_table, caption, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = s_pk_extract_drug_info_prompt(md_table, caption)
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
                    raise ValueError(f"Failed to parse extracted drug information. {e}") from e
            else:
                raise ValueError("No drug information found in the extracted content.")

            if not match_list:
                raise ValueError("Drug information extraction failed: No valid entries found!")

            df_table = pd.DataFrame(match_list, columns=["Drug name", "Analyte", "Specimen"])
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
