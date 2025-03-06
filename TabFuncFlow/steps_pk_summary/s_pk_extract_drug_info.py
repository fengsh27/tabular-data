import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd


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


def s_pk_extract_drug_info_parse(content, usage):
    content = content.replace('\n', '')
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_list = ast.literal_eval(match_angle[2:-2])
            return match_list
        except Exception as e:
            raise ValueError(f"Failed to parse extracted data: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No matching drug info found in content.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_extract_drug_info(md_table, caption, model_name="gemini_15_pro"):
    msg = s_pk_extract_drug_info_prompt(md_table, caption)

    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
    # print(display_md_table(md_table))
    # print(usage, content)

    try:
        match_list = s_pk_extract_drug_info_parse(content, usage)
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_extract_drug_info_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    match_list = list(map(list, set(map(tuple, match_list))))

    if not match_list:
        raise ValueError("Drug info extraction failed: match_list is empty!", f"\n{content}", f"\n<<{usage}>>")

    df_table = pd.DataFrame(match_list, columns=["Drug name", "Analyte", "Specimen"])
    return_md_table = dataframe_to_markdown(df_table)
    # print(display_md_table(return_md_table))
    return return_md_table, res, content, usage, truncated

