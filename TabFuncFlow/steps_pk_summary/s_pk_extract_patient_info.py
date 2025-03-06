import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd


def extract_integers(text):
    """
    Extract only "pure integers":
    - Skip numbers with decimal points (including forms like 3.14, 1.0, .99, etc.)
    - Skip numbers immediately followed (possibly with spaces) by '%' or '％' (e.g., 8%, 8 %)
    - Keep only integers like 42, 100, etc.
    - Remove duplicates and exclude 0
    """
    # 1) (?<![\d.]) ensures that the left side is not a digit or a dot.
    # 2) (\d+) matches a sequence of digits.
    # 3) (?![\d.]| *[%％]) ensures that the right side is not a digit, a dot,
    #    or zero or more spaces followed by '%' or '％'.
    pattern = r'(?<![\d.])(\d+)(?![\d.]| *[%％])'

    # Convert matched digit-strings to integers, use a set to remove duplicates,
    # then remove 0 if present
    return list(
        set(
            int(num_str)
            for num_str in re.findall(pattern, text)
        ) - {0}
    )


def s_pk_extract_patient_info_prompt(md_table, caption):
    int_list = extract_integers(md_table+caption)
    # print("*"*32)
    # print(int_list)
    # print("*"*32)
    return f"""
The following table contains pharmacokinetics (PK) data:  
{display_md_table(md_table)}
Here is the table caption:  
{caption}
Carefully analyze the table, **row by row and column by column**, and follow these steps:  
(1) Identify how many unique [Population, Pregnancy stage, Subject N] combinations are present in the table.  
Population is the patient age group.
Pregnancy stage is the pregnancy stages of patients mentioned in the study.
Subject N is the number of subjects that correspond to the specific parameter.
(2) List each unique combination in the format of a list of lists in one line, using Python string syntax. Your answer should be enclosed in double angle brackets <<>>. 
(3) Verify the source of each [Population, Pregnancy stage, Subject N] combination before including it in your answer.  
(4) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. You must include all the Ns for each age group. 
Specifically, make sure to check every number in this list: {int_list} to determine if it should be listed in Subject N. Fill in "N/A" when you don't know the exact N.
(5) If any information is missing, first try to infer it from the available data (e.g., using context, related entries, or common pharmacokinetic knowledge). Only use "N/A" as a last resort if the information cannot be reasonably inferred. 
"""


def s_pk_extract_patient_info_parse(content, usage):
    content = content.replace('\n', '')
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_list = ast.literal_eval(match_angle[2:-2])
            return match_list
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse extracted patient info: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No matching patient info found in content.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_extract_patient_info(md_table, caption, model_name="gemini_15_pro"):
    msg = s_pk_extract_patient_info_prompt(md_table, caption)
    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
    # print(display_md_table(md_table))
    # print(usage, content)

    try:
        match_list = s_pk_extract_patient_info_parse(content, usage)
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_extract_patient_info_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    match_list = list(map(list, set(map(tuple, match_list))))

    if not match_list:
        raise ValueError("Patient info extraction failed: match_list is empty!", f"\n{content}", f"\n<<{usage}>>")

    df_table = pd.DataFrame(match_list, columns=["Population", "Pregnancy stage", "Subject N"])
    return_md_table = dataframe_to_markdown(df_table)

    return return_md_table, res, content, usage, truncated
