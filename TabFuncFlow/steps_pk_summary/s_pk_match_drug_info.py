import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *


def s_pk_match_drug_info_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table):
    first_line = md_table_aligned_with_1_param_type_and_value.strip().split("\n")[0]
    headers = [col.strip() for col in first_line.split("|") if col.strip()]
    extracted_param_types = f""" "{'", "'.join(headers)}" """
    return f"""
The following main table contains pharmacokinetics (PK) data:  
{display_md_table(md_table_aligned)}
Here is the table caption:  
{caption}
From the main table above, I have extracted the following columns to create Subtable 1:  
{extracted_param_types}  
Below is Subtable 1:
{display_md_table(md_table_aligned_with_1_param_type_and_value)}
Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Drug name" - "Analyte" - "Specimen," as follows:
{display_md_table(drug_md_table)}
Carefully analyze the tables and follow these steps:  
(1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.  
(2) **Strictly ensure that you process only rows 0 to {markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0] - 1} from the Subtable 1.**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(3) Format the final list within double angle brackets, like this:  
   <<[1,1,2,2,3,3]>>
"""

# (2) If a row in Subtable 1 does not match any drug entry (usually due to incorrect data), return -1 for that row.


def s_pk_match_drug_info_parse(content, usage):
    content = content.replace('\n', '')
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_list = ast.literal_eval(match_angle[2:-2])
            if not isinstance(match_list, list):
                raise ValueError(f"Parsed content is not a valid list: {match_list}", f"\n{content}", f"\n<<{usage}>>")
            return match_list
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse matched drug info: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No valid matched drug information found in content.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_match_drug_info(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table, model_name="gemini_15_pro"):

    msg = s_pk_match_drug_info_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, drug_md_table)

    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)

    # print(usage, content)

    try:
        match_list = s_pk_match_drug_info_parse(content, usage)  # Parse extracted matches
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_match_drug_info_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    if not match_list:
        raise ValueError("Drug information matching failed: No valid matches found.", f"\n{content}", f"\n<<{usage}>>")  # Ensures the function does not return None

    # Validate row count against `md_table_aligned_with_1_param_type_and_value`
    expected_rows = markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0]
    if len(match_list) != expected_rows:
        raise ValueError(
            f"Mismatch: Expected {expected_rows} rows, but got {len(match_list)} extracted matches.", f"\n{content}", f"\n<<{usage}>>"
        )

    return match_list, res, content, usage, truncated
# md_table_aligned_with_1_param_type_and_value_list = get_1_param_type_and_value_sub_md_table_list(col_mapping, md_table_aligned)
# print(s_pk_match_drug_info(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value_list[1], md_table_drug))

