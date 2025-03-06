import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *


# def s_pk_get_parameter_type_and_unit_prompt(match_dict, md_table, caption):
#     # md_table = re.sub(r'[^\x00-\x7F]+', '', md_table)
#     parameter_type_count = list(match_dict.values()).count("Parameter type")
#     # parameter_unit_count = list(match_dict.values()).count("Parameter unit")
#     if parameter_type_count == 1:
#         key_with_parameter_type = [key for key, value in match_dict.items() if value == "Parameter type"][0]
#         return f"""
# The following table contains pharmacokinetic (PK) data:
# {display_md_table(md_table)}
# Please note that the column "{key_with_parameter_type}" represents the parameter type.
# Carefully analyze the table and follow these steps:
# (1) Split the column "{key_with_parameter_type}" into two separate lists: one for "Parameter type" and another for "Parameter unit".
# (2) Return a tuple containing two lists:
#     - The first list should contain the extracted "Parameter type" values.
#     - The second list should contain the corresponding "Parameter unit" values.
# (3) **Strictly ensure that you process only rows 0 to {markdown_to_dataframe(md_table).shape[0] - 1} from the column "{key_with_parameter_type}".**
#     - The number of processed rows must **exactly match** the number of rows in the original table—no more, no less.
# (4) The returned list should be enclosed within double angle brackets, like this:
#     <<(["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...])>>
# """

def s_pk_get_parameter_type_and_unit_prompt(md_table_aligned, match_dict, md_table, caption):
    # md_table = re.sub(r'[^\x00-\x7F]+', '', md_table)
    parameter_type_count = list(match_dict.values()).count("Parameter type")
    # parameter_unit_count = list(match_dict.values()).count("Parameter unit")
    if parameter_type_count == 1:
        key_with_parameter_type = [key for key, value in match_dict.items() if value == "Parameter type"][0]
        return f"""
The following main table contains pharmacokinetics (PK) data:  
{display_md_table(md_table_aligned)}
Here is the table caption:  
{caption}
From the main table above, I have extracted some columns to create Subtable 1:  
Below is Subtable 1:
{display_md_table(md_table)}
Please note that the column "{key_with_parameter_type}" in Subtable 1 roughly represents the parameter type.
Carefully analyze the table and follow these steps:  
(1) Refer to the "{key_with_parameter_type}" column in Subtable 1 to construct two separate lists: one for a new "Parameter type" and another for "Parameter unit". If the information in Subtable 1 is too coarse or ambiguous, you may need to refer to the main table and its caption to refine and clarify your summarized "Parameter type" and "Parameter unit".
(2) Return a tuple containing two lists:  
    - The first list should contain the extracted "Parameter type" values.  
    - The second list should contain the corresponding "Parameter unit" values.  
(3) **Strictly ensure that you process only rows 0 to {markdown_to_dataframe(md_table).shape[0] - 1} from the column "{key_with_parameter_type}".**  
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) The returned list should be enclosed within double angle brackets, like this:  
    <<(["Parameter type 1", "Parameter type 2", ...], ["Unit 1", "Unit 2", ...])>>  
"""


def s_pk_get_parameter_type_and_unit_parse(content, usage):
    content = content.replace('\n', '')
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_tuple = ast.literal_eval(match_angle[2:-2])  # Extract tuple from `<<(...)>>`
            if not isinstance(match_tuple, tuple) or len(match_tuple) != 2:
                raise ValueError(f"Parsed content is not a valid (type, unit) tuple: {match_tuple}", f"\n{content}", f"\n<<{usage}>>")
            return match_tuple
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse parameter type and unit: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No valid match_angle.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_get_parameter_type_and_unit(md_table_aligned, col_dict, md_table, caption, model_name="gemini_15_pro"):
    parameter_type_count = list(col_dict.values()).count("Parameter type")
    parameter_unit_count = list(col_dict.values()).count("Parameter unit")
    if parameter_type_count == 1 and parameter_unit_count == 1:
        return None, True, "Skipped", 0, False
    elif parameter_type_count == 1 and parameter_unit_count == 0:
        msg = s_pk_get_parameter_type_and_unit_prompt(md_table_aligned, col_dict, md_table, caption)

        messages = [msg, ]
        question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

        res, content, usage, truncated = get_llm_response(messages, question, model=model_name)

        # print(usage, content)
        try:
            match_tuple = s_pk_get_parameter_type_and_unit_parse(content, usage)
        except Exception as e:
            raise RuntimeError(f"Error in s_pk_get_parameter_type_and_unit_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

        # print("**********************")
        # print(match_tuple[0])
        # print(match_tuple[1])
        # print(markdown_to_dataframe(md_table).shape)
        # print(display_md_table(md_table))
        #
        # print("**********************")

        if match_tuple is None:
            raise ValueError("Parameter type and unit extraction failed: No valid tuple found.", f"\n{content}", f"\n<<{usage}>>")
        expected_rows = markdown_to_dataframe(md_table).shape[0]
        if len(match_tuple[0]) != expected_rows or len(match_tuple[1]) != expected_rows:
            raise ValueError(
                f"Mismatch: Expected {expected_rows} rows, but got {len(match_tuple[0])} (types) and {len(match_tuple[1])} (units).", f"\n{content}", f"\n<<{usage}>>"
            )

        return match_tuple, res, content, usage, truncated
    else:
        raise ValueError(f"Invalid column configuration: {parameter_type_count} 'Parameter type' columns and {parameter_unit_count} 'Parameter unit' columns found.")

