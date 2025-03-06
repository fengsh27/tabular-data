import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
import pandas as pd


def s_pk_get_parameter_value_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value):
    # Extract the first line (headers) from the provided subtable
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
Please review the information in Subtable 1 row by row and complete Subtable 2.
Subtable 2 should have the following column headers only:  

**Main value, Statistics type, Variation type, Variation value, Interval type, Lower bound, Upper bound, P value** 

Main value: the value of main parameter (not a range). 
Statistics type: the statistics method to summary the Main value, like 'Mean,' 'Median,' 'Count,' etc. **This column is required and must be filled in.**
Variation type: the variability measure (describes how spread out the data is) associated with the Main value, like 'Standard Deviation (SD),' 'CV%,' etc.
Variation value: the value (not a range) that corresponds to the specific variation.
Interval type: the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI,' 'Range,' 'IQR,' etc.
Lower bound: the lower bound value of the interval.
Upper bound: is the upper bound value of the interval.
P value: P-value.

Please Note:
(1) An interval consisting of two numbers must be placed separately into the Low limit and High limit fields; it is prohibited to place it in the Variation value field.
(2) For values that do not need to be filled, enter "N/A".
(3) **Important:** Please return Subtable 2 as a list of lists, excluding the headers. Ensure all values are converted to strings.
(4) **Absolutely no calculations are allowed—every value must be taken directly from Subtable 1 without any modifications.**  
(5) Format the final list within double angle brackets, like this:
<<[["0.162", "Mean", "SD", "0.090", "N/A", "N/A", "N/A", ".67"], ["0.428", "Mean", "SD", "0.162", "N/A", "N/A", "N/A", ".015"]]>>
"""


def s_pk_get_parameter_value_parse(content, usage):
    content = content.replace('\n', '')
    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_list = ast.literal_eval(match_angle[2:-2])  # Extract list from `<<(...)>>`
            if not isinstance(match_list, list):
                raise ValueError(f"Parsed content is not a valid list: {match_list}", f"\n{content}", f"\n<<{usage}>>")
            return match_list
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse parameter values: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No valid parameter values found in content.", f"\n{content}", f"\n<<{usage}>>")  # Clearer error message


def s_pk_get_parameter_value(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, model_name="gemini_15_pro"):
    msg = s_pk_get_parameter_value_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value)

    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."
    # question = "When writing code to solve a problem, do not give the final result immediately. First, explain your thought process in detail, including how you analyze the problem, choose an algorithm or approach, and implement key steps. Then, provide the final code solution."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
    # print(display_md_table(md_table))
    # print(usage, content)

    try:
        match_list = s_pk_get_parameter_value_parse(content, usage)  # Parse extracted values
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_get_parameter_value_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    if not match_list:
        raise ValueError(
            "Parameter value extraction failed: No valid values found.", f"\n{content}", f"\n<<{usage}>>")  # Ensures the function does not return None

    df_table = pd.DataFrame(match_list, columns=[
        'Main value', 'Statistics type', 'Variation type', 'Variation value',
        'Interval type', 'Lower bound', 'Upper bound', 'P value'
    ])

    expected_rows = markdown_to_dataframe(md_table_aligned_with_1_param_type_and_value).shape[0]
    if df_table.shape[0] != expected_rows:
        raise ValueError(
            f"Mismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted values.", f"\n{content}", f"\n<<{usage}>>"
        )

    return_md_table = dataframe_to_markdown(df_table)

    return return_md_table, res, content, usage, truncated

# print(s_pk_get_parameter_value_prompt(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value))
# s_pk_get_parameter_value(md_table_aligned, caption, md_table_aligned_with_1_param_type_and_value, model_name="gemini_15_pro")