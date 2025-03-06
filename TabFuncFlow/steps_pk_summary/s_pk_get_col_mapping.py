import ast
from TabFuncFlow.utils.llm_utils import *
from TabFuncFlow.operations.f_transpose import *
from difflib import get_close_matches


def s_pk_get_col_mapping_prompt(md_table):
    df = markdown_to_dataframe(md_table)  # Assuming this is your DataFrame
    column_headers_str = 'These are all its column headers: ' + ", ".join(f'"{col}"' for col in df.columns)
    return f"""
The following table contains pharmacokinetics (PK) data:  
{display_md_table(md_table)}
{column_headers_str}
Carefully analyze the table and follow these steps:  
(1) Examine all column headers and categorize each one into one of the following groups:  
   - **"Parameter type"**: Columns that describe the type of pharmacokinetic parameter.  
   - **"Parameter unit"**: Columns that specify the unit of the parameter type.  
   - **"Parameter value"**: Columns that contain numerical parameter values.  
   - **"P value"**: Columns that represent statistical P values.  
   - **"Uncategorized"**: Columns that do not fit into any of the above categories.  
(3) if a column is only about the subject number, it is considered as "Uncategorized"
(2) Return a dictionary where each key is a column header, and the corresponding value is its assigned category. Your dictionary should be enclosed in double angle brackets <<>>.  
"""


def s_pk_get_col_mapping_parse(content, usage):
    content = content.replace('\n', '')

    matches = re.findall(r'<<.*?>>', content)
    match_angle = matches[-1] if matches else None

    if match_angle:
        try:
            match_dict = ast.literal_eval(match_angle[2:-2])
            if not isinstance(match_dict, dict):
                raise ValueError(f"Parsed content is not a dictionary: {match_dict}", f"\n{content}", f"\n<<{usage}>>")
            return match_dict
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse column mapping: {e}", f"\n{content}", f"\n<<{usage}>>") from e
    else:
        raise ValueError("No valid column mapping found in content.", f"\n{content}", f"\n<<{usage}>>")


def s_pk_get_col_mapping(md_table, model_name="gemini_15_pro"):
    msg = s_pk_get_col_mapping_prompt(md_table)
    messages = [msg, ]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
    # print(display_md_table(md_table))
    # print(usage, content)

    try:
        match_dict = s_pk_get_col_mapping_parse(content, usage)  # Parse the extracted mapping
    except Exception as e:
        raise RuntimeError(f"Error in s_pk_get_col_mapping_parse: {e}", f"\n{content}", f"\n<<{usage}>>") from e

    # Fix column names and match them to predefined categories
    predefined_categories = ["Parameter value", "P value", "Parameter type", "Parameter unit", "Uncategorized"]
    match_dict = {
        fix_col_name(k, md_table): get_close_matches(v, predefined_categories, n=1)[0] if get_close_matches(v, predefined_categories, n=1) else "Uncategorized"
        for k, v in match_dict.items()
    }

    if not match_dict:
        raise ValueError("Column mapping extraction failed: No mappings found.", f"\n{content}", f"\n<<{usage}>>")  # Ensures the function does not return None

    # Ensure column count matches the table
    expected_columns = markdown_to_dataframe(md_table).shape[1]
    if len(match_dict.keys()) != expected_columns:
        raise ValueError(f"Mismatch: Expected {expected_columns} columns, but got {len(match_dict.keys())} in match_dict.", f"\n{content}", f"\n<<{usage}>>")

    # Ensure exactly one "Parameter type" column exists
    parameter_type_count = list(match_dict.values()).count("Parameter type")
    if parameter_type_count != 1:
        raise ValueError(f"Invalid mapping: Expected 1 'Parameter type' column, but found {parameter_type_count}.", f"\n{content}", f"\n<<{usage}>>")

    return match_dict, res, content, usage, truncated
