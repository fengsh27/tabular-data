import json

def generate_columns_definition(pk_or_pe: str) -> str:
    assert pk_or_pe.lower() == 'pk' or pk_or_pe.lower() == 'pe'
    k = pk_or_pe.lower()
    fn = f"./prompts/{k}_prompts.json"
    with open(fn, "r") as fobj:
        json_str = fobj.read()
        json_obj = json.loads(json_str)
    table_extraction = json_obj["table_extraction_prompts"]
    col_defs = table_extraction["output_column_definitions"]
    col_dict = table_extraction["output_columns_map"]
    assert len(col_defs) == len(col_dict)

    result_col_defs = []
    for ix in range(len(col_defs)):
        col_pair = col_dict[ix]
        col_def = col_defs[ix]
        k_len = len(col_pair[0]) + 2 # "{col_pair[0]}: "
        definition = f"{col_def[k_len:]}"
        result_col_defs.append(definition)

    return ".\n ".join(result_col_defs)

        

