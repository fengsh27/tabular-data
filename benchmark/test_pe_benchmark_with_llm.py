import pytest

from datetime import datetime
from string import Template

from .utils import generate_columns_definition
from .common import ResponderWithRetries

def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")

system_prompts_template = Template("""
Please act as a biomedial expert to assess the similarities and differences between two biomedical tables, one is baseline table, the other is extracted table. 
Provide a similarity rating on a scale of 0 to 100, where 100 indicates identical tables and 0 indicates completely different tables.
The tables contains the following columns:
$columns_definition

Note:  Evaluate the similarity of the two tables regardless of their row order and column order. And output similarity score in the format [[{score}]].

""")

table_prompt_template = Template("""
baseline table :
$table_baseline

extracted table:
$table_generated
Please assess the above two tables using baseline table as the standard.
Note:
1. Please use the Parameter type -Value-Unit-P Value-Outcomes-Exposure as an anchor to assess the similarity.
2.If there are missing rows, you have to reduce the similarity score significantly based on the number of missing rows. e.g., the median and mean are different.
3. Cells with NaN values will be considered as a blank cell.
4. You should specify the difference between the two tables.
5. Synonyms may exist in the tables, e.g., population, drug name, specimen, etc.
6. The data should be mapped to the correct columns. If the cell contents do not comply with the column definition, this indicates the data is placed into the wrong cells. You should reduce the similarity score. For example, based on the column definition, summary statistics should not contain variation information.
7. Ignore the lowercase and uppercase letters
""")



@pytest.mark.skip("just for test the feasible of claude api")
def test_claude(client):
    msg, useage = client.create(
        system_prompts="Respond only in Yoda-speak.",
        user_prompts="How are you today?",
    )
    
    print(msg)
    assert msg is not None

@pytest.mark.skip("temporary skip")
@pytest.mark.parametrize("pmid", [
    "15930210",
    "18782787",
    "30308427",
    "33864754",
    "34024233",
    "34083820",
    "34741059",
    "35296792",
    "35997979",
    "36396314",
])
def test_gemini_similarity(client, pmid):
    with open(f"./benchmark/pe/{pmid}_gemini15.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/pe/{pmid}_baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    
    user_message = user_message = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gpt4o,
    })
    cols_definition = generate_columns_definition("pe")
    system_prompts = system_prompts_template.substitute({
        "columns_definition": cols_definition
    })
    msg, usage = client.create(system_prompts,user_message)
    output_msg(f"pmid: {pmid}, gemini")
    output_msg(msg)
    assert msg is not None

@pytest.mark.parametrize("pmid", [
    "15930210",
    "18782787",
    "30308427",
    "33864754", #
    "34024233",
    "34083820",
    "34741059",
    "35296792",
    "35997979",
    "36396314",
])
def test_gpt_similarity(client, pmid):
    with open(f"./benchmark/pe/{pmid}_gpt4o.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/pe/{pmid}_baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    
    user_message = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gpt4o,
    })
    cols_definition = generate_columns_definition("pe")
    system_prompts = system_prompts_template.substitute({
        "columns_definition": cols_definition
    })
    
    responder = ResponderWithRetries(lambda args: client.create(args[0], args[1]))
    msg, usage = responder.respond([system_prompts, user_message])
    output_msg("\n")
    output_msg(f"pmid: {pmid}, gpt")
    output_msg(msg)
    assert msg is not None
