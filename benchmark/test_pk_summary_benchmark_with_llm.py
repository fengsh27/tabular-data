import pytest
import os
from datetime import datetime
from string import Template

from .utils import generate_columns_definition
from .common import (
    ensure_target_result_directory_existed, 
    prepare_dataset_for_benchmark,
    write_LLM_score,
)
from .constant import (
    BASELINE,
    BenchmarkType,
    LLModelType
)

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
baseline table:
$table_baseline

extracted table:
$table_generated
Please assess the above two tables using baseline table as the standard.
Note:
1. Please use the Parameter type -Value-Unit-Summary Statistics-Variation type as an anchor to assess the similarity.
2. If there are missing rows, you have to reduce the similarity score significantly based on the number of missing rows. e.g., the median and mean are different.
3. Cells with NaN values will be considered as a blank cell.
4. You should specify the difference between the two tables.
5. Synonyms may exist in the tables, e.g., population, drug name, specimen, etc.
6. The data should be mapped to the correct columns. If the cell contents do not comply with the column definition, this indicates the data is placed into the wrong cells. You should reduce the similarity score. For example, based on the column definition, summary statistics should not contain variation information.
7. Ignore the lowercase and uppercase letters
""")

@pytest.mark.skip("just for test the feasible of claude api")
def test_claude(client):
    msg, useage = client.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system="Respond only in Yoda-speak.",
        messages=[
            {"role": "user", "content": "How are you today?"}
        ]
    )
    
    print(msg)
    assert msg is not None

target = os.environ.get("TARGET", "yichuan/0213_prompt_chain")
baseline_dir = os.path.join("./benchmark/data/pk-summary", BASELINE)
target_dir = os.path.join("./benchmark/data/pk-summary", target)
result_dir = os.path.join("./benchmark/result/pk-summary", target)

@pytest.fixture(scope="module")
def prepared_dataset():
    return prepare_dataset_for_benchmark(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )

@pytest.fixture(scope="module")
def ensured_result_path():
    result_dir = ensure_target_result_directory_existed(
        target=target,
        benchmark_type=BenchmarkType.PK_SUMMARY,
    )
    return os.path.join(result_dir, "result.log")

def test_gpt4o_benchmark(client, prepared_dataset, ensured_result_path):
    for id in prepared_dataset:
        the_dict = prepared_dataset[id]
        baseline = the_dict["baseline"]
        if not LLModelType.GPT4O.value in the_dict:
            continue
        gpt4o = the_dict[LLModelType.GPT4O.value]
        with open(baseline, "r") as fobj:
            table_baseline = fobj.read()
        with open(gpt4o, "r") as fobj:
            table_gpt4o = fobj.read()
        user_message = user_message = table_prompt_template.substitute({
            "table_baseline": table_baseline,
            "table_generated": table_gpt4o,
        })
        cols_definition = generate_columns_definition(BenchmarkType.PK_SUMMARY)
        system_prompts = system_prompts_template.substitute({
            "columns_definition": cols_definition
        })
        msg, usage = client.create(system_prompts,user_message)
        write_LLM_score(
            output_fun=ensured_result_path,
            model=LLModelType.GPT4O.value,
            pmid=id,
            score=msg,
            token_usage = usage,
        )

def test_gemini15_benchmark(client, prepared_dataset, ensured_result_path):
    for id in prepared_dataset:
        the_dict = prepared_dataset[id]
        baseline = the_dict["baseline"]
        if not LLModelType.GEMINI15.value in the_dict:
            continue
        gemini15 = the_dict[LLModelType.GEMINI15.value]
        with open(baseline, "r") as fobj:
            table_baseline = fobj.read()
        with open(gemini15, "r") as fobj:
            table_gemini15 = fobj.read()
        user_message = user_message = table_prompt_template.substitute({
            "table_baseline": table_baseline,
            "table_generated": table_gemini15,
        })
        cols_definition = generate_columns_definition(BenchmarkType.PK_SUMMARY)
        system_prompts = system_prompts_template.substitute({
            "columns_definition": cols_definition
        })
        msg, usage = client.create(system_prompts,user_message)
        write_LLM_score(
            output_fn=ensured_result_path,
            model=LLModelType.GEMINI15.value,
            pmid=id,
            score=msg,
            token_usage = usage,
        )
