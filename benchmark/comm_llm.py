
import re
from typing import Union
from string import Template
import logging

from benchmark.constant import BenchmarkType, LLModelType
from benchmark.utils import generate_columns_definition
from benchmark.common import LLMClient

logger = logging.getLogger(__name__)

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

def write_LLM_score(
    output_fn: str,
    model: str,
    pmid: str,
    score: str,
    token_usage: str,
):
    SCORE_LENGTH_THRESHOLD=64
    with open(output_fn, "a+") as fobj:
        if len(score) > SCORE_LENGTH_THRESHOLD:
            # output in verbose mode
            fobj.write("\n" + "=" * 81 + "\n")
            fobj.write(f"pmid: {pmid}, model: {model}\n")
            fobj.write(score)
            fobj.write("\n")
            fobj.write(f"token usage: {token_usage}\n")
        else:
            # output in concise mode
            fobj.write(f"{model}, {pmid}, {score}, {token_usage}")


def run_llm_benchmark(
    dataset: dict,
    benchmark_type: Union[BenchmarkType.PE, BenchmarkType.PK_SUMMARY],
    model: Union[LLModelType.GEMINI15, LLModelType.GPT4O],
    result_file: str,
    client: LLMClient,
):
    scores = []
    pat = re.compile(r"\[\[\d+(.+\d+)?\]\]")
    for id in dataset:
        the_dict = dataset[id]
        baseline = the_dict["baseline"]
        if not model.value in the_dict:
            continue
        gpt4o = the_dict[model.value]
        with open(baseline, "r") as fobj:
            table_baseline = fobj.read()
        with open(gpt4o, "r") as fobj:
            table_gpt4o = fobj.read()
        user_message = user_message = table_prompt_template.substitute({
            "table_baseline": table_baseline,
            "table_generated": table_gpt4o,
        })
        cols_definition = generate_columns_definition(benchmark_type)
        system_prompts = system_prompts_template.substitute({
            "columns_definition": cols_definition
        })
        msg, usage = client.create(system_prompts,user_message)
        write_LLM_score(
            output_fn=result_file,
            model=model.value,
            pmid=id,
            score=msg,
            token_usage = usage,
        )
        res = pat.search(msg)
        if res is None:
            logger.error(f"Can't find [[score]] for pmid {id}")
            continue
        scores.append({
            "pmid": id,
            "score": res[0],
            "token_usage": usage,
        })
    # write in concise mode, only output scores
    for score in scores:
        write_LLM_score(
            output_fn=result_file,
            model=model.value,
            pmid=score["pmid"],
            score=score["score"],
            token_usage=score["token_usage"],
        )
