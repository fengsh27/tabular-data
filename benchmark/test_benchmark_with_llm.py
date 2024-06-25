import pytest
import os
import openai
from openai import AzureOpenAI
import google.generativeai as genai
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import anthropic
from datetime import datetime
from string import Template

load_dotenv()

def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")

system_prompts = """
Please act as a biomedial expert to assess the similarities and differences between two biomedical tables. Provide a similarity rating
on a scale of 0 to 100, where 100 indicates identical tables and 0 indicates completely different tables.
The tables contains the following columns:
Drug name: the name of drug mentioned in the paper,
Specimen: what is the specimen, like 'blood', 'breast milk', 'cord blood', and so on.,
Pregnancy Stage: What pregnancy stages of patients mentioned in the paper, like 'postpartum', 'before pregnancy', '1st trimester' and so on. If not mentioned, please label as 'N/A',,
Parameter type: the type of parameter, like 'concentration after the first dose', 'concentration after the second dose', 'clearance', 'Total area under curve' and so on.,
Value: the value of parameter,
Unit: the unit of the value,
Summary Statistics: the statistics method to summary the data, like 'geometric mean', 'arithmetic mean',
Interval type: specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI', 'range' and so on.,
Lower limit: the lower bounds of the interval,
Population: Describe the patient age distribution, including categories such as 'pediatric,' 'adults,' 'old adults,' 'maternal,' 'fetal,' 'neonate,' etc.,
High limit: the higher bounds of the interval,
Subject N:  the number of subjects that correspond to the specific parameter. ,
Variation value: the number that corresponds to the specific variation., 
Variation type: the variability measure (describes how spread out the data is) associated with the specific parameter, e.g., standard deviation (SD), CV%.,
P value: The p-value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported.

Note:  Evaluate the similarity of the two tables regardless of their row order and column order. And output similarity score in the format [[{score}]].

"""

table_prompt_template = Template("""
table 1:
$table_baseline

table 2:
$table_generated
Please assess the above two tables using Table 1 as the standard.
Note:
1. Please use the Parameter type -Value-Unit-Summary Statistics-Variation type as an anchor to assess the similarity.
2.If there are missing rows, you have to reduce the similarity score significantly based on the number of missing rows. e.g., the median and mean are different.
3. Cells with NaN values will be considered as a blank cell.
4. You should specify the difference between the two tables.
5. Synonyms may exist in the tables, e.g., population, drug name, specimen, etc.
6. The data should be mapped to the correct columns. If the cell contents do not comply with the column definition, this indicates the data is placed into the wrong cells. You should reduce the similarity score. For example, based on the column definition, summary statistics should not contain variation information.
7. Ignore the lowercase and uppercase letters
""")

class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
        
    def create(self, system_prompts: str, user_prompts: str):
        res = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompts,
            messages=[
                {"role": "user", "content": user_prompts}
            ]        
        )
        return (res.content[0].text, 0)
    
class GptClient:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
            api_key=os.environ.get("OPENAI_4O_API_KEY", None),
            api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        )
        self.model_4o = os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None)
    def create(self, system_prompts: str, user_prompts: str):
        prompts = [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": user_prompts},
        ]
        res = self.client.chat.completions.create(
            model=self.model_4o,
            temperature=0.0,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            messages=prompts,
        )
        return (res.choices[0].message.content, res.usage.total_tokens)

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GEMINI_15_API_KEY", None))
        self.client =genai.GenerativeModel(os.environ.get("GEMINI_15_MODEL", "gemini-pro"))

    def create(self, system_prompts: str, user_prompts: str):
        msgs = [
            {"role": "user", "parts": [system_prompts, user_prompts]},
        ]
        res = self.client.generate_content(
            msgs,
            generation_config=genai.GenerationConfig(
                candidate_count=1,
                temperature=0,
                max_output_tokens=10000,
            ),
        )
        usage = (
            self.client.count_tokens(res.text).total_tokens + 
            self.client.count_tokens(msgs).total_tokens
            if res is not None and res.text is not None else 0
        )
        return (res.text, usage)

@pytest.fixture
def client():    
    return ClaudeClient() # GptClient, GeminiClient and ClaudeClient are available

@pytest.mark.skip("just for test the feasible of claude api")
def test_claude():
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

# @pytest.mark.skip("temporary skip")
@pytest.mark.parametrize("pmid", [
    "35489632",
    #"35489632",
    #"29943508",
    #"30950674",
    #"34114632",
    #"35465728",
])
def test_gemini_similarity(client, pmid):
    with open(f"./benchmark/data/{pmid}-pk-summary-gemini-flash.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/data/{pmid}-pk-summary-baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    
    user_message = user_message = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gpt4o,
    })
    msg, usage = client.create(system_prompts,user_message)
    output_msg(f"pmid: {pmid}, gemini")
    output_msg(msg)
    assert msg is not None

@pytest.mark.skip("temporary skip")
@pytest.mark.parametrize("pmid", [
    "35489632",
    "30825333",
    "16143486",
    "22050870",
    "17635501",

])
def test_gpt_similarity(client, pmid):
    with open(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/data/{pmid}-pk-summary-baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    
    user_message = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gpt4o,
    })
    msg, usage = client.create(system_prompts, user_message)
    output_msg("\n")
    output_msg(f"pmid: {pmid}, gpt")
    output_msg(msg)
    assert msg is not None

@pytest.mark.skip()
@pytest.mark.parametrize("pmid", [
    "29943508",
    "30950674",
    "34114632",
    "34183327",
    "35465728",
])
def test_5_papers(client, pmid):
    with open(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/data/{pmid}-pk-summary-baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    with open(f"./benchmark/data/{pmid}-pk-summary-gemini15.csv", "r") as fobj:
        table_gemini = fobj.read()
    
    user_msg = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gpt4o,
    })
    msg, _ = client.create(system_prompts, user_msg)
    output_msg("\n")
    output_msg(f"pmid: {pmid}, gpt")
    output_msg(msg)

    user_msg = table_prompt_template.substitute({
        "table_baseline": table_baseline,
        "table_generated": table_gemini,
    })
    msg, _ = client.create(system_prompts, user_msg)
    output_msg("\n")
    output_msg(f"pmid: {pmid}, gemini")
    output_msg(msg)
