import pytest
import os
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import anthropic
from datetime import datetime

load_dotenv()

def output_msg(msg: str):
    with open("./benchmark-claude.log", "a+") as fobj:
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

Note:  Evaluate the similarity of the two tables regardless of their row order. And output similarity score in the format [[{score}]].

"""

@pytest.fixture
def client():
    return anthropic.Anthropic(
        api_key=os.environ.get("CLAUDE_API_KEY"),
    )

@pytest.mark.skip("just for test the feasible of claude api")
def test_claude():
    msg = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system="Respond only in Yoda-speak.",
        messages=[
            {"role": "user", "content": "How are you today?"}
        ]
    )
    
    print(msg)
    assert msg == 'test'

@pytest.mark.parametrize("pmid,expected", [("17635501", 81)])
def test_gpt_similarity(client, pmid, expected):
    with open(f"./benchmark/data/{pmid}-pk-summary-gpt4o.csv", "r") as fobj:
        table_gpt4o = fobj.read()
    with open(f"./benchmark/data/{pmid}-pk-summary-baseline.csv", "r") as fobj:
        table_baseline = fobj.read()
    
    user_message = f"""
table 1:
f{table_baseline}

table 2:
f{table_gpt4o}
Please assess the above two tables.
"""
    msg = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system=system_prompts,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    output_msg("\n")
    output_msg(f"pmid: {pmid}, gpt")
    output_msg(msg.content[0].text)
    assert msg.content == "test"
