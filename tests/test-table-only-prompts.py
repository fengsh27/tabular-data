
from datetime import datetime
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os

def write_log_info(msg: str):
    with open("./logs/test.log", "a+") as fobj:
        now = datetime.now()
        text = f"{now.strftime('%Y-%m-%d %H:%M:%S,%f')} - {msg}\n"
        fobj.write(text)


load_dotenv()
openai_type = os.environ.get("OPENAI_API_TYPE")
if openai_type == "azure":
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_key=os.environ.get("OPENAI_API_KEY", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
    )
    model = os.environ.get("OPENAI_DEPLOYMENT_NAME", None)
else:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))
    model = "gpt-3.5-turbo" # "gpt-4-1106-preview"

def test_table_only_prompts():
    hint_prompts = {
        "role": "user",
        "content": """Please act as a biomedical assistant, extract the following information from the provided biomedical tables and output as a table in markdown format:
1. Drug name, the name of drug mentioned in the paper
2. Specimen, what is the specimen, like "blood", "breast milk", "cord blood", and so on.
3. Pregnancy Stage, pregnancy stage, What pregnancy stages of patients mentioned in the paper, like "postpartum", "before pregnancy", "1st trimester" and so on. If not mentioned, please label as "N/A",
4. Parameter type, the type of parameter, like "concentration after the first dose", "concentration after the second dose", "clearance", "Total area under curve" and so on.
5. Value, the value of parameter
6. unit,  the unit of the value
7. Summary Statistics, the statistics method to summary the data, like "geometric mean", "arithmetic mean"
8. Interval type, specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like "95% CI", "range" and so on.
9. lower limit, the lower bounds of the interval
10.  high limit. the higher bounds of the interval
11. Population: Describe the patient age distribution, including categories such as "pediatric," "adults," "old adults," "maternal," "fetal," "neonate," etc.
Please note: 
1. only output markdown table without any other characters and embed the text in code chunks, so it won't convert to HTML in the assistant.
2. give me all extracted data as compoleted as possible, I don't want you to omit any data.
3. if the information that is not provided, please leave it empty 
"""
    }
    table_prompts = {
        "role": "user",
        "content": """
Here are the tables (including their caption and footnote):
table caption: Geometric mean (95% CI), GMR, and arithmetic mean of the postpartum (6.3 ± 0.8 weeks) and antepartum (31.1 ± 2.4 weeks) indinavir pharmacokinetic parameters in HIV-1-infected women
,Time point or ratio,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a,Parameter (geometric and arithmetic mean values)a
,Time point or ratio,AUC(0-8) (μg · min/ml),AUC(0-8) (μg · min/ml),CL/F (ml/min),CL/F (ml/min),CL/F (ml/min/kg),CL/F (ml/min/kg),Cmax (μg/ml),Cmax (μg/ml),Tmax (min),Tmax (min),Cmin (ng/ml),Cmin (ng/ml)
,Time point or ratio,Geometric (95% CI),Arithmetic (±SD),Geometric (95% CI),Arithmetic (±SD),Geometric (95% CI),Arithmetic (±SD),Geometric (95% CI),Arithmetic (±SD),Geometric (95% CI),Arithmetic (±SD),Geometric (95% CI),Arithmetic (±SD)
0,Antepartum,"341 (187, 621)",459 (322),"2,354 (1,290, 4,297)","3,558 (3,796)","29.6 (17.0, 51.7)",41.5 (40.1),"3.07 (1.53, 6.16)",4.37 (2.77),"95 (65, 138)",111 (72),"25.5 (4.4, 149.1)",128 (127)
1,Postpartum,"1,285 (927, 1,781)","1,429 (707)","624 (450, 864)",697 (369),"8.6 (6.7, 11.2)",8.8 (4.0),"8.57 (6.49, 11.33)",9.27 (3.81),"97 (69, 136)",113 (87),"96.7 (13.8, 675.5)","806 (1,548)"
2,Postpartum: antepartumb,"3.77 (1.95, 7.28)",,"0.26 (0.14, 0.50)",,"0.29 (0.16, 0.54)",,"2.79 (1.29, 6.04)",,"1.02 (0.60, 1.75)",,"3.80 (1.03, 13.99)",

table footnote: aCLF, oral clearance, where F is bioavailability; Cmax, maximum concentration of drug in plasma; Tmax, time to reach maximum plasma concentration; Cmin, minimum concentration in plasma at the end of the dosing interval.bGeometric mean ratio.

"""
    }
    prompts = [hint_prompts, table_prompts]
    prompts.append({"role": "user", "content": "Now please extract information from the tables"})
    
    try:
        res = client.chat.completions.create(
            model=model, 
            messages=prompts,
            temperature=0.7,
            # max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        assert res is not None
        (res.choices[0].message.content)
        write_log_info(model)
        write_log_info(res.choices[0].message.content)
        # print(res.choices[0].)
        # (True, res.choices[0], res.usage)
    except Exception as e:
        # (False, str(e))
        write_log_info(str(e))
        assert False

