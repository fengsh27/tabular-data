  
import os  
from openai import AzureOpenAI
import pytest
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from extractor.request_geminiai import get_gemini
from extractor.agents.pk_summary.pk_sum_patient_matching_agent import MatchedPatientResult

load_dotenv()

system_prompt = """
The following main table contains pharmacokinetics (PK) data:  
col: | "Parameter type" | "Overall" | "N_0" | "Range_0" | "Mean ± s.d._0" | "Median" | "3 Month to < 3 Years" | "N_1" | "Range_1" | "Mean ± s.d._1" | "3 to < 13 Years" | "N_2" | "Range_2" | "Mean ± s.d._2" | "13 to < 18 Years" | "N_3" | "Range_3" | "Mean ± s.d._3" |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
row 0: | Free Fraction | nan | 61 | 0.07–0.48 | 0.10 ± 0.05 | 0.09 | nan | 17 | 0.07–0.48 | 0.11 ± 0.10 | nan | 28 | 0.07–0.17 | 0.10 ± 0.02 | nan | 16 | 0.07–0.15 | 0.09 ± 0.02 |
row 1: | CL (mL/min/kg) | nan | 63 | 0.3–7.75 | 1.2 ± 0.93 | 1.08 | nan | 18 | 0.63–7.75 | 1.57 ± 1.62 | nan | 29 | 0.30–1.82 | 1.12 ± 0.40 | nan | 16 | 0.43–1.58 | 0.95 ± 0.32 |
row 2: | CL mL/min/m2) | nan | 63 | 6.50–147.17 | 33.33 ± 19.33 | 29.00 | nan | 18 | 12.83–147.17 | 32.83 ± 30.17 | nan | 29 | 6.50–69.17 | 31.83 ± 13.83 | nan | 16 | 16.33–60.00 | 36.67 ± 12.00 |
row 3: | Vdss (L/kg) | nan | 63 | 0.49–3.40 | 1.48 ± 0.54 | 1.37 | nan | 18 | 0.67–3.40 | 1.62 ± 0.59 | nan | 29 | 0.49–3.00 | 1.50 ± 0.61 | nan | 16 | 1.00–1.54 | 1.27 ± 0.17 |
row 4: | Beta (hr−1) | nan | 63 | 0.017–0.118 | 0.048 ± 0.020 | 0.046 | nan | 18 | 0.024–0.118 | 0.053 ± 0.027 | nan | 29 | 0.017–0.092 | 0.048 ± 0.017 | nan | 16 | 0.017–0.084 | 0.044 ± 0.016 |
row 5: | T½ Beta (hr) | nan | 63 | 5.9–42.0 | 16.8 ± 7.1 | 15.1 | nan | 18 | 5.9–28.4 | 15.8 ± 6.5 | nan | 29 | 7.5–40.6 | 16.9 ± 7.4 | nan | 16 | 8.2–42.0 | 17.8 ± 7.7 |
Here is the table caption:  

Bayesian pharmacokinetics parameters (all subjects). CL is clearance. Vdss is volume of distribution at steady state. Beta is the terminal slope of the log concentration versus time profile. T½ Beta is the elimination half-life.

From the main table above, I have extracted the following columns to create Subtable 1:  
 "Parameter type", "Range_0"   
 Below is Subtable 1:
 col: | "Parameter type" | "Range_0" |
 | --- | --- |
 row 0: | Free Fraction | 0.07–0.48 |
 row 1: | CL (mL/min/kg) | 0.3–7.75 |
 row 2: | CL mL/min/m2) | 6.50–147.17 |
 row 3: | Vdss (L/kg) | 0.49–3.40 |
 row 4: | Beta (hr−1) | 0.017–0.118 |
 row 5: | T½ Beta (hr) | 5.9–42.0 |
 Additionally, I have compiled Subtable 2, where each row represents a unique combination of "Population" - "Pregnancy stage" - "Subject N," as follows:
 col: | "Population" | "Pregnancy stage" | "Subject N" |
 | --- | --- | --- |
 row 0: | Overall | N/A | 61 |
 row 1: | Overall | N/A | 63 |
 row 2: | 3 Month to < 3 Years | N/A | 17 |
 row 3: | 3 Month to < 3 Years | N/A | 18 |
 row 4: | 3 to < 13 Years | N/A | 28 |
 row 5: | 3 to < 13 Years | N/A | 29 |
 row 6: | 13 to < 18 Years | N/A | 16 |
 Carefully analyze the tables and follow these steps:  
 (1) For each row in Subtable 1, find **the best matching one** row in Subtable 2. Return a list of unique row indices (as integers) from Subtable 2 that correspond to each row in Subtable 1.  
 (2) **Strictly ensure that you process only rows 0 to 5 from the Subtable 1.**  
 - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
 (3) The "Subject N" values within each population group sometimes differ slightly across parameters. This reflects data availability for each specific parameter within that age group. 
 - For instance, if the total N is 10 but a specific data point corresponds to 9, the correct Subject N for that row should be 9. It is essential to ensure that each row is matched with the appropriate Subject N accordingly.
 (4) The final list should be like this without removing duplicates or sorting:
 [1,1,2,2,3,3]
 (5) In rare cases where a row in Subtable 1 cannot be matched, return -1 for that row. This should only be used when absolutely necessary.
"""
    
instruction_prompt = """
Do not give the final result immediately. First, explain your reasoning process step by step, then provide the answer.
"""

@pytest.mark.skip()
def test_AzureOpenAI(): 
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://pharmacoinfo-openai-1.openai.azure.com/")  
    deployment = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4o")  
    api_version = os.getenv("OPENAI_API_VERSION", "2024-10-21")
    key = os.getenv("OPENAI_API_KEY", None)
      
    client = AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,  
    )  
      
    chat_prompt = [
        {
            "role": "user",
            "content": "Hi",
        }
    ]
    
    completion = client.chat.completions.create(  
        model=deployment,  
        messages=chat_prompt,
        max_tokens=4096,  
        temperature=0.1,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False  
    )  
      
    print(completion.to_json())  

@pytest.mark.skip()
def test_gemini():
    llm = get_gemini()

@pytest.mark.skip()
def test_AzureOpenAI_with_22050870_table_3():
    
    endpoint = os.getenv("AZURE_OPENAI_4O_ENDPOINT", "https://pharmacoinfo-openai-1.openai.azure.com/")  
    deployment = os.getenv("OPENAI_4O_DEPLOYMENT_NAME", "gpt-4o")  
    api_version = os.getenv("OPENAI_4O_API_VERSION", "2024-10-21")
    key = os.getenv("OPENAI_4O_API_KEY", None)
      
    client = AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,  
    )
    messages = [{
        "role": "system",
        "content": system_prompt,
    }, {
        "role": "user",
        "content": instruction_prompt,
    }]
    completion = client.chat.completions.create(
        messages=messages,
        model=deployment,
        max_tokens=4096,  
        temperature=0.1,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False,
    )
    print(completion.to_json())

# @pytest.mark.skip()
def test_AzureChatOpenAI_with_22050870_table_3():
    client = AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
    )
    messages = [("system", system_prompt), ("user",instruction_prompt)]
    response = client.invoke(messages)
    print(response)

@pytest.mark.skip()
def test_AzureChatOpenAI_structureoutput_with_22050870_table_3():
    client = AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", system_prompt),
            ("user", instruction_prompt)
        ]
    )

    agent = prompt | client.with_structured_output(MatchedPatientResult)
    res = agent.invoke(input={})
    print(res)


