
import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv

from TabFuncFlow.utils.table_utils import dataframe_to_markdown

load_dotenv()

def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_NAME"),
    )
def get_azure_openai():
    
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        temperature=0.0,
        max_tokens=os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096),
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
def get_deepseek():
    return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.0,
        max_completion_tokens=10000,
        max_retries=3,
    )

@pytest.fixture(scope="module")
def llm():    
    return get_openai()

@pytest.fixture(scope="module")
def md_table():
    html_content = """
<section id="S8">
   <h3 class="pmc_sec_title">Pharmacokinetics</h3>
   <p id="P25"><a class="usa-link" href="#T2">Table II</a> summarizes the noncompartmental PK evaluation for the Elective Cohort. Eight pharmacokinetic samples (0.3%) were excluded, seven due to suspected contamination from the infused drug and one due to collection during the lorazepam infusion. Overall, the mean area-under-the-curve (AUC<sub>0-∞</sub>) was 822.5 ng*hr/mL and the median AUC<sub>0-∞</sub> was 601.5 ng*hr/mL with an average dose of 0.04 mg/kg. The overall fit of the population PK model was good over the wide range of individuals in the population. There were no covariates meeting criteria for inclusion into the model. Thirty three subjects (23 Status Cohort, 10 Elective Cohort) had received, at baseline, at least one agent that can induce drug metabolizing enzymes. The calculated value for terminal (beta) half-life was 16 hours for a typical 24 kg child. The empiric Bayesian estimated parameters from the post-hoc analysis are summarized in <a class="usa-link" href="#T3">Table III</a>. The model demonstrated good model prediction of observed concentrations even when applied to the patients exhibiting the highest and lowest individual clearances, and in a patient who received a total of 9 doses of lorazepam during the PK sampling interval.</p>
   <section class="tw xbox font-sm" id="T2">
      <h4 class="obj_head">Table 2.</h4>
      <div class="caption p">
         <p id="P40">Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0-∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life.</p>
      </div>
      <div class="tbl-box p" tabindex="0">
         <table class="content" frame="hsides" rules="groups">
            <thead>
               <tr>
                  <th align="left" colspan="1" rowspan="1"></th>
                  <th align="center" colspan="1" rowspan="1">C<sub>max</sub><br/>(ng/mL)</th>
                  <th align="center" colspan="1" rowspan="1" valign="top">AUC<sub>0-∞</sub> </th>
                  <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/kg)</th>
                  <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/m<sup>2</sup>)</th>
                  <th align="center" colspan="1" rowspan="1">Vdz<br/>(L/kg)</th>
                  <th align="center" colspan="1" rowspan="1">T<sub>1/2</sub><br/>(hr)</th>
               </tr>
            </thead>
            <tbody>
               <tr>
                  <td align="left" colspan="1" rowspan="1">N</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1" valign="top">Range</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">29.3–209.6</td>
                  <td align="center" colspan="1" rowspan="1">253.3–3202.5</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">3.33–131.50</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">5.5–67.5</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">0.33–4.05</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">9.5–47.0</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">56.1 ± 44.9</td>
                  <td align="center" colspan="1" rowspan="1">822.5 ± 706.1</td>
                  <td align="center" colspan="1" rowspan="1">49.33 ± 30.83</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">31.95 ± 13.99</td>
                  <td align="center" colspan="1" rowspan="1">1.92 ± 0.84</td>
                  <td align="center" colspan="1" rowspan="1">20.5 ± 10.2</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1">Median</td>
                  <td align="center" colspan="1" rowspan="1">42.2</td>
                  <td align="center" colspan="1" rowspan="1">601.5</td>
                  <td align="center" colspan="1" rowspan="1">41.50</td>
                  <td align="center" colspan="1" rowspan="1">32.34</td>
                  <td align="center" colspan="1" rowspan="1">1.94</td>
                  <td align="center" colspan="1" rowspan="1">18.1</td>
               </tr>
            </tbody>
         </table>
      </div>
      <div class="p text-right font-secondary"><a class="usa-link" href="table/T2/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div>
   </section>
</section>
"""
    return dataframe_to_markdown(html_content)

@pytest.fixture(scope="module")
def caption():
    return "Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0-∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life."

@pytest.fixture(scope="module")
def md_table_drug():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| N/A | N/A | Plasma |
"""

@pytest.fixture(scope="module")
def md_table_patient():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| N/A | N/A | 15 |
"""

@pytest.fixture(scope="module")
def md_table_aligned():
    return """
| Parameter type | N | Range | Mean ± s.d. | Median |
| --- | --- | --- | --- | --- |
| Cmax(ng/mL) | 15 | 29.3–209.6 | 56.1 ± 44.9 | 42.2 |
| AUC0−∞ | 15 | 253.3–3202.5 | 822.5 ± 706.1 | 601.5 |
| CL(mL/min/kg) | 15 | 3.33–131.50 | 49.33 ± 30.83 | 41.50 |
| CL(mL/min/m) | 15 | 5.5–67.5 | 31.95 ± 13.99 | 32.34 |
| Vdz(L/kg) | 15 | 0.33–4.05 | 1.92 ± 0.84 | 1.94 |
| T1/2(hr) | 15 | 9.5–47.0 | 20.5 ± 10.2 | 18.1 |
"""

@pytest.fixture(scope="module")
def col_mapping():
    return {
    "Parameter type": "Parameter type",
    "N": "Uncategorized",
    "Range": "Parameter value",
    "Mean ± s.d.": "Parameter value",
    "Median": "Parameter value"
}

@pytest.fixture(scope="module")
def md_table_list():
    return ["""
| Parameter type | Range |
| --- | --- |
| Cmax(ng/mL) | 29.3–209.6 |
| AUC0−∞ | 253.3–3202.5 |
| CL(mL/min/kg) | 3.33–131.50 |
| CL(mL/min/m) | 5.5–67.5 |
| Vdz(L/kg) | 0.33–4.05 |
| T1/2(hr) | 9.5–47.0 |
""", """
| Parameter type | Mean ± s.d. |
| --- | --- |
| Cmax(ng/mL) | 56.1 ± 44.9 |
| AUC0−∞ | 822.5 ± 706.1 |
| CL(mL/min/kg) | 49.33 ± 30.83 |
| CL(mL/min/m) | 31.95 ± 13.99 |
| Vdz(L/kg) | 1.92 ± 0.84 |
| T1/2(hr) | 20.5 ± 10.2 |
""", """
| Parameter type | Median |
| --- | --- |
| Cmax(ng/mL) | 42.2 |
| AUC0−∞ | 601.5 |
| CL(mL/min/kg) | 41.50 |
| CL(mL/min/m) | 32.34 |
| Vdz(L/kg) | 1.94 |
"""]


