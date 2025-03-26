
import os
from typing import Optional
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe, single_html_table_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage

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
    return get_azure_openai()

ghtml_content = """
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

ghtml_content1 = """
<section class="tw xbox font-sm" id="T3"><h4 class="obj_head">Table 3.</h4>\n<div class="caption p"><p id="P41">Bayesian pharmacokinetics parameters (all subjects). CL is clearance. Vdss is volume of distribution at steady state. Beta is the terminal slope of the log concentration versus time profile. T<sub>½</sub> Beta is the elimination half-life.</p></div>\n<div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups">\n<thead><tr>\n<th align="left" colspan="1" rowspan="1"></th>\n<th align="center" colspan="1" rowspan="1" valign="top">Free Fraction</th>\n<th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/kg)</th>\n<th align="center" colspan="1" rowspan="1">CL<br/>mL/min/m<sup>2</sup>)</th>\n<th align="center" colspan="1" rowspan="1">Vdss<br/>(L/kg)</th>\n<th align="center" colspan="1" rowspan="1">Beta<br/>(hr<sup>−1</sup>)</th>\n<th align="center" colspan="1" rowspan="1">T<sub>½</sub> Beta<br/>(hr)</th>\n</tr></thead>\n<tbody>\n<tr>\n<td align="left" colspan="1" rowspan="1"><strong>Overall</strong></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">N</td>\n<td align="center" colspan="1" rowspan="1">61</td>\n<td align="center" colspan="1" rowspan="1">63</td>\n<td align="center" colspan="1" rowspan="1">63</td>\n<td align="center" colspan="1" rowspan="1">63</td>\n<td align="center" colspan="1" rowspan="1">63</td>\n<td align="center" colspan="1" rowspan="1">63</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Range</td>\n<td align="center" colspan="1" rowspan="1">0.07–0.48</td>\n<td align="center" colspan="1" rowspan="1">0.3–7.75</td>\n<td align="center" colspan="1" rowspan="1">6.50–147.17</td>\n<td align="center" colspan="1" rowspan="1">0.49–3.40</td>\n<td align="center" colspan="1" rowspan="1">0.017–0.118</td>\n<td align="center" colspan="1" rowspan="1">5.9–42.0</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>\n<td align="center" colspan="1" rowspan="1">0.10 ± 0.05</td>\n<td align="center" colspan="1" rowspan="1">1.2 ± 0.93</td>\n<td align="center" colspan="1" rowspan="1">33.33 ± 19.33</td>\n<td align="center" colspan="1" rowspan="1">1.48 ± 0.54</td>\n<td align="center" colspan="1" rowspan="1">0.048 ± 0.020</td>\n<td align="center" colspan="1" rowspan="1">16.8 ± 7.1</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Median</td>\n<td align="center" colspan="1" rowspan="1">0.09</td>\n<td align="center" colspan="1" rowspan="1">1.08</td>\n<td align="center" colspan="1" rowspan="1">29.00</td>\n<td align="center" colspan="1" rowspan="1">1.37</td>\n<td align="center" colspan="1" rowspan="1">0.046</td>\n<td align="center" colspan="1" rowspan="1">15.1</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1"><strong>3 Month to &lt; 3 Years</strong></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">N</td>\n<td align="center" colspan="1" rowspan="1">17</td>\n<td align="center" colspan="1" rowspan="1">18</td>\n<td align="center" colspan="1" rowspan="1">18</td>\n<td align="center" colspan="1" rowspan="1">18</td>\n<td align="center" colspan="1" rowspan="1">18</td>\n<td align="center" colspan="1" rowspan="1">18</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Range</td>\n<td align="center" colspan="1" rowspan="1">0.07–0.48</td>\n<td align="center" colspan="1" rowspan="1">0.63–7.75</td>\n<td align="center" colspan="1" rowspan="1">12.83–147.17</td>\n<td align="center" colspan="1" rowspan="1">0.67–3.40</td>\n<td align="center" colspan="1" rowspan="1">0.024–0.118</td>\n<td align="center" colspan="1" rowspan="1">5.9–28.4</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>\n<td align="center" colspan="1" rowspan="1">0.11 ± 0.10</td>\n<td align="center" colspan="1" rowspan="1">1.57 ± 1.62</td>\n<td align="center" colspan="1" rowspan="1">32.83 ± 30.17</td>\n<td align="center" colspan="1" rowspan="1">1.62 ± 0.59</td>\n<td align="center" colspan="1" rowspan="1">0.053 ± 0.027</td>\n<td align="center" colspan="1" rowspan="1">15.8 ± 6.5</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1"><strong>3 to &lt; 13 Years</strong></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">N</td>\n<td align="center" colspan="1" rowspan="1">28</td>\n<td align="center" colspan="1" rowspan="1">29</td>\n<td align="center" colspan="1" rowspan="1">29</td>\n<td align="center" colspan="1" rowspan="1">29</td>\n<td align="center" colspan="1" rowspan="1">29</td>\n<td align="center" colspan="1" rowspan="1">29</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Range</td>\n<td align="center" colspan="1" rowspan="1">0.07–0.17</td>\n<td align="center" colspan="1" rowspan="1">0.30–1.82</td>\n<td align="center" colspan="1" rowspan="1">6.50–69.17</td>\n<td align="center" colspan="1" rowspan="1">0.49–3.00</td>\n<td align="center" colspan="1" rowspan="1">0.017–0.092</td>\n<td align="center" colspan="1" rowspan="1">7.5–40.6</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>\n<td align="center" colspan="1" rowspan="1">0.10 ± 0.02</td>\n<td align="center" colspan="1" rowspan="1">1.12 ± 0.40</td>\n<td align="center" colspan="1" rowspan="1">31.83 ± 13.83</td>\n<td align="center" colspan="1" rowspan="1">1.50 ± 0.61</td>\n<td align="center" colspan="1" rowspan="1">0.048 ± 0.017</td>\n<td align="center" colspan="1" rowspan="1">16.9 ± 7.4</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1"><strong>13 to &lt; 18 Years</strong></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n<td align="center" colspan="1" rowspan="1"></td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">N</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n<td align="center" colspan="1" rowspan="1">16</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Range</td>\n<td align="center" colspan="1" rowspan="1">0.07–0.15</td>\n<td align="center" colspan="1" rowspan="1">0.43–1.58</td>\n<td align="center" colspan="1" rowspan="1">16.33–60.00</td>\n<td align="center" colspan="1" rowspan="1">1.00–1.54</td>\n<td align="center" colspan="1" rowspan="1">0.017–0.084</td>\n<td align="center" colspan="1" rowspan="1">8.2–42.0</td>\n</tr>\n<tr>\n<td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>\n<td align="center" colspan="1" rowspan="1">0.09 ± 0.02</td>\n<td align="center" colspan="1" rowspan="1">0.95 ± 0.32</td>\n<td align="center" colspan="1" rowspan="1">36.67 ± 12.00</td>\n<td align="center" colspan="1" rowspan="1">1.27 ± 0.17</td>\n<td align="center" colspan="1" rowspan="1">0.044 ± 0.016</td>\n<td align="center" colspan="1" rowspan="1">17.8 ± 7.7</td>\n</tr>\n</tbody>\n</table></div>\n<div class="p text-right font-secondary"><a class="usa-link" href="table/T3/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div></section>
"""
@pytest.fixture(scope="module")
def html_content():
    return ghtml_content

@pytest.fixture(scope="module")
def md_table():
    return single_html_table_to_markdown(ghtml_content)

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

@pytest.fixture(scope="module")
def df_combined():
    md_table_combined = """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Statistics type | Main value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Cmax | ng/mL | N/A | N/A | N/A | N/A | Range | 29.3 | 209.6 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | AUC0−∞ | ng·h/mL | N/A | N/A | N/A | N/A | Range | 253.3 | 3202.5 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (weight-normalized) | mL/min/kg | N/A | N/A | N/A | N/A | Range | 3.33 | 131.50 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (surface-area-normalized) | mL/min/m | N/A | N/A | N/A | N/A | Range | 5.5 | 67.5 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Vdz | L/kg | N/A | N/A | N/A | N/A | Range | 0.33 | 4.05 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | T1/2 | hr | N/A | N/A | N/A | N/A | Range | 9.5 | 47.0 | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Cmax | ng/mL | Mean | 56.1 | SD | 44.9 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | AUC0−∞ | ng·h/mL | Mean | 822.5 | SD | 706.1 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (weight-normalized) | mL/min/kg | Mean | 49.33 | SD | 30.83 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (surface-area-normalized) | mL/min/m | Mean | 31.95 | SD | 13.99 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Vdz | L/kg | Mean | 1.92 | SD | 0.84 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | T1/2 | hr | Mean | 20.5 | SD | 10.2 | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Cmax | ng/mL | Median | 42.2 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | AUC0−∞ | ng·h/mL | Median | 601.5 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (weight-normalized) | mL/min/kg | Median | 41.50 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | CL (surface-area-normalized) | mL/min/m | Median | 32.34 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | Vdz | L/kg | Median | 1.94 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | N/A | 15 | T1/2 | hr | Median | 18.1 | N/A | N/A | N/A | N/A | N/A | N/A |
"""
    return markdown_to_dataframe(md_table_combined)

@pytest.fixture(scope="module")
def html_content1():
    return ghtml_content1

@pytest.fixture(scope="module")
def md_table1():
    return single_html_table_to_markdown(ghtml_content1)

@pytest.fixture(scope="module")
def caption1():
    return "Bayesian pharmacokinetics parameters (all subjects). CL is clearance. Vdss is volume of distribution at steady state. Beta is the terminal slope of the log concentration versus time profile. T½ Beta is the elimination half-life."

@pytest.fixture(scope="module")
def md_table_aligned1():
    return """
| Parameter type | Overall | N_0 | Range_0 | Mean ± s.d._0 | Median | 3 Month to < 3 Years | N_1 | Range_1 | Mean ± s.d._1 | 3 to < 13 Years | N_2 | Range_2 | Mean ± s.d._2 | 13 to < 18 Years | N_3 | Range_3 | Mean ± s.d._3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Free Fraction |  | 61 | 0.07–0.48 | 0.10 ± 0.05 | 0.09 |  | 17 | 0.07–0.48 | 0.11 ± 0.10 |  | 28 | 0.07–0.17 | 0.10 ± 0.02 |  | 16 | 0.07–0.15 | 0.09 ± 0.02 |
| CL(mL/min/kg) |  | 63 | 0.3–7.75 | 1.2 ± 0.93 | 1.08 |  | 18 | 0.63–7.75 | 1.57 ± 1.62 |  | 29 | 0.30–1.82 | 1.12 ± 0.40 |  | 16 | 0.43–1.58 | 0.95 ± 0.32 |
| CLmL/min/m) |  | 63 | 6.50–147.17 | 33.33 ± 19.33 | 29.00 |  | 18 | 12.83–147.17 | 32.83 ± 30.17 |  | 29 | 6.50–69.17 | 31.83 ± 13.83 |  | 16 | 16.33–60.00 | 36.67 ± 12.00 |
| Vdss(L/kg) |  | 63 | 0.49–3.40 | 1.48 ± 0.54 | 1.37 |  | 18 | 0.67–3.40 | 1.62 ± 0.59 |  | 29 | 0.49–3.00 | 1.50 ± 0.61 |  | 16 | 1.00–1.54 | 1.27 ± 0.17 |
| Beta(hr) |  | 63 | 0.017–0.118 | 0.048 ± 0.020 | 0.046 |  | 18 | 0.024–0.118 | 0.053 ± 0.027 |  | 29 | 0.017–0.092 | 0.048 ± 0.017 |  | 16 | 0.017–0.084 | 0.044 ± 0.016 |
| T½Beta(hr) |  | 63 | 5.9–42.0 | 16.8 ± 7.1 | 15.1 |  | 18 | 5.9–28.4 | 15.8 ± 6.5 |  | 29 | 7.5–40.6 | 16.9 ± 7.4 |  | 16 | 8.2–42.0 | 17.8 ± 7.7 |
"""

@pytest.fixture(scope="module")
def md_table_list1():
    return ["""| Parameter type | Range_0 |
            | --- | --- |
            | Free Fraction | 0.07–0.48 |
            | CL(mL/min/kg) | 0.3–7.75 |
            | CLmL/min/m) | 6.50–147.17 |
            | Vdss(L/kg) | 0.49–3.40 |
            | Beta(hr) | 0.017–0.118 |
            | T½Beta(hr) | 5.9–42.0 |""", """| Parameter type | Mean ± s.d._0 |
            | --- | --- |
            | Free Fraction | 0.10 ± 0.05 |
            | CL(mL/min/kg) | 1.2 ± 0.93 |
            | CLmL/min/m) | 33.33 ± 19.33 |
            | Vdss(L/kg) | 1.48 ± 0.54 |
            | Beta(hr) | 0.048 ± 0.020 |
            | T½Beta(hr) | 16.8 ± 7.1 |""", """| Parameter type | Median |
            | --- | --- |
            | Free Fraction | 0.09 |
            | CL(mL/min/kg) | 1.08 |
            | CLmL/min/m) | 29.00 |
            | Vdss(L/kg) | 1.37 |
            | Beta(hr) | 0.046 |
            | T½Beta(hr) | 15.1 |""", """| Parameter type | Range_1 |
            | --- | --- |
            | Free Fraction | 0.07–0.48 |
            | CL(mL/min/kg) | 0.63–7.75 |
            | CLmL/min/m) | 12.83–147.17 |
            | Vdss(L/kg) | 0.67–3.40 |
            | Beta(hr) | 0.024–0.118 |
            | T½Beta(hr) | 5.9–28.4 |""", """| Parameter type | Mean ± s.d._1 |
            | --- | --- |
            | Free Fraction | 0.11 ± 0.10 |
            | CL(mL/min/kg) | 1.57 ± 1.62 |
            | CLmL/min/m) | 32.83 ± 30.17 |
            | Vdss(L/kg) | 1.62 ± 0.59 |
            | Beta(hr) | 0.053 ± 0.027 |
            | T½Beta(hr) | 15.8 ± 6.5 |""", """| Parameter type | Range_2 |
            | --- | --- |
            | Free Fraction | 0.07–0.17 |
            | CL(mL/min/kg) | 0.30–1.82 |
            | CLmL/min/m) | 6.50–69.17 |
            | Vdss(L/kg) | 0.49–3.00 |
            | Beta(hr) | 0.017–0.092 |
            | T½Beta(hr) | 7.5–40.6 |""", """| Parameter type | Mean ± s.d._2 |
            | --- | --- |
            | Free Fraction | 0.10 ± 0.02 |
            | CL(mL/min/kg) | 1.12 ± 0.40 |
            | CLmL/min/m) | 31.83 ± 13.83 |
            | Vdss(L/kg) | 1.50 ± 0.61 |
            | Beta(hr) | 0.048 ± 0.017 |
            | T½Beta(hr) | 16.9 ± 7.4 |""", """| Parameter type | Range_3 |
            | --- | --- |
            | Free Fraction | 0.07–0.15 |
            | CL(mL/min/kg) | 0.43–1.58 |
            | CLmL/min/m) | 16.33–60.00 |
            | Vdss(L/kg) | 1.00–1.54 |
            | Beta(hr) | 0.017–0.084 |
            | T½Beta(hr) | 8.2–42.0 |""", """| Parameter type | Mean ± s.d._3 |
            | --- | --- |
            | Free Fraction | 0.09 ± 0.02 |
            | CL(mL/min/kg) | 0.95 ± 0.32 |
            | CLmL/min/m) | 36.67 ± 12.00 |
            | Vdss(L/kg) | 1.27 ± 0.17 |
            | Beta(hr) | 0.044 ± 0.016 |
            | T½Beta(hr) | 17.8 ± 7.7 |"""
        ]
     
@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    level = logging.INFO
    logging.basicConfig(level=level)
    file_handler = logging.FileHandler("./logs/test.log")
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

@pytest.fixture(scope="module")
def step_callback():
    total_tokens = {**DEFAULT_TOKEN_USAGE}
    def print_step(
        step_name: Optional[str]=None, 
        step_description: Optional[str]=None,
        step_output: Optional[str]=None,
        step_reasoning_process: Optional[str]=None,
        token_usage: Optional[dict]=None,
    ):
        nonlocal total_tokens
        logger = logging.getLogger(__name__)
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if step_description is not None:
            logger.info(step_description)
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            total_tokens = increase_token_usage(total_tokens, token_usage)
            logger.info(
                f"overall total tokens: {total_tokens['total_tokens']}, overall prompt tokens: {total_tokens['prompt_tokens']}, overall completion tokens: {total_tokens['completion_tokens']}"
            )
        if step_reasoning_process is not None:
            logger.info(f"\n\n{step_reasoning_process}\n\n")
        if step_output is not None:
            logger.info(step_output)
    return print_step

