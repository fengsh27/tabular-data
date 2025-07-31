import os
from typing import Optional
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage

load_dotenv()


def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_NAME"),
    )


def get_azure_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
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
        "Median": "Parameter value",
    }


@pytest.fixture(scope="module")
def md_table_list():
    return [
        """
| Parameter type | Range |
| --- | --- |
| Cmax(ng/mL) | 29.3–209.6 |
| AUC0−∞ | 253.3–3202.5 |
| CL(mL/min/kg) | 3.33–131.50 |
| CL(mL/min/m) | 5.5–67.5 |
| Vdz(L/kg) | 0.33–4.05 |
| T1/2(hr) | 9.5–47.0 |
""",
        """
| Parameter type | Mean ± s.d. |
| --- | --- |
| Cmax(ng/mL) | 56.1 ± 44.9 |
| AUC0−∞ | 822.5 ± 706.1 |
| CL(mL/min/kg) | 49.33 ± 30.83 |
| CL(mL/min/m) | 31.95 ± 13.99 |
| Vdz(L/kg) | 1.92 ± 0.84 |
| T1/2(hr) | 20.5 ± 10.2 |
""",
        """
| Parameter type | Median |
| --- | --- |
| Cmax(ng/mL) | 42.2 |
| AUC0−∞ | 601.5 |
| CL(mL/min/kg) | 41.50 |
| CL(mL/min/m) | 32.34 |
| Vdz(L/kg) | 1.94 |
""",
    ]


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
    return [
        """| Parameter type | Range_0 |
            | --- | --- |
            | Free Fraction | 0.07–0.48 |
            | CL(mL/min/kg) | 0.3–7.75 |
            | CLmL/min/m) | 6.50–147.17 |
            | Vdss(L/kg) | 0.49–3.40 |
            | Beta(hr) | 0.017–0.118 |
            | T½Beta(hr) | 5.9–42.0 |""",
        """| Parameter type | Mean ± s.d._0 |
            | --- | --- |
            | Free Fraction | 0.10 ± 0.05 |
            | CL(mL/min/kg) | 1.2 ± 0.93 |
            | CLmL/min/m) | 33.33 ± 19.33 |
            | Vdss(L/kg) | 1.48 ± 0.54 |
            | Beta(hr) | 0.048 ± 0.020 |
            | T½Beta(hr) | 16.8 ± 7.1 |""",
        """| Parameter type | Median |
            | --- | --- |
            | Free Fraction | 0.09 |
            | CL(mL/min/kg) | 1.08 |
            | CLmL/min/m) | 29.00 |
            | Vdss(L/kg) | 1.37 |
            | Beta(hr) | 0.046 |
            | T½Beta(hr) | 15.1 |""",
        """| Parameter type | Range_1 |
            | --- | --- |
            | Free Fraction | 0.07–0.48 |
            | CL(mL/min/kg) | 0.63–7.75 |
            | CLmL/min/m) | 12.83–147.17 |
            | Vdss(L/kg) | 0.67–3.40 |
            | Beta(hr) | 0.024–0.118 |
            | T½Beta(hr) | 5.9–28.4 |""",
        """| Parameter type | Mean ± s.d._1 |
            | --- | --- |
            | Free Fraction | 0.11 ± 0.10 |
            | CL(mL/min/kg) | 1.57 ± 1.62 |
            | CLmL/min/m) | 32.83 ± 30.17 |
            | Vdss(L/kg) | 1.62 ± 0.59 |
            | Beta(hr) | 0.053 ± 0.027 |
            | T½Beta(hr) | 15.8 ± 6.5 |""",
        """| Parameter type | Range_2 |
            | --- | --- |
            | Free Fraction | 0.07–0.17 |
            | CL(mL/min/kg) | 0.30–1.82 |
            | CLmL/min/m) | 6.50–69.17 |
            | Vdss(L/kg) | 0.49–3.00 |
            | Beta(hr) | 0.017–0.092 |
            | T½Beta(hr) | 7.5–40.6 |""",
        """| Parameter type | Mean ± s.d._2 |
            | --- | --- |
            | Free Fraction | 0.10 ± 0.02 |
            | CL(mL/min/kg) | 1.12 ± 0.40 |
            | CLmL/min/m) | 31.83 ± 13.83 |
            | Vdss(L/kg) | 1.50 ± 0.61 |
            | Beta(hr) | 0.048 ± 0.017 |
            | T½Beta(hr) | 16.9 ± 7.4 |""",
        """| Parameter type | Range_3 |
            | --- | --- |
            | Free Fraction | 0.07–0.15 |
            | CL(mL/min/kg) | 0.43–1.58 |
            | CLmL/min/m) | 16.33–60.00 |
            | Vdss(L/kg) | 1.00–1.54 |
            | Beta(hr) | 0.017–0.084 |
            | T½Beta(hr) | 8.2–42.0 |""",
        """| Parameter type | Mean ± s.d._3 |
            | --- | --- |
            | Free Fraction | 0.09 ± 0.02 |
            | CL(mL/min/kg) | 0.95 ± 0.32 |
            | CLmL/min/m) | 36.67 ± 12.00 |
            | Vdss(L/kg) | 1.27 ± 0.17 |
            | Beta(hr) | 0.044 ± 0.016 |
            | T½Beta(hr) | 17.8 ± 7.7 |""",
    ]


@pytest.fixture(scope="module")
def html_content_29943508():
    return """
<section class="tw xbox font-sm" id="aas13175-tbl-0002" lang="en"><h4 class="obj_head">Table 2.</h4> <div class="caption p"><p>Fentanyl concentrations in umbilical vein and maternal serum. Data are presented as mean (SD) or median [interquartile range] as appropriate</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <col span="1" style="border-right:solid 1px #000000"/> <col span="1" style="border-right:solid 1px #000000"/> <col span="1" style="border-right:solid 1px #000000"/> <col span="1" style="border-right:solid 1px #000000"/> <col span="1" style="border-right:solid 1px #000000"/> <thead valign="top"><tr style="border-bottom:solid 1px #000000"> <th align="left" colspan="1" rowspan="1" valign="top">Variable</th> <th align="center" colspan="1" rowspan="1" valign="top">Adrenaline group (n = 19)</th> <th align="center" colspan="1" rowspan="1" valign="top">Control group (n = 20)</th> <th align="center" colspan="1" rowspan="1" valign="top">Mean difference</th> <th align="center" colspan="1" rowspan="1" valign="top"> <em>P</em>‐value</th> </tr></thead> <tbody> <tr> <td align="left" colspan="1" rowspan="1">Mean serum fentanyl concentration, umbilical vein (nmol/L)</td> <td align="center" colspan="1" rowspan="1">0.162 (0.090) (n = 16)</td> <td align="center" colspan="1" rowspan="1">0.151 (0.070) (n = 20)</td> <td align="center" colspan="1" rowspan="1">0.012 [−0.042; 0.065]</td> <td align="center" colspan="1" rowspan="1">.67</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Median maternal serum fentanyl concentration at birth (nmol/L)</td> <td align="center" colspan="1" rowspan="1">0.268 [0.193; 0.493]<a class="usa-link" href="#aas13175-note-0005"><sup>a</sup></a> (n = 16)</td> <td align="center" colspan="1" rowspan="1">0.291 [0.212; 0.502]<a class="usa-link" href="#aas13175-note-0005"><sup>a</sup></a> (n = 19)</td> <td align="center" colspan="1" rowspan="1">−0.061 [−0.205; 0.082]</td> <td align="center" colspan="1" rowspan="1">.66<a class="usa-link" href="#aas13175-note-0005"><sup>a</sup></a> </td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean AUC 0‐120 min for fentanyl in maternal serum (nmol h/L)</td> <td align="center" colspan="1" rowspan="1">0.428 (0.162) (n = 18)</td> <td align="center" colspan="1" rowspan="1">0.590 (0.197) (n = 15)<a class="usa-link" href="#aas13175-note-0006"><sup>b</sup></a> </td> <td align="center" colspan="1" rowspan="1">−0.162 [−0.289; −0.034]</td> <td align="center" colspan="1" rowspan="1">.015</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/aas13175-tbl-0002/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div> <div class="tw-foot p"> <div class="fn" id="aas13175-note-0004"><p>AUC, Area under the curve. Student's <em>t</em> test was used to calculate <em>P</em>‐values unless otherwise specified. Complete case analysis, numbers in some cells lower than the total numbers of patients included due to missing data (hemolysis of samples, technical laboratory difficulties).</p></div> <div class="fn" id="aas13175-note-0005"> <sup>a</sup><p class="display-inline">Mann–Whitney <em>U</em> test used. Data presented as median [25th; 75th percentile].</p> </div> <div class="fn" id="aas13175-note-0006"> <sup>b</sup><p class="display-inline">Two cases with missing data due to birth prior to 120 min sample.</p> </div> </div></section>
"""


@pytest.fixture(scope="module")
def caption_29943508():
    return """
Fentanyl concentrations in umbilical vein and maternal serum. Data are presented as mean (SD) or median [interquartile range] as appropriate
AUC, Area under the curve. Student's t test was used to calculate P‐values unless otherwise specified. Complete case analysis, numbers in some cells lower than the total numbers of patients included due to missing data (hemolysis of samples, technical laboratory difficulties).

aMann–Whitney U test used. Data presented as median [25th; 75th percentile].

bTwo cases with missing data due to birth prior to 120 min sample.
"""


@pytest.fixture(scope="module")
def md_table_aligned_29943508():
    return """
| Parameter type | Adrenaline group (n\xa0=\xa019) | Control group (n\xa0=\xa020) | Mean difference | P‐value |
| --- | --- | --- | --- | --- |
| Mean serum fentanyl concentration, umbilical vein (nmol/L) | 0.162 (0.090) (n\xa0=\xa016) | 0.151 (0.070) (n\xa0=\xa020) | 0.012 [−0.042; 0.065] | .67 |
| Median maternal serum fentanyl concentration at birth (nmol/L) | 0.268 [0.193; 0.493](n\xa0=\xa016) | 0.291 [0.212; 0.502](n\xa0=\xa019) | −0.061 [−0.205; 0.082] | .66 |
| Mean AUC 0‐120\xa0min for fentanyl in maternal serum (nmol\xa0h/L) | 0.428 (0.162) (n\xa0=\xa018) | 0.590 (0.197) (n\xa0=\xa015) | −0.162 [−0.289; −0.034] | .015 |
"""


@pytest.fixture(scope="module")
def col_mapping_29943508():
    return {
        "Parameter type": "Parameter type",
        "Adrenaline group (n = 19)": "Parameter value",
        "Control group (n = 20)": "Parameter value",
        "Mean difference": "Parameter value",
        "P‐value": "P value",
    }


## =============================================================================
# 16143486_table_4

ghtml_content_16143486_table_4 = """
<div class="tables frame-topbot rowsep-0 colsep-0" id="tbl4"><span class="captions text-s"><span><p><span class="label">Table 4</span>. Transplacental distribution of lorazepam as an enantiomeric mixture at delivery (<em>n</em> = 8)</p></span></span><div class="groups"><table><thead class="valign-top"><tr><th class="rowsep-1 align-left" scope="col">Parturient</th><th class="rowsep-1 align-left" scope="col">Cord blood (ng/ml)</th><th class="rowsep-1 align-left" scope="col">Maternal blood (ng/ml)</th><th class="rowsep-1 align-left" scope="col">Collection time<a class="anchor anchor-primary" data-sd-ui-side-panel-opener="true" data-xocs-content-id="tbl4fn1" data-xocs-content-type="reference" href="#tbl4fn1" name="btbl4fn1"><span class="anchor-text-container"><span class="anchor-text"><sup>a</sup></span></span></a> (min)</th><th class="rowsep-1 align-left" scope="col">Cord blood/maternal blood</th></tr></thead><tbody><tr><td class="align-left">1</td><td class="align-left">5.77</td><td class="align-left">14.74</td><td class="align-left">135</td><td class="align-left">0.392</td></tr><tr><td class="align-left">2</td><td class="align-left">6.82</td><td class="align-left">7.95</td><td class="align-left">426</td><td class="align-left">0.858</td></tr><tr><td class="align-left">3</td><td class="align-left">4.38</td><td class="align-left">10.48</td><td class="align-left">153</td><td class="align-left">0.418</td></tr><tr><td class="align-left">4</td><td class="align-left">8.42</td><td class="align-left">9.60</td><td class="align-left">300</td><td class="align-left">0.878</td></tr><tr><td class="align-left">5</td><td class="align-left">5.87</td><td class="align-left">5.33</td><td class="align-left">390</td><td class="align-left">1.100</td></tr><tr><td class="align-left">6</td><td class="align-left">5.78</td><td class="align-left">9.87</td><td class="align-left">120</td><td class="align-left">0.586</td></tr><tr><td class="align-left">7</td><td class="align-left">7.75</td><td class="align-left">10.94</td><td class="align-left">552</td><td class="align-left">0.708</td></tr><tr><td class="align-left">8</td><td class="align-left">9.45</td><td class="align-left">10.35</td><td class="align-left">207</td><td class="align-left">0.913</td></tr><tr><td class="align-left" colspan="5"><br/></td></tr><tr><td class="align-left">Mean CI 95%</td><td class="align-left">6.78 (5.39–8.17)</td><td class="align-left">9.91 (7.68–12.14)</td><td class="align-left">293.4 (163.2–423)</td><td class="align-left">0.73 (0.52–0.94)</td></tr></tbody></table></div><div class="legend"><div class="u-margin-s-bottom">Parturients were treated with a single oral dose of 2 mg rac-lorazepam; mean (CI 95%).</div></div><dl class="footnotes"><dt id="tbl4fn1">a</dt><dd><div class="u-margin-s-bottom">Time between drug intake and blood collection from the umbilical cord and maternal vein.</div></dd></dl></div>
"""


@pytest.fixture(scope="module")
def html_content_16143486_table_4():
    return ghtml_content_16143486_table_4


@pytest.fixture(scope="module")
def caption_16143486_table_4():
    return """
Transplacental distribution of lorazepam as an enantiomeric mixture at delivery (n = 8)
Parturients were treated with a single oral dose of 2 mg rac-lorazepam; mean (CI 95%).
"""


@pytest.fixture(scope="module")
def md_table_16143486_table_4():
    return single_html_table_to_markdown(ghtml_content_16143486_table_4)


@pytest.fixture(scope="module")
def md_table_drug_16143486_table_4():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Lorazepam | Lorazepam | Cord blood |
| Lorazepam | Lorazepam | Maternal blood |
"""


@pytest.fixture(scope="module")
def md_table_patient_16143486_table_4():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Parturient | at delivery | 8 |
"""


@pytest.fixture(scope="module")
def md_table_patient_refined_16143486_table_4():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Maternal | Delivery | N/A | 8 |
"""


@pytest.fixture(scope="module")
def md_table_summary_16143486_table_4():
    return """
| Parturient | Cord blood (ng/ml) | Maternal blood (ng/ml) | Collection time(min) | Cord blood/maternal blood |
| --- | --- | --- | --- | --- |
| Mean CI 95% | 6.78 (5.39–8.17) | 9.91 (7.68–12.14) | 293.4 (163.2–423) | 0.73 (0.52–0.94) |
"""


@pytest.fixture(scope="module")
def col_mapping_16143486_table_4():
    return {"Parameter type": "Parameter type", "Mean CI 95%": "Parameter value"}


@pytest.fixture(scope="module")
def md_table_list_16143486_table_4():
    return [
        """
| Parameter type | Mean CI 95% |
| --- | --- |
| Cord blood (ng/ml) | 6.78 (5.39–8.17) |
| Maternal blood (ng/ml) | 9.91 (7.68–12.14) |
| Collection time(min) | 293.4 (163.2–423) |
| Cord blood/maternal blood | 0.73 (0.52–0.94) |
"""
    ]


@pytest.fixture(scope="module")
def md_table_aligned_16143486_table_4():
    return """
| Parameter type | Mean CI 95% |
| --- | --- |
| Cord blood (ng/ml) | 6.78 (5.39–8.17) |
| Maternal blood (ng/ml) | 9.91 (7.68–12.14) |
| Collection time(min) | 293.4 (163.2–423) |
| Cord blood/maternal blood | 0.73 (0.52–0.94) |
"""


@pytest.fixture(scope="module")
def type_unit_list_16143486_table_4():
    return [
        """
| Parameter type | Parameter unit |
| --- | --- |
| Cord blood concentration | ng/ml |
| Maternal blood concentration | ng/ml |
| Sample collection time | minutes |
| Cord blood to maternal blood ratio | unitless |"""
    ]


## =============================================================================
# 30825333_table_2

ghtml_content_30825333_table_2 = """
<div class="article-table-content" id="phar2243-tbl-0002"> <header class="article-table-caption"><span class="table-caption__label">Table 2. </span>Equations Used for Calculation of Ketamine Pharmacokinetic Parameters</header> <div class="article-table-content-wrapper" tabindex="0"> <table class="table article-section__table"> <thead> <tr> <th class="bottom-bordered-cell right-bordered-cell left-aligned">Parameter</th> <th class="bottom-bordered-cell center-aligned">Abbreviation</th> <th class="bottom-bordered-cell center-aligned">Equation</th> </tr> </thead> <tbody> <tr> <td class="right-bordered-cell left-aligned">Total area under curve</td> <td class="center-aligned">AUC<sub>0→∞</sub></td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0001" class="section_image" loading="lazy" src="/cms/asset/9683e94a-bcc5-47d2-ba3f-5146380cb3ea/phar2243-math-0001.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Total area under moment curve</td> <td class="center-aligned">AUMC<sub>0→∞</sub></td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0002" class="section_image" loading="lazy" src="/cms/asset/62ace473-892b-4e2e-b554-f4000769a6c1/phar2243-math-0002.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Total body clearance</td> <td class="center-aligned">CL</td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0003" class="section_image" loading="lazy" src="/cms/asset/e663a0e3-fd9c-4112-bb21-f8db34566b45/phar2243-math-0003.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Mean residence time</td> <td class="center-aligned">MRT</td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0004" class="section_image" loading="lazy" src="/cms/asset/9121d278-38a3-4978-9ebe-d0d40eb3ea11/phar2243-math-0004.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Distribution volume at steady state</td> <td class="center-aligned"> <i>V</i> <sub>ss</sub> </td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0005" class="section_image" loading="lazy" src="/cms/asset/e0383310-809b-4b85-a191-c0d1baac0e2d/phar2243-math-0005.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Elimination half-life</td> <td class="center-aligned"> <i>t</i> <sub>0.5</sub> </td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0006" class="section_image" loading="lazy" src="/cms/asset/80d7b6ca-6713-4fe9-a648-8fff9d14947b/phar2243-math-0006.png"/> </td> </tr> </tbody> </table> </div> <div class="article-section__table-footnotes"> <ul> <li id="phar2243-note-0002"> <i>t</i><sub>last</sub>, <i>C</i><sub>last</sub> are the last observation time and concentrations, respectively. <i>R</i>,<i> T</i> are the continuous infusion rate and duration, respectively. <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0007" class="section_image" loading="lazy" src="/cms/asset/c119e6dd-9945-47bb-bc9f-e297d560dca1/phar2243-math-0007.png"/>, <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0008" class="section_image" loading="lazy" src="/cms/asset/2a652916-1643-4e07-a855-e49c19566e4f/phar2243-math-0008.png"/> are the area under curve and area under moment curve from zero to the last observation time, respectively. <i>k</i><sub>e</sub> is the elimination rate constant. </li> </ul> </div> <div class="article-section__table-source"></div> </div>
"""


@pytest.fixture(scope="module")
def html_content_30825333_table_2():
    return ghtml_content_30825333_table_2


@pytest.fixture(scope="module")
def caption_30825333_table_2():
    return """
Table 2. Equations Used for Calculation of Ketamine Pharmacokinetic Parameters
tlast, Clast are the last observation time and concentrations, respectively. R, T are the continuous infusion rate and duration, respectively. , are the area under curve and area under moment curve from zero to the last observation time, respectively. ke is the elimination rate constant.
"""


@pytest.fixture(scope="module")
def md_table_30825333_table_2():
    return single_html_table_to_markdown(ghtml_content_30825333_table_2)


@pytest.fixture(scope="module")
def md_table_drug_30825333_table_2():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Ketamine | Ketamine | Plasma |
"""


@pytest.fixture(scope="module")
def md_table_patient_30825333_table_2():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| N/A | N/A | N/A |
"""


@pytest.fixture(scope="module")
def md_table_patient_refined_30825333_table_2():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |
"""


@pytest.fixture(scope="module")
def md_table_summary_30825333_table_2():
    return """
| Parameter | Abbreviation | Equation |
| --- | --- | --- |
| Total area under curve | AUC0→∞ |  |
| Total area under moment curve | AUMC0→∞ |  |
| Total body clearance | CL |  |
| Mean residence time | MRT |  |
| Distribution volume at steady state | Vss |  |
| Elimination half-life | t0.5 |  |
"""


@pytest.fixture(scope="module")
def md_table_aligned_30825333_table_2():
    return """
| Parameter type | Abbreviation | Equation |
| --- | --- | --- |
| Total area under curve | AUC0→∞ |  |
| Total area under moment curve | AUMC0→∞ |  |
| Total body clearance | CL |  |
| Mean residence time | MRT |  |
| Distribution volume at steady state | Vss |  |
| Elimination half-life | t0.5 |  |
"""


@pytest.fixture(scope="module")
def col_mapping_30825333_table_2():
    return {
        "Parameter type": "Parameter type",
        "Abbreviation": "Uncategorized",
        "Equation": "Uncategorized",
    }


@pytest.fixture(scope="module")
def md_table_list_30825333_table_2():
    return [
        """
| Parameter type | Abbreviation | Equation |
| --- | --- | --- |
| Total area under curve | AUC0→∞ |  |
| Total area under moment curve | AUMC0→∞ |  |
| Total body clearance | CL |  |
| Mean residence time | MRT |  |
| Distribution volume at steady state | Vss |  |
| Elimination half-life | t0.5 |  |
"""
    ]


@pytest.fixture(scope="module")
def drug_list_30825333_table_2():
    return [
        """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Ketamine | Ketamine | Plasma |
| Ketamine | Ketamine | Plasma |
| Ketamine | Ketamine | Plasma |
| Ketamine | Ketamine | Plasma |
| Ketamine | Ketamine | Plasma |
| Ketamine | Ketamine | Plasma |"""
    ]


@pytest.fixture(scope="module")
def patient_list_30825333_table_2():
    return [
        """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A |"""
    ]


@pytest.fixture(scope="module")
def type_unit_list_30825333_table_2():
    return []


@pytest.fixture(scope="module")
def value_list_30825333_table_2():
    return []


@pytest.fixture(scope="module")
def df_combined_30825333_table_2():
    import pandas as pd

    return pd.DataFrame()


## =============================================================================
# 34183327_table_2

ghtml_content_30825333_table_2 = """
<div class="article-table-content" id="phar2243-tbl-0002"> <header class="article-table-caption"><span class="table-caption__label">Table 2. </span>Equations Used for Calculation of Ketamine Pharmacokinetic Parameters</header> <div class="article-table-content-wrapper" tabindex="0"> <table class="table article-section__table"> <thead> <tr> <th class="bottom-bordered-cell right-bordered-cell left-aligned">Parameter</th> <th class="bottom-bordered-cell center-aligned">Abbreviation</th> <th class="bottom-bordered-cell center-aligned">Equation</th> </tr> </thead> <tbody> <tr> <td class="right-bordered-cell left-aligned">Total area under curve</td> <td class="center-aligned">AUC<sub>0→∞</sub></td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0001" class="section_image" loading="lazy" src="/cms/asset/9683e94a-bcc5-47d2-ba3f-5146380cb3ea/phar2243-math-0001.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Total area under moment curve</td> <td class="center-aligned">AUMC<sub>0→∞</sub></td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0002" class="section_image" loading="lazy" src="/cms/asset/62ace473-892b-4e2e-b554-f4000769a6c1/phar2243-math-0002.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Total body clearance</td> <td class="center-aligned">CL</td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0003" class="section_image" loading="lazy" src="/cms/asset/e663a0e3-fd9c-4112-bb21-f8db34566b45/phar2243-math-0003.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Mean residence time</td> <td class="center-aligned">MRT</td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0004" class="section_image" loading="lazy" src="/cms/asset/9121d278-38a3-4978-9ebe-d0d40eb3ea11/phar2243-math-0004.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Distribution volume at steady state</td> <td class="center-aligned"> <i>V</i> <sub>ss</sub> </td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0005" class="section_image" loading="lazy" src="/cms/asset/e0383310-809b-4b85-a191-c0d1baac0e2d/phar2243-math-0005.png"/> </td> </tr> <tr> <td class="right-bordered-cell left-aligned">Elimination half-life</td> <td class="center-aligned"> <i>t</i> <sub>0.5</sub> </td> <td class="center-aligned"> <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0006" class="section_image" loading="lazy" src="/cms/asset/80d7b6ca-6713-4fe9-a648-8fff9d14947b/phar2243-math-0006.png"/> </td> </tr> </tbody> </table> </div> <div class="article-section__table-footnotes"> <ul> <li id="phar2243-note-0002"> <i>t</i><sub>last</sub>, <i>C</i><sub>last</sub> are the last observation time and concentrations, respectively. <i>R</i>,<i> T</i> are the continuous infusion rate and duration, respectively. <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0007" class="section_image" loading="lazy" src="/cms/asset/c119e6dd-9945-47bb-bc9f-e297d560dca1/phar2243-math-0007.png"/>, <img alt="urn:x-wiley:02770008:media:phar2243:phar2243-math-0008" class="section_image" loading="lazy" src="/cms/asset/2a652916-1643-4e07-a855-e49c19566e4f/phar2243-math-0008.png"/> are the area under curve and area under moment curve from zero to the last observation time, respectively. <i>k</i><sub>e</sub> is the elimination rate constant. </li> </ul> </div> <div class="article-section__table-source"></div> </div>
"""


@pytest.fixture(scope="module")
def html_content_30825333_table_2():
    return ghtml_content_30825333_table_2


@pytest.fixture(scope="module")
def md_table_drug_34183327_table_2():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Isoniazid | Isoniazid | Plasma |
| Isoniazid | Isoniazid | CSF |
| Rifampicin | Rifampicin | Plasma |
| Rifampicin | Rifampicin | CSF |
| Pyrazinamide | Pyrazinamide | Plasma |
| Pyrazinamide | Pyrazinamide | CSF |
"""


@pytest.fixture(scope="module")
def md_table_list_34183327_table_2():
    return [
        """
| Parameter type | Isoniazid | Age, years_0 | Random blood glucose, mg/dL_0 | Drug dose, mg/kg_0 | Drug administration via NGT, no/yes_0 | Rifampicin | Age, years_1 | Random blood glucose, mg/dL_1 | Drug dose, mg/kg_1 | Drug administration via NGT, no/yes_1 | Pyrazinamide | Random blood glucose, mg/dL_2 | Drug dose, mg/kg_2 | Drug administration via NGT, no/yes_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUC0–24,hour∙mg/L(B (95%\u2009CI)) |  | n/a | −0.002 (−0.006 to 0.003) | 0.016 (−0.048 to 0.080) | 0.439 (0.143 to 0.735)** |  | −0.009 (−0.028 to 0.010) | −0.003 (−0.007 to 0.001) | 0.014 (−0.021 to 0.048) | n/a |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
| Cmax, mg/L(B (95%\u2009CI)) |  | −0.020 (−0.043 to 0.003) | −0.004 (−0.009 to 0.001) | n/a | 0.130 (−0.160 to 0.420) |  | −0.008 (−0.029 to 0.012) | −0.005 (−0.009 to −0.0003)* | n/a | 0.067 (−0.194 to 0.328) |  | −0.003 (−0.005 to −0.001)** | 0.010 (0.001 to 0.020)* | 0.036 (−0.095 to 0.167) |
| CCSF0–8, mg/L(B (95%\u2009CI)) |  | n/a | −0.007 (−0.015 to 0.001) | 0.046 (−0.058 to 0.151) | 0.289 (−0.197 to 0.775) |  | −0.021 (−0.052 to 0.009) | n/a | 0.030 (−0.030 to 0.091) | 0.019 (−0.365 to 0.403) |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
"""
    ]


@pytest.fixture(scope="module")
def md_table_aligned_34183327_table_2():
    return """
| Parameter type | Isoniazid | Age, years_0 | Random blood glucose, mg/dL_0 | Drug dose, mg/kg_0 | Drug administration via NGT, no/yes_0 | Rifampicin | Age, years_1 | Random blood glucose, mg/dL_1 | Drug dose, mg/kg_1 | Drug administration via NGT, no/yes_1 | Pyrazinamide | Random blood glucose, mg/dL_2 | Drug dose, mg/kg_2 | Drug administration via NGT, no/yes_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUC0–24,hour∙mg/L(B (95% CI)) |  | n/a | −0.002 (−0.006 to 0.003) | 0.016 (−0.048 to 0.080) | 0.439 (0.143 to 0.735)** |  | −0.009 (−0.028 to 0.010) | −0.003 (−0.007 to 0.001) | 0.014 (−0.021 to 0.048) | n/a |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
| Cmax, mg/L(B (95% CI)) |  | −0.020 (−0.043 to 0.003) | −0.004 (−0.009 to 0.001) | n/a | 0.130 (−0.160 to 0.420) |  | −0.008 (−0.029 to 0.012) | −0.005 (−0.009 to −0.0003)* | n/a | 0.067 (−0.194 to 0.328) |  | −0.003 (−0.005 to −0.001)** | 0.010 (0.001 to 0.020)* | 0.036 (−0.095 to 0.167) |
| CCSF0–8, mg/L(B (95% CI)) |  | n/a | −0.007 (−0.015 to 0.001) | 0.046 (−0.058 to 0.151) | 0.289 (−0.197 to 0.775) |  | −0.021 (−0.052 to 0.009) | n/a | 0.030 (−0.030 to 0.091) | 0.019 (−0.365 to 0.403) |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
"""


@pytest.fixture(scope="module")
def caption_34183327_table_2():
    return """
Summary of pharmacokinetic (PK) parameters of isoniazid, rifampicin and pyrazinamide among Indonesian children treated for TBM
Data are presented as geometric mean (range). The first PK assessment was performed on day 2 of treatment and the second PK assessment was performed on day 10 of treatment. *Paired-sample t-test on log-transformed data of 12 patients for whom PK data were available both at the first and second PK assessments. †At the first PK assessment, 6, 7 and 7 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively; and at the second PK assessment, 4, 4 and 3 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively. AUC0–24, area under the plasma concentration–time curve from 0 to 24 hours postdose; C CSF0–8, drug concentration in cerebrospinal fluid during 0–8 hours postdose; C max, peak plasma concentration; n/a, non-applicable; TBM, tuberculous meningitis.
"""


## =============================================================================
# 35489632_table_2


@pytest.fixture(scope="module")
def caption_35489632_table_2():
    return """
Table 2. Comparisons of unbound plasma meropenem concentrations classified by patients without and with augmented renal clearance.
Data are shown as geometric mean (95% CI).
"""


@pytest.fixture(scope="module")
def md_table_patient_35489632_table_2():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Patients without ARC | N/A | 28 |
| Patients with ARC | N/A | 26 |
| Patients without ARC | N/A | 11 |
| Patients with ARC | N/A | 7 |
"""


@pytest.fixture(scope="module")
def md_table_patient_refined_35489632_table_2():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Adults | N/A | N/A | 28 |
| Adults | N/A | N/A | 26 |
| Adults | N/A | N/A | 11 |
| Adults | N/A | N/A | 7 |
"""


@pytest.fixture(scope="module")
def md_table_aligned_35489632_table_2():
    return """
| Parameter type | Patients without ARC | Patients with ARC | P-value |
| --- | --- | --- | --- |
| Empty Cell | GM (95% CI) | GM (95% CI) | P-value |
| EI | N= 28 | N= 26 |  |
| Cmid(mg/L) | 19.9 (13.5−29.5) | 14.8 (11.4−19.1) | 0.20 |
| Ctrough(mg/L) | 3.5 (2.0−6.1) | 1.6 (1.0−2.6) | 0.04 |
| IB | N= 11 | N= 7 |  |
| Cmid(mg/L) | 4.9 (2.6−9.2) | 1.9 (0.4−9.6) | 0.14 |
| Ctrough(mg/L) | 0.8 (0.4−1.6) | 0.9 (0.2−4.2) | 0.85 |
"""


@pytest.fixture(scope="module")
def md_table_list_35489632_table_2():
    return [
        """
| Parameter type | Patients without ARC | Patients with ARC | P-value |
| --- | --- | --- | --- |
| Empty Cell | GM (95% CI) | GM (95% CI) | P-value |
| EI | N= 28 | N= 26 |  |
| Cmid(mg/L) | 19.9 (13.5−29.5) | 14.8 (11.4−19.1) | 0.20 |
| Ctrough(mg/L) | 3.5 (2.0−6.1) | 1.6 (1.0−2.6) | 0.04 |
| IB | N= 11 | N= 7 |  |
| Cmid(mg/L) | 4.9 (2.6−9.2) | 1.9 (0.4−9.6) | 0.14 |
| Ctrough(mg/L) | 0.8 (0.4−1.6) | 0.9 (0.2−4.2) | 0.85 |
"""
    ]


## =============================================================================
# 22050870_table_3

ghtml_content_22050870_table_3 = """
<section class="tw xbox font-sm" id="T3"><h4 class="obj_head">Table 3.</h4> <div class="caption p"><p id="P41">Bayesian pharmacokinetics parameters (all subjects). CL is clearance. Vdss is volume of distribution at steady state. Beta is the terminal slope of the log concentration versus time profile. T<sub>½</sub> Beta is the elimination half-life.</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <thead><tr> <th align="left" colspan="1" rowspan="1"></th> <th align="center" colspan="1" rowspan="1" valign="top">Free Fraction</th> <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/kg)</th> <th align="center" colspan="1" rowspan="1">CL<br/>mL/min/m<sup>2</sup>)</th> <th align="center" colspan="1" rowspan="1">Vdss<br/>(L/kg)</th> <th align="center" colspan="1" rowspan="1">Beta<br/>(hr<sup>−1</sup>)</th> <th align="center" colspan="1" rowspan="1">T<sub>½</sub> Beta<br/>(hr)</th> </tr></thead> <tbody> <tr> <td align="left" colspan="1" rowspan="1"><strong>Overall</strong></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1">N</td> <td align="center" colspan="1" rowspan="1">61</td> <td align="center" colspan="1" rowspan="1">63</td> <td align="center" colspan="1" rowspan="1">63</td> <td align="center" colspan="1" rowspan="1">63</td> <td align="center" colspan="1" rowspan="1">63</td> <td align="center" colspan="1" rowspan="1">63</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Range</td> <td align="center" colspan="1" rowspan="1">0.07–0.48</td> <td align="center" colspan="1" rowspan="1">0.3–7.75</td> <td align="center" colspan="1" rowspan="1">6.50–147.17</td> <td align="center" colspan="1" rowspan="1">0.49–3.40</td> <td align="center" colspan="1" rowspan="1">0.017–0.118</td> <td align="center" colspan="1" rowspan="1">5.9–42.0</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td> <td align="center" colspan="1" rowspan="1">0.10 ± 0.05</td> <td align="center" colspan="1" rowspan="1">1.2 ± 0.93</td> <td align="center" colspan="1" rowspan="1">33.33 ± 19.33</td> <td align="center" colspan="1" rowspan="1">1.48 ± 0.54</td> <td align="center" colspan="1" rowspan="1">0.048 ± 0.020</td> <td align="center" colspan="1" rowspan="1">16.8 ± 7.1</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Median</td> <td align="center" colspan="1" rowspan="1">0.09</td> <td align="center" colspan="1" rowspan="1">1.08</td> <td align="center" colspan="1" rowspan="1">29.00</td> <td align="center" colspan="1" rowspan="1">1.37</td> <td align="center" colspan="1" rowspan="1">0.046</td> <td align="center" colspan="1" rowspan="1">15.1</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>3 Month to &lt; 3 Years</strong></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1">N</td> <td align="center" colspan="1" rowspan="1">17</td> <td align="center" colspan="1" rowspan="1">18</td> <td align="center" colspan="1" rowspan="1">18</td> <td align="center" colspan="1" rowspan="1">18</td> <td align="center" colspan="1" rowspan="1">18</td> <td align="center" colspan="1" rowspan="1">18</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Range</td> <td align="center" colspan="1" rowspan="1">0.07–0.48</td> <td align="center" colspan="1" rowspan="1">0.63–7.75</td> <td align="center" colspan="1" rowspan="1">12.83–147.17</td> <td align="center" colspan="1" rowspan="1">0.67–3.40</td> <td align="center" colspan="1" rowspan="1">0.024–0.118</td> <td align="center" colspan="1" rowspan="1">5.9–28.4</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td> <td align="center" colspan="1" rowspan="1">0.11 ± 0.10</td> <td align="center" colspan="1" rowspan="1">1.57 ± 1.62</td> <td align="center" colspan="1" rowspan="1">32.83 ± 30.17</td> <td align="center" colspan="1" rowspan="1">1.62 ± 0.59</td> <td align="center" colspan="1" rowspan="1">0.053 ± 0.027</td> <td align="center" colspan="1" rowspan="1">15.8 ± 6.5</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>3 to &lt; 13 Years</strong></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1">N</td> <td align="center" colspan="1" rowspan="1">28</td> <td align="center" colspan="1" rowspan="1">29</td> <td align="center" colspan="1" rowspan="1">29</td> <td align="center" colspan="1" rowspan="1">29</td> <td align="center" colspan="1" rowspan="1">29</td> <td align="center" colspan="1" rowspan="1">29</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Range</td> <td align="center" colspan="1" rowspan="1">0.07–0.17</td> <td align="center" colspan="1" rowspan="1">0.30–1.82</td> <td align="center" colspan="1" rowspan="1">6.50–69.17</td> <td align="center" colspan="1" rowspan="1">0.49–3.00</td> <td align="center" colspan="1" rowspan="1">0.017–0.092</td> <td align="center" colspan="1" rowspan="1">7.5–40.6</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td> <td align="center" colspan="1" rowspan="1">0.10 ± 0.02</td> <td align="center" colspan="1" rowspan="1">1.12 ± 0.40</td> <td align="center" colspan="1" rowspan="1">31.83 ± 13.83</td> <td align="center" colspan="1" rowspan="1">1.50 ± 0.61</td> <td align="center" colspan="1" rowspan="1">0.048 ± 0.017</td> <td align="center" colspan="1" rowspan="1">16.9 ± 7.4</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>13 to &lt; 18 Years</strong></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> <td align="center" colspan="1" rowspan="1"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1">N</td> <td align="center" colspan="1" rowspan="1">16</td> <td align="center" colspan="1" rowspan="1">16</td> <td align="center" colspan="1" rowspan="1">16</td> <td align="center" colspan="1" rowspan="1">16</td> <td align="center" colspan="1" rowspan="1">16</td> <td align="center" colspan="1" rowspan="1">16</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Range</td> <td align="center" colspan="1" rowspan="1">0.07–0.15</td> <td align="center" colspan="1" rowspan="1">0.43–1.58</td> <td align="center" colspan="1" rowspan="1">16.33–60.00</td> <td align="center" colspan="1" rowspan="1">1.00–1.54</td> <td align="center" colspan="1" rowspan="1">0.017–0.084</td> <td align="center" colspan="1" rowspan="1">8.2–42.0</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td> <td align="center" colspan="1" rowspan="1">0.09 ± 0.02</td> <td align="center" colspan="1" rowspan="1">0.95 ± 0.32</td> <td align="center" colspan="1" rowspan="1">36.67 ± 12.00</td> <td align="center" colspan="1" rowspan="1">1.27 ± 0.17</td> <td align="center" colspan="1" rowspan="1">0.044 ± 0.016</td> <td align="center" colspan="1" rowspan="1">17.8 ± 7.7</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/T3/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div></section>
"""


@pytest.fixture(scope="module")
def html_content_22050870_table_3():
    return ghtml_content_22050870_table_3


@pytest.fixture(scope="module")
def caption_22050870_table_3():
    return """
Bayesian pharmacokinetics parameters (all subjects). CL is clearance. Vdss is volume of distribution at steady state. Beta is the terminal slope of the log concentration versus time profile. T½ Beta is the elimination half-life.
"""


@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    level = logging.INFO
    logging.basicConfig(level=level)
    file_handler = logging.FileHandler("./logs/test.log")
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        step_name: Optional[str] = None,
        step_description: Optional[str] = None,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
        token_usage: Optional[dict] = None,
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
