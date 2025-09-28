import os
from pathlib import Path
from typing import Optional
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv
import logging

from TabFuncFlow.utils.table_utils import (
    markdown_to_dataframe,
    single_html_table_to_markdown,
)
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.database.pmid_db import PMIDDB
from extractor.request_sonnet import get_sonnet
from extractor.request_metallama import get_meta_llama
from extractor.request_openai import get_5_openai, get_openai
from system_tests.conftest_data import curated_data_29100749, data_caption_29100749_table_2, data_caption_32635742_table_0, data_col_mapping_29100749_table_2, data_md_table_32635742_table_0, data_md_table_aligned_29100749_table_2, data_md_table_drug_29100749_table_2, data_md_table_list_29100749_table_2, data_source_table_29100749_table_2, paper_abstract_29100749, paper_abstract_32635742, paper_title_29100749, paper_title_32635742

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
        max_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
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

def get_gemini():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL"),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

@pytest.fixture(scope="module")
def llm():
    return get_azure_openai()  # get_openai() # get_deepseek() # get_sonnet() # get_meta_llama() # get_gemini() # get_5_openai() # 


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
def title_29943508():
    return "Effects of Adrenaline on maternal and fetal fentanyl absorption in epidural analgesia: A randomized trial"


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
def md_table_drug_29943508_table_1():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Fentanyl | Fentanyl | Umbilical vein |
| Fentanyl | Fentanyl | Maternal serum |
"""

@pytest.fixture(scope="module")
def md_table_aligned_29943508_table_1():
    return """
| Parameter type | Adrenaline group (n = 19) | Control group (n = 20) | Mean difference | P‐value |
| --- | --- | --- | --- | --- |
| Mean serum fentanyl concentration, umbilical vein (nmol/L) | 0.162 (0.090) (n = 16) | 0.151 (0.070) (n = 20) | 0.012 [−0.042; 0.065] | .67 |
| Median maternal serum fentanyl concentration at birth (nmol/L) | 0.268 [0.193; 0.493]a (n = 16) | 0.291 [0.212; 0.502]a (n = 19) | −0.061 [−0.205; 0.082] | .66a |
| Mean AUC 0‐120 min for fentanyl in maternal serum (nmol h/L) | 0.428 (0.162) (n = 18) | 0.590 (0.197) (n = 15)b | −0.162 [−0.289; −0.034] | .015 |
"""


@pytest.fixture(scope="module")
def col_mapping_29943508():
    return {
        'Parameter type': 'Parameter type', 
        'Adrenaline group (n = 19)': 'Parameter value', 
        'Control group (n = 20)': 'Parameter value', 
        'Mean difference': 'Parameter value', 
        'P‐value': 'P value'
    }
"""
    return {
        "Parameter type": "Parameter type",
        "Adrenaline group (n = 19)": "Parameter value",
        "Control group (n = 20)": "Parameter value",
        "Mean difference": "Parameter value",
        "P‐value": "P value",
    }
"""
@pytest.fixture(scope="module")
def df_combined_29943508():
    return """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Statistics type | Main value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fentanyl | Fentanyl | umbilical vein | Maternal | Delivery | N/A | 16 | Mean serum fentanyl concentration | nmol/L | Mean | 0.162 | SD | 0.090 | N/A | N/A | N/A | .67 |
| Fentanyl | Fentanyl | maternal serum | Maternal | Delivery | N/A | 16 | Median maternal serum fentanyl concentration at birth | nmol/L | Median | 0.268 | N/A | N/A | Range | 0.193 | 0.493 | .66 |
| Fentanyl | Fentanyl | maternal serum | Maternal | Delivery | N/A | 18 | Mean AUC 0‐120 min for fentanyl in maternal serum | nmol h/L | Mean | 0.428 | SD | 0.162 | N/A | N/A | N/A | .015 |
| Fentanyl | Fentanyl | umbilical vein | Maternal | Delivery | N/A | 20 | Mean serum fentanyl concentration | nmol/L | Mean | 0.151 | SD | 0.070 | N/A | N/A | N/A | .67 |
| Fentanyl | Fentanyl | maternal serum | Maternal | Delivery | N/A | 19 | Median maternal serum fentanyl concentration at birth | nmol/L | Median | 0.291 | N/A | N/A | Range | 0.212 | 0.502 | .66 |
| Fentanyl | Fentanyl | maternal serum | Maternal | Delivery | N/A | 15 | Mean AUC 0‐120 min for fentanyl in maternal serum | nmol h/L | Mean | 0.590 | SD | 0.197 | N/A | N/A | N/A | .015 |
"""

## =============================================================================
# 16143486_table_2

ghtml_content_16143486_table_2 = """
<div class="tables frame-topbot rowsep-0 colsep-0" id="tbl2"><span class="captions text-s"><span><p><span class="label">Table 2</span>. Kinetic disposition of lorazepam and its metabolite glucuronide in parturients treated with a single oral dose of 2 mg <em>rac-</em>lorazepam; mean (CI 95%)</p></span></span><div class="groups"><table><thead class="valign-top"><tr><td class="rowsep-1" scope="col"><span class="screen-reader-only">Empty Cell</span></td><th class="rowsep-1 align-left" scope="col">Lorazepam isomeric mixture</th><th class="rowsep-1 align-left" scope="col">Lorazepam-glucuronide isomeric mixture</th></tr></thead><tbody><tr><td class="align-left"><em>C</em><sub>max</sub> (ng/ml)</td><td class="align-left">12.96 (9.42–16.49)</td><td class="align-left">35.55 (8.27–62.83)</td></tr><tr><td class="align-left">t<sub>max</sub> (h)</td><td class="align-left">3.10 (2.57–3.63)</td><td class="align-left">4.33 (2.90–5.77)</td></tr><tr><td class="align-left"><em>t</em><sub>1/2a</sub> (h)</td><td class="align-left">3.16 (2.62–3.68)</td><td class="align-left">1.37 (1.15–1.58)</td></tr><tr><td class="align-left"><em>K</em><sub>a</sub> (h<sup>−1</sup>)</td><td class="align-left">0.23 (0.19–0.28)</td><td class="align-left">0.52 (0.44–0.59)</td></tr><tr><td class="align-left"><em>t</em><sub>1/2</sub><em>β</em> (h)</td><td class="align-left">10.35 (9.39–11.32)</td><td class="align-left">18.17 (14.10–22.23)</td></tr><tr><td class="align-left"><em>β</em> (h<sup>−1</sup>)</td><td class="align-left">0.068 (0.061–0.075)</td><td class="align-left">0.039 (0.032–0.047)</td></tr><tr><td class="align-left">AUC<sup>0–∞</sup> ((ng h)/ml)</td><td class="align-left">175.25 (145.74–204.75)</td><td class="align-left">481.19 (252.87–709.51)</td></tr><tr><td class="align-left">Cl<sub>T</sub>/F (ml/(min kg))</td><td class="align-left">2.61 (2.34–2.88)</td><td class="align-left">–</td></tr><tr><td class="align-left">Vd/F (l)</td><td class="align-left">178.78 (146.46–211.10)</td><td class="align-left">–</td></tr></tbody></table></div><div class="legend"><div class="u-margin-s-bottom">–, Not determined.</div></div></div>
"""

@pytest.fixture(scope="module")
def html_content_16143486_table_2():
    return ghtml_content_16143486_table_2


@pytest.fixture(scope="module")
def caption_16143486_table_2():
    return """
Table 2. Kinetic disposition of lorazepam and its metabolite glucuronide in parturients treated with a single oral dose of 2 mg rac-lorazepam; mean (CI 95%)
–, Not determined.
"""


@pytest.fixture(scope="module")
def md_table_16143486_table_2():
    return single_html_table_to_markdown(ghtml_content_16143486_table_2)


@pytest.fixture(scope="module")
def md_table_drug_16143486_table_2():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Lorazepam | Lorazepam | Plasma |
| Lorazepam | Lorazepam-glucuronide | Plasma |
"""


@pytest.fixture(scope="module")
def md_table_patient_16143486_table_2():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| parturients | N/A | N/A |
"""


@pytest.fixture(scope="module")
def md_table_patient_refined_16143486_table_2():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Maternal | N/A | N/A | N/A |
"""


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
def md_table_aligned_34183327_table_3():
    return """
| Parameter type | Isoniazid | Age, years_0 | Random blood glucose, mg/dL_0 | Drug dose, mg/kg_0 | Drug administration via NGT, no/yes_0 | Rifampicin | Age, years_1 | Random blood glucose, mg/dL_1 | Drug dose, mg/kg_1 | Drug administration via NGT, no/yes_1 | Pyrazinamide | Random blood glucose, mg/dL_2 | Drug dose, mg/kg_2 | Drug administration via NGT, no/yes_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUC0–24,hour∙mg/L(B (95% CI)) |  | n/a | −0.002 (−0.006 to 0.003) | 0.016 (−0.048 to 0.080) | 0.439 (0.143 to 0.735)** |  | −0.009 (−0.028 to 0.010) | −0.003 (−0.007 to 0.001) | 0.014 (−0.021 to 0.048) | n/a |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
| Cmax, mg/L(B (95% CI)) |  | −0.020 (−0.043 to 0.003) | −0.004 (−0.009 to 0.001) | n/a | 0.130 (−0.160 to 0.420) |  | −0.008 (−0.029 to 0.012) | −0.005 (−0.009 to −0.0003)* | n/a | 0.067 (−0.194 to 0.328) |  | −0.003 (−0.005 to −0.001)** | 0.010 (0.001 to 0.020)* | 0.036 (−0.095 to 0.167) |
| CCSF0–8, mg/L(B (95% CI)) |  | n/a | −0.007 (−0.015 to 0.001) | 0.046 (−0.058 to 0.151) | 0.289 (−0.197 to 0.775) |  | −0.021 (−0.052 to 0.009) | n/a | 0.030 (−0.030 to 0.091) | 0.019 (−0.365 to 0.403) |  | −0.006 (−0.010 to −0.003)** | 0.010 (−0.006 to 0.027) | −0.068 (−0.293 to 0.156) |
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
# 22050807_table_1

@pytest.fixture(scope="module")
def title_22050807():
    return """The pharmacokinetics of intravenous lorazepam in pediatric patients with and without status epilepticus"""

@pytest.fixture(scope="module")
def caption_22050807_table_1():
    return """Non-compartmental pharmacokinetics parameters from Elective Cohort patients. Cmax is maximum concentration. AUC0−∞ is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T1/2 is half-life."""

@pytest.fixture(scope="module")
def md_table_drug_22050807_table_1():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Lorazepam | Lorazepam | Plasma |
"""

@pytest.fixture(scope="module")
def md_table_patient_22050807_table_1():
    return """| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Elective Cohort | N/A | 15 |"""

@pytest.fixture(scope="module")
def md_table_patient_refined_22050807_table_1():
    return """| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Elective Cohort | N/A | N/A | 15 |"""

@pytest.fixture(scope="module")
def md_table_summary_22050807_table_1():
    return """
| Unnamed: 0 | Cmax (ng/mL) | AUC0−∞ | CL (mL/min/kg) | CL (mL/min/m2) | Vdz (L/kg) | T1/2 (hr) |
| --- | --- | --- | --- | --- | --- | --- |
| N | 15 | 15 | 15 | 15 | 15 | 15 |
| Range | 29.3–209.6 | 253.3–3202.5 | 3.33–131.50 | 5.5–67.5 | 0.33–4.05 | 9.5–47.0 |
| Mean ± s.d. | 56.1 ± 44.9 | 822.5 ± 706.1 | 49.33 ± 30.83 | 31.95 ± 13.99 | 1.92 ± 0.84 | 20.5 ± 10.2 |
| Median | 42.2 | 601.5 | 41.50 | 32.34 | 1.94 | 18.1 |
"""

@pytest.fixture(scope="module")
def md_table_aligned_22050807_table_1():
    return """
| Parameter type | N | Range | Mean ± s.d. | Median |
| --- | --- | --- | --- | --- |
| Cmax (ng/mL) | 15 | 29.3–209.6 | 56.1 ± 44.9 | 42.2 |
| AUC0−∞ | 15 | 253.3–3202.5 | 822.5 ± 706.1 | 601.5 |
| CL (mL/min/kg) | 15 | 3.33–131.50 | 49.33 ± 30.83 | 41.50 |
| CL (mL/min/m2) | 15 | 5.5–67.5 | 31.95 ± 13.99 | 32.34 |
| Vdz (L/kg) | 15 | 0.33–4.05 | 1.92 ± 0.84 | 1.94 |
| T1/2 (hr) | 15 | 9.5–47.0 | 20.5 ± 10.2 | 18.1 |
"""

## =============================================================================
# 22050870_table_2

ghtml_content_22050870_table_2 = """
<section class="tw xbox font-sm" id="T2"><h4 class="obj_head">Table 2.</h4> <div class="caption p"><p id="P40">Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0−∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life.</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <thead><tr> <th align="left" colspan="1" rowspan="1"></th> <th align="center" colspan="1" rowspan="1">C<sub>max</sub><br/>(ng/mL)</th> <th align="center" colspan="1" rowspan="1" valign="top">AUC<sub>0−∞</sub> </th> <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/kg)</th> <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/m<sup>2</sup>)</th> <th align="center" colspan="1" rowspan="1">Vdz<br/>(L/kg)</th> <th align="center" colspan="1" rowspan="1">T<sub>1/2</sub><br/>(hr)</th> </tr></thead> <tbody> <tr> <td align="left" colspan="1" rowspan="1">N</td> <td align="center" colspan="1" rowspan="1">15</td> <td align="center" colspan="1" rowspan="1">15</td> <td align="center" colspan="1" rowspan="1">15</td> <td align="center" colspan="1" rowspan="1">15</td> <td align="center" colspan="1" rowspan="1">15</td> <td align="center" colspan="1" rowspan="1">15</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top">Range</td> <td align="center" colspan="1" rowspan="1" valign="top">29.3–209.6</td> <td align="center" colspan="1" rowspan="1">253.3–3202.5</td> <td align="center" colspan="1" rowspan="1" valign="top">3.33–131.50</td> <td align="center" colspan="1" rowspan="1" valign="top">5.5–67.5</td> <td align="center" colspan="1" rowspan="1" valign="top">0.33–4.05</td> <td align="center" colspan="1" rowspan="1" valign="top">9.5–47.0</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td> <td align="center" colspan="1" rowspan="1" valign="top">56.1 ± 44.9</td> <td align="center" colspan="1" rowspan="1">822.5 ± 706.1</td> <td align="center" colspan="1" rowspan="1">49.33 ± 30.83</td> <td align="center" colspan="1" rowspan="1" valign="top">31.95 ± 13.99</td> <td align="center" colspan="1" rowspan="1">1.92 ± 0.84</td> <td align="center" colspan="1" rowspan="1">20.5 ± 10.2</td> </tr> <tr> <td align="left" colspan="1" rowspan="1">Median</td> <td align="center" colspan="1" rowspan="1">42.2</td> <td align="center" colspan="1" rowspan="1">601.5</td> <td align="center" colspan="1" rowspan="1">41.50</td> <td align="center" colspan="1" rowspan="1">32.34</td> <td align="center" colspan="1" rowspan="1">1.94</td> <td align="center" colspan="1" rowspan="1">18.1</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/T2/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div></section>
"""

@pytest.fixture(scope="module")
def title_22050870():
    return """
Pharmacokinetics of intravenous lorazepam in pediatric patients with and without status epilepticus    
"""

@pytest.fixture(scope="module")
def caption_22050870_table_2():
    return """
Non-compartmental pharmacokinetics parameters from Elective Cohort patients. Cmax is maximum concentration. AUC0−∞ is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T1/2 is half-life.
"""


@pytest.fixture(scope="module")
def drug_list_22050870_table_2():
    return [
        """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
""",
        """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
""",
        """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
| N/A | N/A | Plasma |
""",
    ]


@pytest.fixture(scope="module")
def md_table_list_22050870_table_2():
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
| T1/2(hr) | 18.1 |
""",
    ]


@pytest.fixture(scope="module")
def patient_list_22050870_table_2():
    return [
        """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
""",
        """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
""",
        """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
| N/A | N/A | N/A | 15 |
""",
    ]


@pytest.fixture(scope="module")
def md_table_aligned_22050870_table_2():
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
def type_unit_list_22050870_table_2():
    return [
        """
| "Parameter type" | "Parameter unit" |
| --- | --- |
| Cmax | ng/mL |
| AUC0−∞ | N/A |
| CL | mL/min/kg |
| CL | mL/min/m |
| Vdz | L/kg |
| T1/2 | hr |
""",
        """
| "Parameter type" | "Parameter unit" |
| --- | --- |
| Cmax | ng/mL |
| AUC0−∞ | N/A |
| CL | mL/min/kg |
| CL | mL/min/m |
| Vdz | L/kg |
| T1/2 | hr |
""",
        """
| "Parameter type" | "Parameter unit" |
| --- | --- |
| Cmax | ng/mL |
| AUC0−∞ | N/A |
| CL | mL/min/kg |
| CL | mL/min/m |
| Vdz | L/kg |
| T1/2 | hr |""",
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

@pytest.fixture(scope="module")
def md_table_22050870_table_3():
    return single_html_table_to_markdown(ghtml_content_22050870_table_3)

@pytest.fixture(scope="module")
def md_table_patient_22050870_table_3():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Overall | N/A | 61 |
| Overall | N/A | 63 |
| 3 Month to < 3 Years | N/A | 17 |
| 3 Month to < 3 Years | N/A | 18 |
| 3 to < 13 Years | N/A | 28 |
| 3 to < 13 Years | N/A | 29 |
| 13 to < 18 Years | N/A | 16 |
"""

@pytest.fixture(scope="module")
def md_table_patient_refined_22050870_table_3():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | Overall | 61 |
| N/A | N/A | Overall | 63 |
| Children | N/A | 3 Month to < 3 Years | 17 |
| Children | N/A | 3 Month to < 3 Years | 18 |
| Children | N/A | 3 to < 13 Years | 28 |
| Children | N/A | 3 to < 13 Years | 29 |
| Adolescents | N/A | 13 to < 18 Years | 16 |
"""

@pytest.fixture(scope="module")
def md_table_aligned_22050870_table_3():
    return """
| Parameter type | Overall | N_0 | Range_0 | Mean ± s.d._0 | Median | 3 Month to < 3 Years | N_1 | Range_1 | Mean ± s.d._1 | 3 to < 13 Years | N_2 | Range_2 | Mean ± s.d._2 | 13 to < 18 Years | N_3 | Range_3 | Mean ± s.d._3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Free Fraction | nan | 61 | 0.07–0.48 | 0.10 ± 0.05 | 0.09 | nan | 17 | 0.07–0.48 | 0.11 ± 0.10 | nan | 28 | 0.07–0.17 | 0.10 ± 0.02 | nan | 16 | 0.07–0.15 | 0.09 ± 0.02 |
| CL (mL/min/kg) | nan | 63 | 0.3–7.75 | 1.2 ± 0.93 | 1.08 | nan | 18 | 0.63–7.75 | 1.57 ± 1.62 | nan | 29 | 0.30–1.82 | 1.12 ± 0.40 | nan | 16 | 0.43–1.58 | 0.95 ± 0.32 |
| CL mL/min/m2) | nan | 63 | 6.50–147.17 | 33.33 ± 19.33 | 29.00 | nan | 18 | 12.83–147.17 | 32.83 ± 30.17 | nan | 29 | 6.50–69.17 | 31.83 ± 13.83 | nan | 16 | 16.33–60.00 | 36.67 ± 12.00 |
| Vdss (L/kg) | nan | 63 | 0.49–3.40 | 1.48 ± 0.54 | 1.37 | nan | 18 | 0.67–3.40 | 1.62 ± 0.59 | nan | 29 | 0.49–3.00 | 1.50 ± 0.61 | nan | 16 | 1.00–1.54 | 1.27 ± 0.17 |
| Beta (hr−1) | nan | 63 | 0.017–0.118 | 0.048 ± 0.020 | 0.046 | nan | 18 | 0.024–0.118 | 0.053 ± 0.027 | nan | 29 | 0.017–0.092 | 0.048 ± 0.017 | nan | 16 | 0.017–0.084 | 0.044 ± 0.016 |
| T½ Beta (hr) | nan | 63 | 5.9–42.0 | 16.8 ± 7.1 | 15.1 | nan | 18 | 5.9–28.4 | 15.8 ± 6.5 | nan | 29 | 7.5–40.6 | 16.9 ± 7.4 | nan | 16 | 8.2–42.0 | 17.8 ± 7.7 |
"""

@pytest.fixture(scope="module")
def md_table_list_22050870_table_3():
    return ["""
| Parameter type | Range_0 |
| --- | --- |
| Free Fraction | 0.07–0.48 |
| CL (mL/min/kg) | 0.3–7.75 |
| CL mL/min/m2) | 6.50–147.17 |
| Vdss (L/kg) | 0.49–3.40 |
| Beta (hr−1) | 0.017–0.118 |
| T½ Beta (hr) | 5.9–42.0 |""", """
| Parameter type | Mean ± s.d._0 |
| --- | --- |
| Free Fraction | 0.10 ± 0.05 |
| CL (mL/min/kg) | 1.2 ± 0.93 |
| CL mL/min/m2) | 33.33 ± 19.33 |
| Vdss (L/kg) | 1.48 ± 0.54 |
| Beta (hr−1) | 0.048 ± 0.020 |
| T½ Beta (hr) | 16.8 ± 7.1 |""", """
| Parameter type | Median |
| --- | --- |
| Free Fraction | 0.09 |
| CL (mL/min/kg) | 1.08 |
| CL mL/min/m2) | 29.00 |
| Vdss (L/kg) | 1.37 |
| Beta (hr−1) | 0.046 |
| T½ Beta (hr) | 15.1 |""", """
| Parameter type | Range_1 |
| --- | --- |
| Free Fraction | 0.07–0.48 |
| CL (mL/min/kg) | 0.63–7.75 |
| CL mL/min/m2) | 12.83–147.17 |
| Vdss (L/kg) | 0.67–3.40 |
| Beta (hr−1) | 0.024–0.118 |
| T½ Beta (hr) | 5.9–28.4 |""", """
| Parameter type | Mean ± s.d._1 |
| --- | --- |
| Free Fraction | 0.11 ± 0.10 |
| CL (mL/min/kg) | 1.57 ± 1.62 |
| CL mL/min/m2) | 32.83 ± 30.17 |
| Vdss (L/kg) | 1.62 ± 0.59 |
| Beta (hr−1) | 0.053 ± 0.027 |
| T½ Beta (hr) | 15.8 ± 6.5 |""", """
| Parameter type | Range_2 |
| --- | --- |
| Free Fraction | 0.07–0.17 |
| CL (mL/min/kg) | 0.30–1.82 |
| CL mL/min/m2) | 6.50–69.17 |
| Vdss (L/kg) | 0.49–3.00 |
| Beta (hr−1) | 0.017–0.092 |
| T½ Beta (hr) | 7.5–40.6 |""", """
| Parameter type | Mean ± s.d._2 |
| --- | --- |
| Free Fraction | 0.10 ± 0.02 |
| CL (mL/min/kg) | 1.12 ± 0.40 |
| CL mL/min/m2) | 31.83 ± 13.83 |
| Vdss (L/kg) | 1.50 ± 0.61 |
| Beta (hr−1) | 0.048 ± 0.017 |
| T½ Beta (hr) | 16.9 ± 7.4 |""", """
| Parameter type | Range_3 |
| --- | --- |
| Free Fraction | 0.07–0.15 |
| CL (mL/min/kg) | 0.43–1.58 |
| CL mL/min/m2) | 16.33–60.00 |
| Vdss (L/kg) | 1.00–1.54 |
| Beta (hr−1) | 0.017–0.084 |
| T½ Beta (hr) | 8.2–42.0 |""", """
| Parameter type | Mean ± s.d._3 |
| --- | --- |
| Free Fraction | 0.09 ± 0.02 |
| CL (mL/min/kg) | 0.95 ± 0.32 |
| CL mL/min/m2) | 36.67 ± 12.00 |
| Vdss (L/kg) | 1.27 ± 0.17 |
| Beta (hr−1) | 0.044 ± 0.016 |
| T½ Beta (hr) | 17.8 ± 7.7 |"""]

# ==========================================================================================
# 28794838_table_2


@pytest.fixture(scope="module")
def caption_28794838_table_2():
    return """
Values are mean ± SD (range), median [interquartile range] (range) or number (%). Mechanical ventilation: respiratory management with mechanical ventilation when obtaining sample. *One sedative and one analgesic drug, †≥ 3 sedative and analgesic drugs. RASS: Richmond Agitation-Sedation Scale.
"""


@pytest.fixture(scope="module")
def md_table_drug_28794838_table_2():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Dexmedetomidine | Dexmedetomidine | Plasma |
| Fentanyl | Fentanyl | Plasma |
| Midazolam | Midazolam | Plasma |
"""


@pytest.fixture(scope="module")
def md_table_aligned_28794838_table_2():
    return """
| In 96 samples of 27 infants | Parameter type |Ks
| --- | --- |
| Drug treatment | nan |
| Duration of infusion (h) | 60 [12, 108] (6–276) |
| Plasma concentrations (ng/ml) | 0.86 ± 0.65 (0.07–4.68) |
| Dosages (µg/kg/h) | 0.63 [0.40–0.71] (0.12–1.40) |
| Combined administration | nan |
| No drug (only dexmedetomidine) | 11 (11.5%) |
| 1 drug (with fentanyl or morphine) | 22 (22.9%) |
| 2 drugs* | 49 (51.0%) |
| 3 or more drugs† | 14 (14.6%) |
| Fentanyl | 71 (74.0%) |
| Midazolam | 51 (57.3%) |
| Management with artificial ventilation | 74 (77.1%) |
| RASS | nan |
| ≥ 1 | 0 (0%) |
| 0 | 10 (10.4%) |
| −1 | 12 (12.5%) |
| −2 | 45 (46.9%) |
| −3 | 20 (20.8%) |
| −4 | 6 (6.3%) |
| −5 | 3 (3.1%) |
"""


@pytest.fixture(scope="module")
def col_mapping_28794838_table_2():
    return {
        "In 96 samples of 27 infants": "Uncategorized",
        "Parameter type": "Parameter type",
    }

# ============================================================================================
# 34114632 table 2

@pytest.fixture(scope="module")
def title_34114632():
    return "The Pharmacokinetics of Crushed Levetiracetam Tablets Administered to Neonates\nFree"

@pytest.fixture(scope="module")
def caption_34114632_table_2():
    return """
Open in new tab
Table 2\u2002Pharmacokinetic parameters of levetiracetam in our study compared with historical data
Cmax, maximum plasma concentration; Tmax, time to maximum concentration; AUC0–12, area under the curve from time 0 to 12\u2009h; Ctrough, trough plasma concentration.aCompared with historical data by Fountain, et al. [18].bCompared with historical data.cT-test used to assess data.dToo small sample to compare to historical data.eMedian (range); all other data expressed as mean (±standard deviation).fNon-normally distributed data.
"""

@pytest.fixture(scope="module")
def md_table_34114632_table_2():
    return """
| Unnamed: 0_level_0/PK parameters | Dose range 1 (5–15 mg/kg/12 h)/Historical dataa | Dose range 1 (5–15 mg/kg/12 h)/Current data | Dose range 1 (5–15 mg/kg/12 h)/p valueb,c | Dose range 2 (15–25 mg/kg/12 h)/Historical dataa | Dose range 2 (15–25 mg/kg/12 h)/Current data | Dose range 2 (15–25 mg/kg/12 h)/p valueb,c | Dose range 3 (25–35 mg/kg/12 h)/Historical dataa | Dose range 3 (25–35 mg/kg/12 h)/Current data | Dose range 3 (25–35 mg/kg/12 h)/p valued |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cmax (µg/mL) | 24.8\u2009±\u20098.3 | 19.19\u2009±\u20094.12 | 0.07 | 57.1\u2009±\u200914.9 | 35.12\u2009±\u200910.54 | 0.001 | 73.2\u2009±\u200919.2 | 36.11 (27.58–44.64)e | – |
| AUC0–12 (h*µg\u2009/mL) | 145\u2009±\u200944 | 167.0\u2009±\u200945.6 | 0.26 | 322\u2009±\u200971 | 316.5\u2009±\u2009108.4 | 0.88 | 433\u2009±\u200994 | 290.9 (176.14–405.59)e | – |
| Ctrough (µg/mL) | 8.4\u2009±\u20093.8 | 9.99\u2009±\u20093.86 | 0.34 | 15.6\u2009±\u20095.3 | 19.25\u2009±\u20098.48 | 0.22 | 20.6\u2009±\u20095.8 | 13.03 (2.98–23.07)e | – |
| Tmax (h) | 0.5 (0.25–3.0)e | 1.5 (1.5–2.5)e,f | – | 0.5 (0.5–3.0)e | 2.5 (2.0–3.3)e,f | – | 0.5 (0.5–3)e | 1.5 (1.5)e,f | – |
"""

@pytest.fixture(scope="module")
def md_table_drug_34114632_table_2():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Levetiracetam | Levetiracetam | Plasma |
"""





# ============================================================================================
# 34114632 table 3


@pytest.fixture(scope="module")
def html_content_34114632_table_3():
    return """
<div class="table-wrap table-wide standard-table"><div class="table-wrap-title" data-id="fmab041-T3" id="fmab041-T3"><div class="graphic-wrap table-open-button-wrap"><a aria-describedby="label-1661" class="fig-view-orig at-tableViewLarge openInAnotherWindow btn js-view-large" data-google-interstitial="false" href=" /view-large/263706192" role="button" target="_blank"> Open in new tab </a></div><div class="caption caption-id-" id="caption-1661"><p class="chapter-para"><strong><span class="small-caps">Table</span> 3</strong> <em>Comparison of pharmacokinetic parameters between participants receiving levetiracetam via naso- or orogastric tube vs. participants receiving the drug orally</em></p></div> </div><div class="table-overflow"><table aria-describedby=" caption-1661" role="table"><thead><tr><th>Pharmacokinetic parameter<span aria-hidden="true" style="display: none;"> . </span></th><th>Naso- or orogastric tube administration, <em>n</em> = 14/19<span aria-hidden="true" style="display: none;"> . </span></th><th>Oral administration, <em>n</em> = 5/19<span aria-hidden="true" style="display: none;"> . </span></th><th><em>P</em> value<span class="xrefLink" id="jumplink-tblfn12"></span><a class="link link-ref link-reveal xref-fn js-xref-fn" data-google-interstitial="false" data-open="tblfn12" href="javascript:;" reveal-id="tblfn12"><sup>a</sup></a><span aria-hidden="true" style="display: none;"> . </span></th></tr></thead><tbody><tr><td>AUC<sub>0–12</sub> (h*μg/mL)</td><td>220 (157.5–355.4)</td><td>213.8 (154.0–348.8)</td><td>0.90</td></tr><tr><td><em>C</em><sub>max</sub> (μg/mL)</td><td>23.8 (18.8–41.3)</td><td>26.4 (19.3–34.2)</td><td>0.84</td></tr><tr><td><em>T</em><sub>max</sub> (h)</td><td>1.5 (1.5–2.5)</td><td>2.5 (1.5–2.9)</td><td>0.46</td></tr></tbody></table></div><div class="table-modal"><table><thead><tr><th>Pharmacokinetic parameter<span aria-hidden="true" style="display: none;"> . </span></th><th>Naso- or orogastric tube administration, <em>n</em> = 14/19<span aria-hidden="true" style="display: none;"> . </span></th><th>Oral administration, <em>n</em> = 5/19<span aria-hidden="true" style="display: none;"> . </span></th><th><em>P</em> value<span class="xrefLink" id="jumplink-tblfn12"></span><a class="link link-ref link-reveal xref-fn js-xref-fn" data-google-interstitial="false" data-open="tblfn12" href="javascript:;" reveal-id="tblfn12"><sup>a</sup></a><span aria-hidden="true" style="display: none;"> . </span></th></tr></thead><tbody><tr><td>AUC<sub>0–12</sub> (h*μg/mL)</td><td>220 (157.5–355.4)</td><td>213.8 (154.0–348.8)</td><td>0.90</td></tr><tr><td><em>C</em><sub>max</sub> (μg/mL)</td><td>23.8 (18.8–41.3)</td><td>26.4 (19.3–34.2)</td><td>0.84</td></tr><tr><td><em>T</em><sub>max</sub> (h)</td><td>1.5 (1.5–2.5)</td><td>2.5 (1.5–2.9)</td><td>0.46</td></tr></tbody></table></div><div class="table-wrap-foot"><span id="fn-tblfn11"></span><div class="footnote" content-id="tblfn11"><span class="fn"><p class="chapter-para">All values expressed as medians and interquartile ranges. Maximum plasma concentration (<em>C</em><sub>max</sub>), time to maximum plasma concentration (<em>T</em><sub>max</sub>) and area under the curve 0–12 h (AUC<sub>0–12</sub>).</p></span></div><span id="fn-tblfn12"></span><div class="footnote" content-id="tblfn12"><span class="fn"><span class="label fn-label"><span class="end-note-link" data-fn-id="tblfn12" rel="nofollow">a</span></span><p class="chapter-para">The study was not powered to specifically compare naso- or orogastric administration vs. oral administration.</p></span></div></div></div>
"""


@pytest.fixture(scope="module")
def caption_34114632_table_3():
    return """
Open in new tab Table 3 Comparison of pharmacokinetic parameters between participants receiving levetiracetam via naso- or orogastric tube vs. participants receiving the drug orally
All values expressed as medians and interquartile ranges. Maximum plasma concentration (Cmax), time to maximum plasma concentration (Tmax) and area under the curve 0–12 h (AUC0–12).aThe study was not powered to specifically compare naso- or orogastric administration vs. oral administration.
"""


@pytest.fixture(scope="module")
def md_table_drug_34114632_table_3():
    return """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| Levetiracetam | Levetiracetam | Plasma |
"""


@pytest.fixture(scope="module")
def md_table_patient_34114632_table_3():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| N/A | N/A | "14" |
| N/A | N/A | "5" |
| N/A | N/A | "19" |
"""


@pytest.fixture(scope="module")
def md_table_34114632_table_3():
    md_table = single_html_table_to_markdown(html_content_34114632_table_3)
    return md_table

@pytest.fixture(scope="module")
def md_table_aligned_34114632_table_3():
    return """| Parameter type | Naso- or orogastric tube administration, n = 14/19 | Oral administration, n = 5/19 | P valuea |
| --- | --- | --- | --- |
| AUC0–12 (h*μg/mL) | 220 (157.5–355.4) | 213.8 (154.0–348.8) | 0.9 |
| Cmax (μg/mL) | 23.8 (18.8–41.3) | 26.4 (19.3–34.2) | 0.84 |
| Tmax (h) | 1.5 (1.5–2.5) | 2.5 (1.5–2.9) | 0.46 |
"""

@pytest.fixture(scope="module")
def col_mapping_34114632_table_3():
    return {
        'Parameter type': 'Parameter type', 
        'Naso- or orogastric tube administration, n = 14/19': 'Parameter value', 
        'Oral administration, n = 5/19': 'Parameter value', 
        'P valuea': 'P value'
    }



## =============================================================================
# 17635501_table_3

ghtml_content_17635501_table_3 = """
<section class="tw xbox font-sm" id="tbl3"><h4 class="obj_head">Table 3.</h4> <div class="caption p"><p>Pharmacokinetic parameters of lorazepam (LZP) following administration of a single dose (0.1 mg kg<sup>−1</sup>) either intravenously (i.v.) or intramuscularly (i.m.) in children with severe malaria and convulsions</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <thead><tr> <th align="left" colspan="1" rowspan="1">Parameter</th> <th align="left" colspan="1" rowspan="1">n</th> <th align="left" colspan="1" rowspan="1">I.v. LZP</th> <th align="left" colspan="1" rowspan="1">n</th> <th align="left" colspan="1" rowspan="1">I.m. LZP</th> <th align="left" colspan="1" rowspan="1">95% CI for the difference between the means or medians</th> </tr></thead> <tbody> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>C</em><sub>max</sub> (ng ml<sup>−1</sup>)</strong></td> <td align="right" colspan="1" rowspan="1">11</td> <td align="right" colspan="1" rowspan="1">65.1 (47.5, 86)</td> <td align="right" colspan="1" rowspan="1">10</td> <td align="right" colspan="1" rowspan="1">45.3 (29.6, 66.3)</td> <td align="left" colspan="1" rowspan="1">−43.5, 5.0</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>t</em><sub>max</sub> (h)*</strong></td> <td align="right" colspan="1" rowspan="1">11</td> <td align="right" colspan="1" rowspan="1">0. 5 (0.167–0.67)</td> <td align="right" colspan="1" rowspan="1">10</td> <td align="right" colspan="1" rowspan="1">0.42 (0.167–1.0)</td> <td align="left" colspan="1" rowspan="1">−0.33, 0.17</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>t</em><sub>1/2</sub> (elimination), h</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">23.7 (9.8, 37.6)</td> <td align="right" colspan="1" rowspan="1">5</td> <td align="right" colspan="1" rowspan="1">36.9 (−1.5, 75.5)</td> <td align="left" colspan="1" rowspan="1">−41.3, 14.9</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>AUC<sub>0–∞</sub> (ng ml<sup>−1</sup> h<sup>−1</sup>)</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">2062.5 (600.6, 3771.4)</td> <td align="right" colspan="1" rowspan="1">5</td> <td align="right" colspan="1" rowspan="1">1843.6 (296.7, 3390.5)</td> <td align="left" colspan="1" rowspan="1">−1267.8, 1883.0</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>k</em><sub><em>a</em></sub> (h<sup>−1</sup>)*</strong></td> <td colspan="1" rowspan="1"></td> <td align="right" colspan="1" rowspan="1">–</td> <td align="right" colspan="1" rowspan="1">6</td> <td align="right" colspan="1" rowspan="1">9.8 (0.033, 22.8)</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>t</em><sub>1/2</sub> (absorption), h*</strong></td> <td colspan="1" rowspan="1"></td> <td align="right" colspan="1" rowspan="1">–</td> <td align="right" colspan="1" rowspan="1">6</td> <td align="right" colspan="1" rowspan="1">0.035 (0.01, 0.071)</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>CL (l h<sup>−1</sup>)</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">0.64 (0.36, 0.92)</td> <td colspan="1" rowspan="1"></td> <td align="right" colspan="1" rowspan="1">–</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>V</em><sub>C</sub> (l kg<sup>−1</sup>)</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">1.67 (1.25, 2.10)</td> <td colspan="1" rowspan="1"></td> <td align="right" colspan="1" rowspan="1">–</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong><em>V</em><sub>ss</sub> (l kg<sup>−1</sup>)</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">2.59 (1.56, 3.62)</td> <td colspan="1" rowspan="1"></td> <td align="right" colspan="1" rowspan="1">–</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> <tr> <td align="left" colspan="1" rowspan="1"><strong>Bioavailability (<em>F</em>)</strong></td> <td align="right" colspan="1" rowspan="1">9</td> <td align="right" colspan="1" rowspan="1">Assume 100%</td> <td align="right" colspan="1" rowspan="1">6</td> <td align="right" colspan="1" rowspan="1">89.4%</td> <td align="left" colspan="1" rowspan="1">–</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/tbl3/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div> <div class="tw-foot p"><div class="fn" id="fn3"><p>Values are presented as mean (95% CI) or median (range) *.</p></div></div></section>
"""


@pytest.fixture(scope="module")
def html_content_17635501table_3():
    return ghtml_content_17635501_table_3

@pytest.fixture(scope="module")
def paper_title_17635501():
    return "Pharmacokinetics and clinical efficacy of lorazepam in children with severe malaria and convulsions"

@pytest.fixture(scope="module")
def paper_abstract_17635501():
    return "Pharmacokinetic parameters of lorazepam (LZP) following administration of a single dose (0.1 mg kg−1) either intravenously (i.v.) or intramuscularly (i.m.) in children with severe malaria and convulsions"


@pytest.fixture(scope="module")
def caption_17635501_table_3():
    return """
Pharmacokinetic parameters of lorazepam (LZP) following administration of a single dose (0.1 mg kg−1) either intravenously (i.v.) or intramuscularly (i.m.) in children with severe malaria and convulsions
Values are presented as mean (95% CI) or median (range) *.
"""

@pytest.fixture(scope="module")
def md_table_17635501_table_3():
    return single_html_table_to_markdown(ghtml_content_17635501_table_3)

@pytest.fixture(scope="module")
def md_table_patient_17635501_table_3():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Children | N/A | 11.0 |
| Children | N/A | 10.0 |
| Children | N/A | 9.0 |
| Children | N/A | 5.0 |
| Children | N/A | 6.0 |
"""

@pytest.fixture(scope="module")
def md_table_patient_refined_17635501_table_3():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Children | N/A | N/A | 11.0 |
| Children | N/A | N/A | 10.0 |
| Children | N/A | N/A | 9.0 |
| Children | N/A | N/A | 5.0 |
| Children | N/A | N/A | 6.0 |
"""

@pytest.fixture(scope="module")
def md_table_aligned_17635501_table_3():
    return """
| Parameter type | n | I.v. LZP | n.1 | I.m. LZP | 95% CI for the difference between the means or medians |
| --- | --- | --- | --- | --- | --- |
| Cmax (ng ml−1) | 11.0 | 65.1 (47.5, 86) | 10.0 | 45.3 (29.6, 66.3) | −43.5, 5.0 |
| tmax (h)* | 11.0 | 0. 5 (0.167–0.67) | 10.0 | 0.42 (0.167–1.0) | −0.33, 0.17 |
| t1/2 (elimination), h | 9.0 | 23.7 (9.8, 37.6) | 5.0 | 36.9 (−1.5, 75.5) | −41.3, 14.9 |
| AUC0–∞ (ng ml−1 h−1) | 9.0 | 2062.5 (600.6, 3771.4) | 5.0 | 1843.6 (296.7, 3390.5) | −1267.8, 1883.0 |
| ka (h−1)* | nan | – | 6.0 | 9.8 (0.033, 22.8) | – |
| t1/2 (absorption), h* | nan | – | 6.0 | 0.035 (0.01, 0.071) | – |
| CL (l h−1) | 9.0 | 0.64 (0.36, 0.92) | nan | – | – |
| VC (l kg−1) | 9.0 | 1.67 (1.25, 2.10) | nan | – | – |
| Vss (l kg−1) | 9.0 | 2.59 (1.56, 3.62) | nan | – | – |
| Bioavailability (F) | 9.0 | Assume 100% | 6.0 | 89.4% | – |
"""

@pytest.fixture(scope="module")
def md_table_list_17635501_table_3():
    return ["""
| Parameter type | I.v. LZP |
| --- | --- |
| Cmax (ng ml−1) | 65.1 (47.5, 86) |
| tmax (h)* | 0. 5 (0.167–0.67) |
| t1/2 (elimination), h | 23.7 (9.8, 37.6) |
| AUC0–∞ (ng ml−1 h−1) | 2062.5 (600.6, 3771.4) |
| ka (h−1)* | – |
| t1/2 (absorption), h* | – |
| CL (l h−1) | 0.64 (0.36, 0.92) |
| VC (l kg−1) | 1.67 (1.25, 2.10) |
| Vss (l kg−1) | 2.59 (1.56, 3.62) |
| Bioavailability (F) | Assume 100% |
""", """
| Parameter type | I.m. LZP |
| --- | --- |
| Cmax (ng ml−1) | 45.3 (29.6, 66.3) |
| tmax (h)* | 0.42 (0.167–1.0) |
| t1/2 (elimination), h | 36.9 (−1.5, 75.5) |
| AUC0–∞ (ng ml−1 h−1) | 1843.6 (296.7, 3390.5) |
| ka (h−1)* | 9.8 (0.033, 22.8) |
| t1/2 (absorption), h* | 0.035 (0.01, 0.071) |
| CL (l h−1) | – |
| VC (l kg−1) | – |
| Vss (l kg−1) | – |
| Bioavailability (F) | 89.4% |
""", """
| Parameter type | 95% CI for the difference between the means or medians |
| --- | --- |
| Cmax (ng ml−1) | −43.5, 5.0 |
| tmax (h)* | −0.33, 0.17 |
| t1/2 (elimination), h | −41.3, 14.9 |
| AUC0–∞ (ng ml−1 h−1) | −1267.8, 1883.0 |
| ka (h−1)* | – |
| t1/2 (absorption), h* | – |
| CL (l h−1) | – |
| VC (l kg−1) | – |
| Vss (l kg−1) | – |
| Bioavailability (F) | – |
"""]

@pytest.fixture(scope="module")
def curated_table_17635501_table_3():
    # From meta llama curated data
    return """
,Drug name,Analyte,Specimen,Population,Pregnancy stage,Pediatric/Gestational age,Subject N,Parameter type,Parameter unit,Parameter statistic,Parameter value,Variation type,Variation value,Interval type,Lower bound,Upper bound,P value,Time value,Time unit
0,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,11,Cmax,ng ml−1,Mean,65.1,N/A,N/A,Range,47.5,86,N/A,N/A,N/A
1,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,11,tmax,h,Median,0.5,N/A,N/A,Range,0.167,0.67,N/A,N/A,N/A
2,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,t1/2 (elimination),h,Mean,23.7,N/A,N/A,Range,9.8,37.6,N/A,N/A,N/A
3,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,AUC0–∞,ng ml−1 h−1,Mean,2062.5,N/A,N/A,Range,600.6,3771.4,N/A,N/A,N/A
4,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,CL,l h−1,Mean,0.64,N/A,N/A,Range,0.36,0.92,N/A,N/A,N/A
5,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,VC,l kg−1,Mean,1.67,N/A,N/A,Range,1.25,2.10,N/A,N/A,N/A
6,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,Vss,l kg−1,Mean,2.59,N/A,N/A,Range,1.56,3.62,N/A,N/A,N/A
7,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,9,Bioavailability,N/A,N/A,100%,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A
8,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,10,Cmax,ng ml−1,Mean,45.3,N/A,N/A,Range,29.6,66.3,N/A,N/A,N/A
9,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,10,tmax,h,Median,0.42,N/A,N/A,Range,0.167,1.0,N/A,N/A,N/A
10,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,5,t1/2 (elimination),h,Mean,36.9,N/A,N/A,Range,-1.5,75.5,N/A,N/A,N/A
11,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,5,AUC0–∞,ng ml−1 h−1,Mean,1843.6,N/A,N/A,Range,296.7,3390.5,N/A,N/A,N/A
12,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,6,ka,h−1,Median,9.8,N/A,N/A,Range,0.033,22.8,N/A,N/A,N/A
13,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,6,t1/2 (absorption),h,Median,0.035,N/A,N/A,Range,0.01,0.071,N/A,N/A,N/A
14,Lorazepam,Lorazepam,Plasma,children,N/A,N/A,6,Bioavailability,N/A,N/A,89.4,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A

"""

@pytest.fixture(scope="module")
def md_table_combined_17635501_table_3():
    return """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Main value | Statistics type | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 11 | Maximum Concentration (Cmax) | ng/ml | 65.1 | Mean | N/A | N/A | 95% CI | 47.5 | 86 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 11 | Time to Reach Maximum Concentration (tmax) | h | 0.5 | Median | N/A | N/A | Range | 0.167 | 0.67 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Elimination Half-Life (t1/2) | h | 23.7 | Mean | N/A | N/A | 95% CI | 9.8 | 37.6 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Area Under Concentration-Time Curve (AUC0–∞) | ng·h/ml | 2062.5 | Mean | N/A | N/A | 95% CI | 600.6 | 3771.4 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 6 | Absorption Rate Constant (ka) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 6 | Absorption Half-Life (t1/2 absorption) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Clearance (CL) | l/h | 0.64 | Mean | N/A | N/A | 95% CI | 0.36 | 0.92 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Volume of Central Compartment (VC) | l/kg | 1.67 | Mean | N/A | N/A | 95% CI | 1.25 | 2.10 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Volume at Steady-State (Vss) | l/kg | 2.59 | Mean | N/A | N/A | 95% CI | 1.56 | 3.62 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Bioavailability | Relative (%) | 100 | Assume | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 10 | Maximum Concentration (Cmax) | ng/ml | 45.3 | Mean | N/A | N/A | Range | 29.6 | 66.3 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 10 | Time to Reach Maximum Concentration (tmax) | h | 0.42 | Mean | N/A | N/A | Range | 0.167 | 1.0 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 5 | Elimination Half-Life (t1/2) | h | 36.9 | Mean | N/A | N/A | Range | -1.5 | 75.5 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 5 | Area Under Concentration-Time Curve (AUC0–∞) | ng·h/ml | 1843.6 | Mean | N/A | N/A | Range | 296.7 | 3390.5 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 6 | Absorption Rate Constant (ka) | N/A | 9.8 | Mean | N/A | N/A | Range | 0.033 | 22.8 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 6 | Absorption Half-Life (t1/2 absorption) | N/A | 0.035 | Mean | N/A | N/A | Range | 0.01 | 0.071 | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Clearance (CL) | l/h | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Volume of Central Compartment (VC) | l/kg | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Volume at Steady-State (Vss) | l/kg | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 6 | Bioavailability | Relative (%) | 89.4 | Mean | N/A | N/A | N/A | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 11 | Maximum Concentration (Cmax) | ng/ml | 65.1 | Mean | 95% CI | 43.25 | -43.5, 5.0 | -43.5 | 5.0 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 11 | Time to Reach Maximum Concentration (tmax) | h | 0.42 | Mean | 95% CI | 0.167 | -0.33, 0.17 | -0.33 | 0.17 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Elimination Half-Life (t1/2) | h | 23.7 | Median | 95% CI | 9.8 | -41.3, 14.9 | -41.3 | 14.9 | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Area Under Concentration-Time Curve (AUC0–∞) | ng·h/ml | 2062.5 | Mean | 95% CI | 600.6 | -1267.8, 1883.0 | -1267.8 | 1883.0 | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Absorption Rate Constant (ka) | N/A | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Absorption Half-Life (t1/2 absorption) | N/A | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Clearance (CL) | l/h | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Volume of Central Compartment (VC) | l/kg | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | Children | N/A | N/A | 9 | Volume at Steady-State (Vss) | l/kg | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
| Lorazepam | Lorazepam | Plasma | ERROR | ERROR | ERROR | ERROR | Bioavailability | Relative (%) | N/A | N/A | N/A | N/A | - | N/A | N/A | N/A |
"""

@pytest.fixture(scope="module")
def df_combined_17635501_table_3(md_table_combined_17635501_table_3):
    return markdown_to_dataframe(md_table_combined_17635501_table_3)



## =============================================================================
# 34183327_table_3

ghtml_content_34183327_table_2 = """
<section class="tw xbox font-sm" id="T2"><h3 class="obj_head">Table 2.</h3> <div class="caption p"><p>Summary of pharmacokinetic (PK) parameters of isoniazid, rifampicin and pyrazinamide among Indonesian children treated for TBM</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <thead><tr> <td align="left" colspan="1" rowspan="1" valign="bottom">PK parameters</td> <td align="left" colspan="1" rowspan="1" valign="bottom">First PK assessment (n=20)</td> <td align="left" colspan="1" rowspan="1" valign="bottom">Second PK assessment (n=12)</td> <td align="left" colspan="1" rowspan="1" valign="bottom">P value*</td> </tr></thead> <tbody> <tr> <td align="left" colspan="1" rowspan="1" valign="top">Isoniazid</td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> AUC<sub>0–24</sub> (h∙mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">18.5 (5.1–47.4)</td> <td align="center" colspan="1" rowspan="1" valign="top">14.5 (5.9–44.2)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.888</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>max</sub> (mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">4.6 (1.0–10.0)</td> <td align="center" colspan="1" rowspan="1" valign="top">4.7 (2.5–13.6)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.366</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF0–2</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">1.4 (0.5–6.1)</td> <td align="center" colspan="1" rowspan="1" valign="top">1.6 (1.2–2.5)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF3–5</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">1.6 (0.3–5.0)</td> <td align="center" colspan="1" rowspan="1" valign="top">1.7 (0.6–5.0)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF6–8</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">1.3 (1.2–4.3)</td> <td align="center" colspan="1" rowspan="1" valign="top">2.3 (1.9–2.8)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top">Rifampicin</td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> AUC<sub>0–24</sub> (h∙mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">66.9 (21.7–118.6)</td> <td align="center" colspan="1" rowspan="1" valign="top">71.8 (36.1–116.5)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.442</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>max</sub> (mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">9.4 (2.9–23.7)</td> <td align="center" colspan="1" rowspan="1" valign="top">10.4 (5.7–23.3)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.499</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF0–2</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">0.2 (0.1–0.4)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.1 (0.1–0.1)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF3–5</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">0.3 (0.1–0.8)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.1 (0.1–0.3)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF6–8</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">0.4 (0.1–1.4)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.2 (0.1–0.7)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top">Pyrazinamide</td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> <td align="left" colspan="1" rowspan="1" valign="top"></td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> AUC<sub>0–24</sub> (h∙mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">315.5 (100.6–599.0)</td> <td align="center" colspan="1" rowspan="1" valign="top">328.4 (143.3–1477.7)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.482</td> </tr> <tr> <td align="left" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>max</sub> (mg/L)</td> <td align="center" colspan="1" rowspan="1" valign="top">37.7 (15.9–61.7)</td> <td align="center" colspan="1" rowspan="1" valign="top">40.5 (22.7–88.4)</td> <td align="center" colspan="1" rowspan="1" valign="top">0.350</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF0–2</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">24.4 (11.1–54.9)</td> <td align="center" colspan="1" rowspan="1" valign="top">25.6 (21.3–37.1)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF3–5</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">30.0 (19.2–43.3)</td> <td align="center" colspan="1" rowspan="1" valign="top">24.7 (15.9–38.1)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> <tr> <td align="center" colspan="1" rowspan="1" valign="top"> <em>C</em> <sub>CSF6–8</sub> (mg/L)†</td> <td align="center" colspan="1" rowspan="1" valign="top">19.6 (7.2–37.7)</td> <td align="center" colspan="1" rowspan="1" valign="top">39.4 (23.1–70.8)</td> <td align="left" colspan="1" rowspan="1" valign="top">n/a</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/T2/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div> <div class="tw-foot p"> <div class="fn" id="T2_FN1"><p>Data are presented as geometric mean (range). The first PK assessment was performed on day 2 of treatment and the second PK assessment was performed on day 10 of treatment.</p></div> <div class="fn" id="T2_FN2"><p>*Paired-sample t-test on log-transformed data of 12 patients for whom PK data were available both at the first and second PK assessments.</p></div> <div class="fn" id="T2_FN3"><p>†At the first PK assessment, 6, 7 and 7 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively; and at the second PK assessment, 4, 4 and 3 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively.</p></div> <div class="fn" id="T2_FN4"><p>AUC<sub>0–24</sub>, area under the plasma concentration–time curve from 0 to 24 hours postdose; <em>C</em> <sub>CSF0–8</sub>, drug concentration in cerebrospinal fluid during 0–8 hours postdose; <em>C</em> <sub>max</sub>, peak plasma concentration; n/a, non-applicable; TBM, tuberculous meningitis.</p></div>
"""


@pytest.fixture(scope="module")
def html_content_34183327table_2():
    return ghtml_content_34183327_table_2


@pytest.fixture(scope="module")
def caption_34183327_table_2():
    return """
Summary of pharmacokinetic (PK) parameters of isoniazid, rifampicin and pyrazinamide among Indonesian children treated for TBM.
Data are presented as geometric mean (range). The first PK assessment was performed on day 2 of treatment and the second PK assessment was performed on day 10 of treatment. *Paired-sample t-test on log-transformed data of 12 patients for whom PK data were available both at the first and second PK assessments. †At the first PK assessment, 6, 7 and 7 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively; and at the second PK assessment, 4, 4 and 3 CSF samples for each drug were available at 0–2 hours, 3–5 hours and 6–8 hours, respectively. AUC0–24, area under the plasma concentration–time curve from 0 to 24 hours postdose; C CSF0–8, drug concentration in cerebrospinal fluid during 0–8 hours postdose; C max, peak plasma concentration; n/a, non-applicable; TBM, tuberculous meningitis.
"""

@pytest.fixture(scope="module")
def md_table_34183327_table_2():
    return single_html_table_to_markdown(ghtml_content_34183327_table_2)

@pytest.fixture(scope="module")
def md_table_patient_34183327_table_2():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| Indonesian children | N/A | 20 |
| Indonesian children | N/A | 12 |
| Indonesian children | N/A | 6 |
| Indonesian children | N/A | 7 |
| Indonesian children | N/A | 4 |
| Indonesian children | N/A | 3 |
"""

@pytest.fixture(scope="module")
def md_table_patient_refined_34183327_table_2():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Children | N/A | N/A | 20 |
| Children | N/A | N/A | 12 |
| Children | N/A | N/A | 6 |
| Children | N/A | N/A | 7 |
| Children | N/A | N/A | 4 |
| Children | N/A | N/A | 3 |
"""

@pytest.fixture(scope="module")
def md_table_aligned_34183327_table_2():
    return """
| Parameter type | First PK assessment (n=20) | Second PK assessment (n=12) | P value* |
| --- | --- | --- | --- |
| Isoniazid | nan | nan | nan |
| AUC0–24 (h∙mg/L) | 18.5 (5.1–47.4) | 14.5 (5.9–44.2) | 0.888 |
| C max (mg/L) | 4.6 (1.0–10.0) | 4.7 (2.5–13.6) | 0.366 |
| C CSF0–2 (mg/L)† | 1.4 (0.5–6.1) | 1.6 (1.2–2.5) | nan |
| C CSF3–5 (mg/L)† | 1.6 (0.3–5.0) | 1.7 (0.6–5.0) | nan |
| C CSF6–8 (mg/L)† | 1.3 (1.2–4.3) | 2.3 (1.9–2.8) | nan |
| Rifampicin | nan | nan | nan |
| AUC0–24 (h∙mg/L) | 66.9 (21.7–118.6) | 71.8 (36.1–116.5) | 0.442 |
| C max (mg/L) | 9.4 (2.9–23.7) | 10.4 (5.7–23.3) | 0.499 |
| C CSF0–2 (mg/L)† | 0.2 (0.1–0.4) | 0.1 (0.1–0.1) | nan |
| C CSF3–5 (mg/L)† | 0.3 (0.1–0.8) | 0.1 (0.1–0.3) | nan |
| C CSF6–8 (mg/L)† | 0.4 (0.1–1.4) | 0.2 (0.1–0.7) | nan |
| Pyrazinamide | nan | nan | nan |
| AUC0–24 (h∙mg/L) | 315.5 (100.6–599.0) | 328.4 (143.3–1477.7) | 0.482 |
| C max (mg/L) | 37.7 (15.9–61.7) | 40.5 (22.7–88.4) | 0.35 |
| C CSF0–2 (mg/L)† | 24.4 (11.1–54.9) | 25.6 (21.3–37.1) | nan |
| C CSF3–5 (mg/L)† | 30.0 (19.2–43.3) | 24.7 (15.9–38.1) | nan |
| C CSF6–8 (mg/L)† | 19.6 (7.2–37.7) | 39.4 (23.1–70.8) | nan |
"""

@pytest.fixture(scope="module")
def md_table_list_34183327_table_2():
    return ["""
 | Parameter type | First PK assessment (n=20) | P value* |
 | --- | --- | --- |
 | Isoniazid | nan | nan |
 | AUC0–24 (h∙mg/L) | 18.5 (5.1–47.4) | 0.888 |
 | C max (mg/L) | 4.6 (1.0–10.0) | 0.366 |
 | C CSF0–2 (mg/L)† | 1.4 (0.5–6.1) | nan |
 | C CSF3–5 (mg/L)† | 1.6 (0.3–5.0) | nan |
 | C CSF6–8 (mg/L)† | 1.3 (1.2–4.3) | nan |
 | Rifampicin | nan | nan |
 | AUC0–24 (h∙mg/L) | 66.9 (21.7–118.6) | 0.442 |
 | C max (mg/L) | 9.4 (2.9–23.7) | 0.499 |
 | C CSF0–2 (mg/L)† | 0.2 (0.1–0.4) | nan |
 | C CSF3–5 (mg/L)† | 0.3 (0.1–0.8) | nan |
 | C CSF6–8 (mg/L)† | 0.4 (0.1–1.4) | nan |
 | Pyrazinamide | nan | nan |
 | AUC0–24 (h∙mg/L) | 315.5 (100.6–599.0) | 0.482 |
 | C max (mg/L) | 37.7 (15.9–61.7) | 0.35 |
 | C CSF0–2 (mg/L)† | 24.4 (11.1–54.9) | nan |
 | C CSF3–5 (mg/L)† | 30.0 (19.2–43.3) | nan |
 | C CSF6–8 (mg/L)† | 19.6 (7.2–37.7) | nan |
 """, """
 | Parameter type | Second PK assessment (n=12) | P value* |
 | --- | --- | --- |
 | Isoniazid | nan | nan |
 | AUC0–24 (h∙mg/L) | 14.5 (5.9–44.2) | 0.888 |
 | C max (mg/L) | 4.7 (2.5–13.6) | 0.366 |
 | C CSF0–2 (mg/L)† | 1.6 (1.2–2.5) | nan |
 | C CSF3–5 (mg/L)† | 1.7 (0.6–5.0) | nan |
 | C CSF6–8 (mg/L)† | 2.3 (1.9–2.8) | nan |
 | Rifampicin | nan | nan |
 | AUC0–24 (h∙mg/L) | 71.8 (36.1–116.5) | 0.442 |
 | C max (mg/L) | 10.4 (5.7–23.3) | 0.499 |
 | C CSF0–2 (mg/L)† | 0.1 (0.1–0.1) | nan |
 | C CSF3–5 (mg/L)† | 0.1 (0.1–0.3) | nan |
 | C CSF6–8 (mg/L)† | 0.2 (0.1–0.7) | nan |
 | Pyrazinamide | nan | nan |
 | AUC0–24 (h∙mg/L) | 328.4 (143.3–1477.7) | 0.482 |
 | C max (mg/L) | 40.5 (22.7–88.4) | 0.35 |
 | C CSF0–2 (mg/L)† | 25.6 (21.3–37.1) | nan |
 | C CSF3–5 (mg/L)† | 24.7 (15.9–38.1) | nan |
 | C CSF6–8 (mg/L)† | 39.4 (23.1–70.8) | nan |
 """]


## =============================================================================
# 35465728_table_2

ghtml_content_35465728_table_2 = """
<section class="tw xbox font-sm" id="T2"><h4 class="obj_head">TABLE 2.</h4> <div class="caption p"><p>Vancomycin exposure before and after implementation of the vancomycin dose-optimization protocol</p></div> <div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups"> <colgroup span="1"> <col span="1"/> <col span="1"/> <col span="1"/> <col span="1"/> <col span="1"/> </colgroup> <thead> <tr> <th colspan="1" rowspan="2">Factor</th> <th colspan="1" rowspan="1"></th> <th colspan="2" rowspan="1">Data for:<a class="usa-link" href="#T2F1"><sup><em>a</em></sup></a><hr/> </th> <th colspan="1" rowspan="2"> <em>P</em> value</th> </tr> <tr> <th colspan="1" rowspan="1"></th> <th colspan="1" rowspan="1">Before group</th> <th colspan="1" rowspan="1">After group</th> </tr> </thead> <tbody> <tr> <td colspan="1" rowspan="1">Initial Cavg</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"> <em>n</em> = 60</td> <td colspan="1" rowspan="1"> <em>n</em> = 59</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> Repartition<a class="usa-link" href="#T2F2"><sup><em>b</em></sup></a> </td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1">  Subtherapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">41 (68.3)</td> <td colspan="1" rowspan="1">6 (10.2)</td> <td colspan="1" rowspan="1">&lt;0.001</td> </tr> <tr> <td colspan="1" rowspan="1">  Therapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">17 (28.3)</td> <td colspan="1" rowspan="1">44 (74.6)</td> <td colspan="1" rowspan="1">&lt;0.001</td> </tr> <tr> <td colspan="1" rowspan="1">  Supra-therapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">2 (3.3)</td> <td colspan="1" rowspan="1">9 (15.3)</td> <td colspan="1" rowspan="1">0.001</td> </tr> <tr> <td colspan="1" rowspan="1"> Concentration (mg/L)</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">12.9 [11.3–17.0]</td> <td colspan="1" rowspan="1">20.3 [17.0–22.2]</td> <td colspan="1" rowspan="1">&lt;0.001</td> </tr> <tr> <td colspan="1" rowspan="1">All Cavg</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"> <em>n</em> = 116</td> <td colspan="1" rowspan="1"> <em>n</em> = 103</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> Repartition<a class="usa-link" href="#T2F2"><sup><em>b</em></sup></a> </td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1">  Subtherapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">78 (67.2)</td> <td colspan="1" rowspan="1">13 (12.6)</td> <td colspan="1" rowspan="1">&lt;0.001</td> </tr> <tr> <td colspan="1" rowspan="1">  Therapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">36 (31.0)</td> <td colspan="1" rowspan="1">77 (74.8)</td> <td colspan="1" rowspan="1">&lt;0.001</td> </tr> <tr> <td colspan="1" rowspan="1">  Supra-therapeutic</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">2 (1.7)</td> <td colspan="1" rowspan="1">13 (12.6)</td> <td colspan="1" rowspan="1">0.025</td> </tr> <tr> <td colspan="1" rowspan="1"> Concentration (mg/L)</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">13.1 [11.3–16.2]</td> <td colspan="1" rowspan="1">19.8 [16.8–22.1]</td> <td colspan="1" rowspan="1">&lt; 0.001</td> </tr> <tr> <td colspan="1" rowspan="1">Initial Cavg/MIC ratio</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"> <em>n</em><strong> = </strong>22</td> <td colspan="1" rowspan="1"> <em>n</em><strong> = </strong>17</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> Repartition</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1">  &lt;8</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">10 (45.5)</td> <td colspan="1" rowspan="1">2 (11.8)</td> <td colspan="1" rowspan="1">0.02</td> </tr> <tr> <td colspan="1" rowspan="1">  ≥8</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">12 (54.5)</td> <td colspan="1" rowspan="1">15 (88.2)</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> Cavg/MIC ratio</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">8.8 [6.2–11.5]</td> <td colspan="1" rowspan="1">12.8 [10.9–20.9]</td> <td colspan="1" rowspan="1">0.004</td> </tr> <tr> <td colspan="1" rowspan="1">Initial AUC/MIC ratio</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"> <em>n</em> <strong>=</strong> 22</td> <td colspan="1" rowspan="1"> <em>n</em> <strong>=</strong> 17</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> Repartition</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1">  &lt;400</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">20/22 (90.9)</td> <td colspan="1" rowspan="1">10/17 (58.8)</td> <td colspan="1" rowspan="1">0.02</td> </tr> <tr> <td colspan="1" rowspan="1">  ≥400</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">2/22 (9.1)</td> <td colspan="1" rowspan="1">7/17 (41.2)</td> <td colspan="1" rowspan="1"></td> </tr> <tr> <td colspan="1" rowspan="1"> AUC/MIC ratio</td> <td colspan="1" rowspan="1"></td> <td colspan="1" rowspan="1">211 [149–275]</td> <td colspan="1" rowspan="1">307 [262–502]</td> <td colspan="1" rowspan="1">0.006</td> </tr> </tbody> </table></div> <div class="p text-right font-secondary"><a class="usa-link" href="table/T2/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div> <div class="tw-foot p"> <div class="fn" id="T2F1"> <sup>a</sup><p class="display-inline">Data are reported as <em>n</em> (%) or medians [25<sup>th</sup> to 75<sup>th</sup> percentiles].</p> </div> <div class="fn" id="T2F2"> <sup>b</sup><p class="display-inline">Therapeutic range, Cavg between 15 and 25 mg/L; subtherapeutic, Cavg &lt; 15 mg/L; supra-therapeutic, Cavg &gt; 25mg/L. Cavg, average concentration; AUC, area under the curve.</p> </div> </div></section>
"""

@pytest.fixture(scope="module")
def html_content_35465728_table_2():
    return ghtml_content_35465728_table_2

@pytest.fixture(scope="module")
def title_35465728():
    return "Implementation of a Vancomycin Dose-Optimization Protocol in Neonates: Impact on Vancomycin Exposure, Biological Parameters, and Clinical Outcomes"
    
@pytest.fixture(scope="module")
def md_table_35465728_table_2():
    return single_html_table_to_markdown(ghtml_content_35465728_table_2)

@pytest.fixture(scope="module")
def caption_35465728_table_2():
    return """
Vancomycin exposure before and after implementation of the vancomycin dose-optimization protocol.
aData are reported as n (%) or medians [25th to 75th percentiles].

bTherapeutic range, Cavg between 15 and 25 mg/L; subtherapeutic, Cavg < 15 mg/L; supra-therapeutic, Cavg > 25mg/L. Cavg, average concentration; AUC, area under the curve.
"""


@pytest.fixture(scope="module")
def md_table_patient_35465728_table_2():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| N/A | N/A | 60 |
| N/A | N/A | 59 |
| N/A | N/A | 116 |
| N/A | N/A | 103 |
| N/A | N/A | 22 |
| N/A | N/A | 17 |
"""


@pytest.fixture(scope="module")
def md_table_patient_refined_35465728_table_2():
    return """
| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| N/A | N/A | N/A | 60 |
| N/A | N/A | N/A | 59 |
| N/A | N/A | N/A | 116 |
| N/A | N/A | N/A | 103 |
| N/A | N/A | N/A | 22 |
| N/A | N/A | N/A | 17 |
"""

@pytest.fixture(scope="module")
def md_table_summary_35465728_table_2():
    return """
| Factor/Factor | Unnamed: 1_level_0/Unnamed: 1_level_1 | Data for:a/Before group | Data for:a/After group | P value/P value |
| --- | --- | --- | --- | --- |
| Initial Cavg | nan | n = 60 | n = 59 | nan |
| Repartitionb | nan | nan | nan | nan |
| Subtherapeutic | nan | 41 (68.3) | 6 (10.2) | <0.001 |
| Therapeutic | nan | 17 (28.3) | 44 (74.6) | <0.001 |
| Supra-therapeutic | nan | 2 (3.3) | 9 (15.3) | 0.001 |
| Concentration (mg/L) | nan | 12.9 [11.3–17.0] | 20.3 [17.0–22.2] | <0.001 |
| All Cavg | nan | n = 116 | n = 103 | nan |
| Repartitionb | nan | nan | nan | nan |
| Subtherapeutic | nan | 78 (67.2) | 13 (12.6) | <0.001 |
| Therapeutic | nan | 36 (31.0) | 77 (74.8) | <0.001 |
| Supra-therapeutic | nan | 2 (1.7) | 13 (12.6) | 0.025 |
| Concentration (mg/L) | nan | 13.1 [11.3–16.2] | 19.8 [16.8–22.1] | < 0.001 |
| Initial Cavg/MIC ratio | nan | n = 22 | n = 17 | nan |
| Repartition | nan | nan | nan | nan |
| <8 | nan | 10 (45.5) | 2 (11.8) | 0.02 |
| ≥8 | nan | 12 (54.5) | 15 (88.2) | nan |
| Cavg/MIC ratio | nan | 8.8 [6.2–11.5] | 12.8 [10.9–20.9] | 0.004 |
| Initial AUC/MIC ratio | nan | n = 22 | n = 17 | nan |
| Repartition | nan | nan | nan | nan |
| <400 | nan | 20/22 (90.9) | 10/17 (58.8) | 0.02 |
| ≥400 | nan | 2/22 (9.1) | 7/17 (41.2) | nan |
| AUC/MIC ratio | nan | 211 [149–275] | 307 [262–502] | 0.006 |
"""


@pytest.fixture(scope="module")
def md_table_aligned_35465728_table_2():
    return """
| Parameter type | Initial Cavg | Repartitionb_0 | Subtherapeutic_0 | Therapeutic_0 | Supra-therapeutic_0 | Concentration (mg/L)_0 | All Cavg | Repartitionb_1 | Subtherapeutic_1 | Therapeutic_1 | Supra-therapeutic_1 | Concentration (mg/L)_1 | Initial Cavg/MIC ratio | Repartition_0 | <8 | ≥8 | Cavg/MIC ratio | Initial AUC/MIC ratio | Repartition_1 | <400 | ≥400 | AUC/MIC ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| Data for:a/Before group | n = 60 | nan | 41 (68.3) | 17 (28.3) | 2 (3.3) | 12.9 [11.3–17.0] | n = 116 | nan | 78 (67.2) | 36 (31.0) | 2 (1.7) | 13.1 [11.3–16.2] | n = 22 | nan | 10 (45.5) | 12 (54.5) | 8.8 [6.2–11.5] | n = 22 | nan | 20/22 (90.9) | 2/22 (9.1) | 211 [149–275] |
| Data for:a/After group | n = 59 | nan | 6 (10.2) | 44 (74.6) | 9 (15.3) | 20.3 [17.0–22.2] | n = 103 | nan | 13 (12.6) | 77 (74.8) | 13 (12.6) | 19.8 [16.8–22.1] | n = 17 | nan | 2 (11.8) | 15 (88.2) | 12.8 [10.9–20.9] | n = 17 | nan | 10/17 (58.8) | 7/17 (41.2) | 307 [262–502] |
| P value/P value | nan | nan | <0.001 | <0.001 | 0.001 | <0.001 | nan | nan | <0.001 | <0.001 | 0.025 | < 0.001 | nan | nan | 0.02 | nan | 0.004 | nan | nan | 0.02 | nan | 0.006 |
"""

@pytest.fixture(scope="module")
def col_mapping_35465728_table_2():
    return {
        'Parameter type': 'Parameter type', 
        'Initial Cavg': 'Parameter value', 
        'Repartitionb_0': 'Parameter value', 
        'Subtherapeutic_0': 'Parameter value', 
        'Therapeutic_0': 'Parameter value', 
        'Supra-therapeutic_0': 'Parameter value', 
        'Concentration (mg/L)_0': 'Parameter value', 
        'All Cavg': 'Parameter value', 
        'Repartitionb_1': 'Parameter value', 
        'Subtherapeutic_1': 'Parameter value', 
        'Therapeutic_1': 'Parameter value', 
        'Supra-therapeutic_1': 'Parameter value', 
        'Concentration (mg/L)_1': 'Parameter value', 
        'Initial Cavg/MIC ratio': 'Parameter value', 
        'Repartition_0': 'Parameter value', 
        '<8': 'Parameter value', 
        '≥8': 'Parameter value', 
        'Cavg/MIC ratio': 'Parameter value', 
        'Initial AUC/MIC ratio': 'Parameter value', 
        'Repartition_1': 'Parameter value', 
        '<400': 'Parameter value', 
        '≥400': 'Parameter value', 
        'AUC/MIC ratio': 'Parameter value'
    }

@pytest.fixture(scope="module")
def md_table_list_35465728_table_2():
    return ["""
| Parameter type | Initial Cavg |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | n = 60 |
| Data for:a/After group | n = 59 |
| P value/P value | nan |
""", """
| Parameter type | Subtherapeutic_0 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 41 (68.3) |
| Data for:a/After group | 6 (10.2) |
| P value/P value | <0.001 |
""", """
| Parameter type | Therapeutic_0 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 17 (28.3) |
| Data for:a/After group | 44 (74.6) |
| P value/P value | <0.001 |
""", """
| Parameter type | Supra-therapeutic_0 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 2 (3.3) |
| Data for:a/After group | 9 (15.3) |
| P value/P value | 0.001 |
""", """
| Parameter type | Concentration (mg/L)_0 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 12.9 [11.3–17.0] |
| Data for:a/After group | 20.3 [17.0–22.2] |
| P value/P value | <0.001 |
""", """
| Parameter type | All Cavg |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | n\u2009=\u2009116 |
| Data for:a/After group | n\u2009=\u2009103 |
| P value/P value | nan |
""", """
| Parameter type | Subtherapeutic_1 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 78 (67.2) |
| Data for:a/After group | 13 (12.6) |
| P value/P value | <0.001 |
""", """
| Parameter type | Therapeutic_1 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 36 (31.0) |
| Data for:a/After group | 77 (74.8) |
| P value/P value | <0.001 |
""", """
| Parameter type | Supra-therapeutic_1 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 2 (1.7) |
| Data for:a/After group | 13 (12.6) |
| P value/P value | 0.025 |
""", """
| Parameter type | Concentration (mg/L)_1 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 13.1 [11.3–16.2] |
| Data for:a/After group | 19.8 [16.8–22.1] |
| P value/P value | < 0.001 |
""", """
| Parameter type | Initial Cavg/MIC ratio |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | n\u2009=\u200922 |
| Data for:a/After group | n\u2009=\u200917 |
| P value/P value | nan |
""", """
| Parameter type | <8 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 10 (45.5) |
| Data for:a/After group | 2 (11.8) |
| P value/P value | 0.02 |
""", """
| Parameter type | ≥8 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 12 (54.5) |
| Data for:a/After group | 15 (88.2) |
| P value/P value | nan |
""", """
| Parameter type | Cavg/MIC ratio |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 8.8 [6.2–11.5] |
| Data for:a/After group | 12.8 [10.9–20.9] |
| P value/P value | 0.004 |
""", """
| Parameter type | Initial AUC/MIC ratio |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | n = 22 |
| Data for:a/After group | n = 17 |
| P value/P value | nan |
""", """
| Parameter type | <400 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 20/22 (90.9) |
| Data for:a/After group | 10/17 (58.8) |
| P value/P value | 0.02 |
""", """
| Parameter type | ≥400 |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 2/22 (9.1) |
| Data for:a/After group | 7/17 (41.2) |
| P value/P value | nan |
""", """
| Parameter type | AUC/MIC ratio |
| --- | --- |
| Unnamed: 1_level_0/Unnamed: 1_level_1 | nan |
| Data for:a/Before group | 211 [149–275] |
| Data for:a/After group | 307 [262–502] |
| P value/P value | 0.006 |
"""
    ]


@pytest.fixture(scope="module")
def md_table_combined_31935538():
    return """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Parameter statistic | Parameter value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Body weight | kg | Mean | 20.2 | SD | 9.2 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Serum creatinine | mg/dL | Mean | 0.3 | SD | 0.1 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Creatinine clearance | mL/min | Mean | 199.7 | SD | 69.6 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Dose | mg/kg/day | Mean | 64.3 | SD | 7.7 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Elimination rate constant | hr⁻¹ | Mean | 5.293 | SD | 0.090 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Adjusted R-square | N/A | Mean | 0.979 | SD | 0.033 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Volume of distribution | L/kg | Mean | 0.55 | SD | 0.10 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Volume of distribution at steady state | L/kg | Mean | 0.49 | SD | 0.09 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Half-life | hr | Mean | 2.65 | SD | 1.12 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Clearance | mL/kg/min | Mean | 2.63 | SD | 0.69 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Trough concentration | mg/L | Mean | 5.69 | SD | 3.00 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Area under the curve (AUCₐ₄,ss) | mg·hr/L | Mean | 450.2 | SD | 184.5 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Body weight | kg | Median | 16.5 | N/A | N/A | Range | 13.4 | 28.0 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Serum creatinine | mg/dL | Median | 0.4 | N/A | N/A | Range | 0.3 | 0.4 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Creatinine clearance | mL/min | Median | 183.1 | N/A | N/A | Range | 151.2 | 216.0 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Dose | mg/kg/day | Median | 62.5 | N/A | N/A | Range | 59.3 | 66.5 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Elimination rate constant | hr⁻¹ | Median | 0.297 | N/A | N/A | Range | 0.244 | 0.328 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Adjusted R-square | N/A | Median | 0.992 | N/A | N/A | Range | 0.988 | 0.995 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Volume of distribution | L/kg | Median | 0.58 | N/A | N/A | Range | 0.48 | 0.62 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Volume of distribution at steady state | L/kg | Median | 0.49 | N/A | N/A | Range | 0.33 | 0.62 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Half-life | hr | Median | 2.33 | N/A | N/A | Range | 2.11 | 2.84 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Clearance | mL/kg/min | Median | 2.82 | N/A | N/A | Range | 2.17 | 3.03 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Trough concentration | mg/L | Median | 4.81 | N/A | N/A | Range | 3.66 | 7.67 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | 1–12 years | 14 | Area under the curve (AUCₐ₄,ss) | mg·hr/L | Median | 359.0 | N/A | N/A | Range | 320.3 | 576.2 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Central compartment volume of distribution | L | Mean | 4.31 | N/A | N/A | Range | 3.56 | 5.07 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Peripheral compartment volume of distribution | L | Mean | 4.63 | N/A | N/A | Range | 3.33 | 5.93 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Systemic clearance | L/h | Mean | 2.54 | N/A | N/A | Range | 1.84 | 3.24 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Distributive clearance | L/h | Mean | 3.51 | N/A | N/A | Range | 2.09 | 4.92 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Multiplicative residual error | N/A | Mean | 0.16 | N/A | N/A | Range | 0.12 | 0.21 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Central compartment volume of distribution | L | Mean | 0.096 | CV% | 4.5 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Systemic clearance | L/h | Mean | 0.335 | CV% | 9.3 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Central compartment volume of distribution | L | Mean | 3.86 | N/A | N/A | Range | 2.92 | 4.81 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-1 | N/A | Mean | 0.19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-2 | N/A | Mean | 1.07 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-3 | N/A | Mean | 0.16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-4 | N/A | Mean | 0.97 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-5 | N/A | Mean | 0.13 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-6 | N/A | Mean | 1.19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Multiplicative error | N/A | Mean | 0.12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Central compartment volume of distribution | L | Mean | 0.048 | RSE (%) | 2.1 | Range | 2.92 | 4.81 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Systemic clearance | L/h | Mean | 0.062 | RSE (%) | 1.9 | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Central compartment volume of distribution | L | Median | 3.86 | N/A | N/A | Range | 3.01 | 4.74 | 0.048 | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-1 | N/A | Median | 0.23 | N/A | N/A | Range | 0.07 | 0.30 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-2 | N/A | Median | 1.07 | N/A | N/A | Range | 0.88 | 1.27 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-3 | N/A | Median | 0.16 | N/A | N/A | Range | 0.04 | 0.29 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-4 | N/A | Median | 1.01 | N/A | N/A | Range | 0.70 | 1.24 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-5 | N/A | Median | 0.36 | N/A | N/A | Range | -0.11 | 0.37 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | THETA-6 | N/A | Median | 1.15 | N/A | N/A | Range | 0.69 | 1.69 | N/A | N/A | N/A |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Multiplicative error | N/A | Median | 0.11 | N/A | N/A | Range | 0.08 | 0.15 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Probability for AUC 24 in [≥400, <800] mg-h/L | Probability (unitless) | N/A  | 0.258  | N/A | N/A    | N/A | N/A | N/A | N/A | 24 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for AUC 24 in [≥400, <800] mg-h/L | mg-h/L                 | Mean | 535.86 | SD  | 103.82 | N/A | N/A | N/A | N/A | 24 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Median | 509.99 | N/A | N/A | Range | 451.43 | 606.06 | N/A | 24 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 6 hours | mg/L | Mean | 7.04 | SD | 2.17 | N/A | N/A | N/A | N/A | 6 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 6 hours | mg/L | Median | 6.95 | N/A | N/A | Range | 5.60 | 8.39 | N/A | 6 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 24 hours | mg/L | Mean | 11.75 | SD | 4.22 | N/A | N/A | N/A | N/A | 24 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 24 hours | mg/L | Median | 11.18 | N/A | N/A | Range | 8.56 | 14.56 | N/A | 24 | Hour |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Probability for AUC 24 in [≥400, <800] mg-h/L | Probability (unitless) | Prob | 0.406 | N/A | N/A | N/A | N/A | N/A | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Mean | 561.00 | SD | 110.84 | N/A | N/A | N/A | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Median | 543.13 | N/A | N/A | Range | 463.04 | 648.78 | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 6 hours | mg/L | Mean | 6.81 | SD | 2.54 | N/A | N/A | N/A | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 6 hours | mg/L | Median | 6.67 | N/A | N/A | Range | 4.95 | 8.57 | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 24 hours | mg/L | Mean | 10.10 | SD | 4.36 | N/A | N/A | N/A | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 24 hours | mg/L | Median | 9.50 | N/A | N/A | Range | 6.77 | 13.08 | 0.406 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Probability for AUC 24 in [≥400, <800] mg-h/L | Probability (unitless) | N/A | 0.434 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Mean | 580.26 | SD | 111.92 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Median | 570.48 | N/A | N/A | Range | 483.27 | 672.66 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 6 hours | mg/L | Mean | 6.20 | SD | 2.81 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 6 hours | mg/L | Median | 5.98 | N/A | N/A | Range | 3.99 | 8.22 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 24 hours | mg/L | Mean | 8.51 | SD | 4.20 | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 24 hours | mg/L | Median | 7.91 | N/A | N/A | Range | 5.21 | 11.24 | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Probability for AUC 24 in [≥400, <800] mg-h/L | Probability (unitless) | Probability | 0.401 | N/A | N/A | N/A | N/A | N/A | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Mean | 595.12 | SD | 114.17 | N/A | N/A | N/A | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for AUC 24 in [≥400, <800] mg-h/L | mg-h/L | Median | 591.48 | N/A | N/A | Range | 498.17 | 691.21 | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 6 hours | mg/L | Mean | 5.49 | SD | 2.78 | N/A | N/A | N/A | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 6 hours | mg/L | Median | 5.20 | N/A | N/A | Range | 3.36 | 7.39 | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Mean and SD for C trough at 24 hours | mg/L | Mean | 7.18 | SD | 3.88 | N/A | N/A | N/A | 0.401 | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | Children | N/A | N/A | 10000 | Median and IQR for C trough at 24 hours | mg/L | Median | 6.55 | N/A | N/A | Range | 4.18 | 9.61 | 0.401 | N/A | N/A |
"""

@pytest.fixture(scope="module")
def caption_29718415_table_3():
    """ table 3 (index 2) """
    return """
Table 3.
Open in new tab
Vancomycin Dosing Information
Abbreviations: CIV, continuous infusion vancomycin; IIV, intermittent infusion vancomycin; SD, standard deviation; SS, steady state; SVC, serum vancomycin concentration; TDD, total daily dose.aIncludes all patients for whom an SVC was measured, including those whose only SVC measurement was performed before reaching steady state.bDefined as ≥23 hours after initiating continuous infusion.cIncludes only the patients who achieved a therapeutic SVC.
"""

@pytest.fixture(scope="module")
def md_table_assembly_29718415():
    return """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Main value | Statistics type | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 94 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 28 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 21 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 40 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 24 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 15 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 30 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 31 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 31 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 52 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 38 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 25 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 24 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 22 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 27 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 41 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 21 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 24 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 15 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 14 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 15 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 15 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 27 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 35 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 38 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 18 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 28 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 25 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 35 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 26 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 22 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 22 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 14 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 18 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 22 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 24 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 54 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 21 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 29 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 54 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 39 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 21 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 47 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 56 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 34 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 40 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 32 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 15 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 6 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 4 | Count | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Plasma | N/A | N/A | N/A | N/A | Total Views: | count | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
"""

@pytest.fixture(scope="module")
def md_table_curated_29718415_table_3():
    """ table index is 2 """
    return """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Parameter statistic | Parameter value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 18 | Initial Steady-State Serum Vancomycin Concentration | µg/mL | Mean | 12.5 | SD | 3.4 | N/A | N/A | N/A | .077 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 18 | Total Daily Dose at Time of Initial Steady-State Serum Vancomycin Concentration | mg/kg per day | Mean | 46.8 | SD | 5.4 | N/A | N/A | N/A | .008 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 18 | Total Daily Dose at Time of Initial Steady-State Serum Vancomycin Concentration | mg/kg per day | Mean | 41.9 | SD | 7.5 | N/A | N/A | N/A | .125 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 21 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 72.5 | SD | 12.6 | N/A | N/A | N/A | .040 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 21 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 41.5 | SD | 7.0 | N/A | N/A | N/A | .023 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 21 | Patients Reaching Therapeutic Serum Vancomycin Concentration | Count (%) | Count | 31 | N/A | N/A | N/A | N/A | N/A | .365 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 21 | Total Daily Dose at Time of First Therapeutic Serum Vancomycin Concentration | mg/kg per day | Mean | 50.6 | SD | 6.3 | N/A | N/A | N/A | .008 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 52 | Initial Steady-State Serum Vancomycin Concentration | µg/mL | Mean | 13.6 | SD | 3.7 | N/A | N/A | N/A | .009 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 52 | Initial Steady-State Serum Vancomycin Concentration | µg/mL | Mean | 15.6 | SD | 3.1 | N/A | N/A | N/A | .009 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 52 | Total Daily Dose at Time of Initial Steady-State Serum Vancomycin Concentration | mg/kg per day | Mean | 43.6 | SD | 5.4 | N/A | N/A | N/A | .008 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Patients Reaching Therapeutic Serum Vancomycin Concentration | Count (%) | Count | 23 | N/A | N/A | N/A | N/A | N/A | <.005 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Patients Reaching Therapeutic Serum Vancomycin Concentration | Count (%) | Count | 14 | N/A | N/A | N/A | N/A | N/A | .365 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Time to Therapeutic Serum Vancomycin Concentration | Days | Mean | 1.8 | SD | 1.1 | N/A | N/A | N/A | .831 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Time to Therapeutic Serum Vancomycin Concentration | Days | Mean | 1.6 | SD | 1.3 | N/A | N/A | N/A | .535 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Total Daily Dose at Time of First Therapeutic Serum Vancomycin Concentration | mg/kg per day | Mean | 39.4 | SD | 7.3 | N/A | N/A | N/A | <.005 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 54 | Total Daily Dose at Time of First Therapeutic Serum Vancomycin Concentration | mg/kg per day | Mean | 44.7 | SD | 10.2 | N/A | N/A | N/A | .008 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 71 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 72.9 | SD | 13.8 | N/A | N/A | N/A | .023 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 71 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 43.1 | SD | 7.1 | N/A | N/A | N/A | .002 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Adolescents | N/A | 8 to <18 years | 71 | Patients Reaching Steady-State Serum Vancomycin Concentration | Count (%) | Count | 39 | N/A | N/A | N/A | N/A | N/A | .167 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 23 | Initial Steady-State Serum Vancomycin Concentration | µg/mL | Mean | 11.2 | SD | 2.4 | N/A | N/A | N/A | .077 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 28 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 44.5 | SD | 4.8 | N/A | N/A | N/A | .023 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 28 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 45.6 | SD | 4.2 | N/A | N/A | N/A | .002 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 31 | Patients Reaching Steady-State Serum Vancomycin Concentration | Count (%) | Count | 28 | N/A | N/A | N/A | N/A | N/A | .564 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 38 | Goal Serum Vancomycin Concentration of 15–20 µg/mL (N = 164) | Count (n) | Count | 56 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 39 | Total Daily Dose at Time of Initial Steady-State Serum Vancomycin Concentration | mg/kg per day | Mean | 44.4 | SD | 5.1 | N/A | N/A | N/A | .125 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 56 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 79.1 | SD | 8.5 | N/A | N/A | N/A | .040 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Children | N/A | 2 to <8 years | 56 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 78.7 | SD | 10.9 | N/A | N/A | N/A | .023 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 13 | Initial Steady-State Serum Vancomycin Concentration | µg/mL | Mean | 10.4 | SD | 1.6 | N/A | N/A | N/A | .077 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 13 | Patients Reaching Steady-State Serum Vancomycin Concentration | Count (%) | Count | 76 | N/A | N/A | N/A | N/A | N/A | .564 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 17 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 79.5 | SD | 9.6 | N/A | N/A | N/A | .040 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 17 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 46.2 | SD | 3.8 | N/A | N/A | N/A | .023 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 32 | Initial Total Daily Dose (on Continuous Infusion Vancomycin) | mg/kg per day | Mean | 47.0 | SD | 5.0 | N/A | N/A | N/A | .002 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 32 | Patients Reaching Steady-State Serum Vancomycin Concentration | Count (%) | Count | 86 | N/A | N/A | N/A | N/A | N/A | .167 | N/A | N/A |
| Vancomycin | Vancomycin | Serum | Infants | N/A | >31 days to <2 years | 37 | Final Total Daily Dose (on Intermittent Infusion Vancomycin) | mg/kg per day | Mean | 77.9 | SD | 12.4 | N/A | N/A | N/A | .023 | N/A | N/A |
"""

@pytest.fixture(scope="module")
def verification_reasoning_process_29718415_table_3():
    return """
**FinalAnswer**: Incorrect  
**Explanation**: The curated table has several discrepancies when compared directly with the source table in **Table 3** from the paper. I have identified key mismatched values and improper associations between parameters, groups, and statistics. Below are the details:  

1. **Population Mislabeling**: Some row entries incorrectly specify the population as "Adolescents" instead of the correct age-based category from the source data. For example:
   - In the curated table, entries for "Initial Total Daily Dose (on Continuous Infusion Vancomycin)" of **43.1 ± 7.1 mg/kg per day** and **44.5 ± 4.8 mg/kg per day** were labeled as "Adolescents" when they should refer to "Children" (2 to <8 years) based on **Table 3**.  

2. **Count (%) Discrepancy**: The row for "Patients Reaching Therapeutic Serum Vancomycin Concentration" lists **23** as the count for "Adolescents", which is incorrect. Based on **Table 3**, the correct value for "Adolescents" (8 to <18 years) is **54 (76%)**, not **23**.

3. **Duplicate or Misallocated Values**:
   - Several entries duplicate meaningful values without clarity on segmentation, such as assigning **"39.4 mg/kg per day"** and **"44.7 mg/kg per day"** for "Total Daily Dose at Time of First Therapeutic Serum Vancomycin Concentration". These entries come from two separate observations but are improperly allocated in the curated table. The rows fail to distinguish subpopulations effectively.

4. **Omission of Specific Age Groups or Incorrect Assignments**: In multiple records—with age groups like "Infants" (>31 days to <2 years) and "Children" (2 to <8 years)—the presented "Count (%)" values are not directly aligned with Table 3 data, such as assigning **"76" patients reaching SS SVC** to "Infants," which is not supported by the source Table.

5. **Ambiguous Representation of "Mean ± SD" Entries**: Duplicate parameter values (e.g., "1.8 ± 1.1 days" for time to therapeutic concentration) appear under incorrect age descriptors and are not aligned with their respective SVC group values.   

**SuggestedFix**: Below is a corrected version of the curated table with accurate representation based on Table 3 from the source:

| Drug name | Analyte    | Specimen | Population | Pregnancy stage | Pediatric/Gestational age     | Subject N | Parameter type                                              | Parameter unit | Parameter statistic | Parameter value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value | Time value | Time unit |
|-----------|------------|----------|------------|-----------------|--------------------------------|-----------|-----------------------------------------------------------|----------------|----------------------|-----------------|----------------|-----------------|---------------|-------------|-------------|---------|------------|-----------|
| Vancomycin | Vancomycin | Serum    | Infants    | N/A             | >31 days to <2 years          | 13        | Initial Steady-State Serum Vancomycin Concentration        | µg/mL         | Mean                | 10.4           | SD             | 1.6             | N/A           | N/A         | N/A         | .077    | N/A        | N/A       |
| Vancomycin | Vancomycin | Serum    | Infants    | N/A             | >31 days to <2 years          | 17        | Total Daily Dose at Time of First Therapeutic SVC          | mg/kg per day | Mean                | 48.4           | SD             | 4.6             | N/A           | N/A         | N/A         | <.005   | N/A        | N/A       |
| Vancomycin | Vancomycin | Serum    | Children   | N/A             | 2 to <8 years                 | 31        | Total Daily Dose at Time of First Therapeutic SVC          | mg/kg per day | Mean                | 45.6           | SD             | 5.5             | N/A           | N/A         | N/A         | <.005   | N/A        | N/A       |
| Vancomycin | Vancomycin | Serum    | Adolescents | N/A             | 8 to <18 years               | 14        | Total Daily Dose at Time of First Therapeutic SVC          | mg/kg per day | Mean                | 39.4           | SD             | 7.3             | N/A           | N/A         | N/A         | <.005   | N/A        | N/A       |
| Vancomycin | Vancomycin | Serum    | Children   | N/A             | 2 to <8 years                 | 28        | Patients Reaching Steady-State Vancomycin Concentration    | Count (%)     | Count               | 28             | N/A            | N/A             | N/A           | N/A         | N/A         | .564    | N/A        | N/A       |
"""

# ============================================================================================
# 18426260
# ============================================================================================
#
@pytest.fixture(scope="module")
def md_table_list_18426260_table_0():
    return """["| Parameter type | S-citalopram_0 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 40 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 32 |\n| ('30 Weeks', 'Dose,  mg/d') | 40 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 40 |\n| ('36 Weeks', 'Dose,  mg/d') | 50 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 25 |\n| ('Delivery', 'Dose,  mg/d') | 50 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 15 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 50 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 18 |", "| Parameter type | R-citalopram_0 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 62 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 64 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 41 |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 24 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 36 |", "| Parameter type | S-citalopram_1 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 20 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 14 |\n| ('30 Weeks', 'Dose,  mg/d') | 20 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 14 |\n| ('36 Weeks', 'Dose,  mg/d') | 40 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 18 |\n| ('Delivery', 'Dose,  mg/d') | NA |\n| ('Delivery', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 40 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 54 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 40 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 27 |", "| Parameter type | R-citalopram_1 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 24 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 24 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 36 |\n| ('Delivery', 'Dose,  mg/d') | NA |\n| ('Delivery', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 101 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 75 |", "| Parameter type | S-citalopram_2 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 30 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 19 |\n| ('30 Weeks', 'Dose,  mg/d') | 30 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 9 |\n| ('36 Weeks', 'Dose,  mg/d') | 20 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 5 |\n| ('Delivery', 'Dose,  mg/d') | … |\n| ('Delivery', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | … |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | … |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | … |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | … |", "| Parameter type | R-citalopram_2 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 41 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 22 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 14 |\n| ('Delivery', 'Dose,  mg/d') | … |\n| ('Delivery', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | … |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | … |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | … |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | … |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | … |", "| Parameter type | S-citalopram_3 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 10 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 17 |\n| ('30 Weeks', 'Dose,  mg/d') | 10 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 10 |\n| ('36 Weeks', 'Dose,  mg/d') | 10 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 13 |\n| ('Delivery', 'Dose,  mg/d') | 10 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 14 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 10 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 24 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | S-citalopram_4 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 20 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 58 |\n| ('30 Weeks', 'Dose,  mg/d') | 20 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 70 |\n| ('36 Weeks', 'Dose,  mg/d') | 20 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 67 |\n| ('Delivery', 'Dose,  mg/d') | NA |\n| ('Delivery', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 20 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 95 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 20 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 63 |", "| Parameter type | S-sertraline_0 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | NA |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('30 Weeks', 'Dose,  mg/d') | 50 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 27 |\n| ('36 Weeks', 'Dose,  mg/d') | NA |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Delivery', 'Dose,  mg/d') | 50 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 11 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | 50 |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 27 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 50 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 21 |", "| Parameter type | N-desmethylsertraline_0 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 44 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 26 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 52 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 50 |", "| Parameter type | S-sertraline_1 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 50 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 10 |\n| ('30 Weeks', 'Dose,  mg/d') | 50 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 20 |\n| ('36 Weeks', 'Dose,  mg/d') | 100 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 16 |\n| ('Delivery', 'Dose,  mg/d') | 50 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 7 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 50 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 22 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | 50 |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 15 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | N-desmethylsertraline_1 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 18 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 40 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 66 |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 32 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 41 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 29 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | S-sertraline_2 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 50 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 4 |\n| ('30 Weeks', 'Dose,  mg/d') | 75 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 10 |\n| ('36 Weeks', 'Dose,  mg/d') | 100 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 39 |\n| ('Delivery', 'Dose,  mg/d') | NA |\n| ('Delivery', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 200 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 21 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 200 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 21 |", "| Parameter type | N-desmethylsertraline_2 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 16 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 17 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 74 |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 59 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 29 |", "| Parameter type | S-sertraline_3 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | NA |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('30 Weeks', 'Dose,  mg/d') | 150 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 17 |\n| ('36 Weeks', 'Dose,  mg/d') | 200 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 43 |\n| ('Delivery', 'Dose,  mg/d') | 200 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 32 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 200 |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 39 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | N-desmethylsertraline_3 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 141 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 147 |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 120 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 149 |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | S-sertraline_4 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 100 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 93 |\n| ('30 Weeks', 'Dose,  mg/d') | 100 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 45 |\n| ('36 Weeks', 'Dose,  mg/d') | NA |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Delivery', 'Dose,  mg/d') | 100 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 28 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | 100 |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 62 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | N-desmethylsertraline_4 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 72 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 92 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 59 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 74 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | NA |", "| Parameter type | S-sertraline_5 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') | 50 |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 32 |\n| ('30 Weeks', 'Dose,  mg/d') | 75 |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 18 |\n| ('36 Weeks', 'Dose,  mg/d') | 75 |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 23 |\n| ('Delivery', 'Dose,  mg/d') | 100 |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 32 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') | NA |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | 100 |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 83 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 125 |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 99 |", "| Parameter type | N-desmethylsertraline_5 |\n| --- | --- |\n| ('20 Weeks', 'Dose,  mg/d') |  |\n| ('20 Weeks', 'Plasma  Level,  ng/mL') | 52 |\n| ('30 Weeks', 'Dose,  mg/d') |  |\n| ('30 Weeks', 'Plasma  Level,  ng/mL') | 77 |\n| ('36 Weeks', 'Dose,  mg/d') |  |\n| ('36 Weeks', 'Plasma  Level,  ng/mL') | 73 |\n| ('Delivery', 'Dose,  mg/d') |  |\n| ('Delivery', 'Plasma  Level,  ng/mL') | 97 |\n| ('Postpartum 2 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | NA |\n| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') |  |\n| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 128 |\n| ('Postpartum  12 Weeks', 'Dose,  mg/d') |  |\n| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 183 |"]"""

@pytest.fixture(scope="module")
def md_table_patient_18426260_table_0():
    return """"""

@pytest.fixture(scope="module")
def md_table_aligned_18426260_table_0():
    return """
| Parameter type | 1 | S-citalopram_0 | R-citalopram_0 | 2 | S-citalopram_1 | R-citalopram_1 | 3b | S-citalopram_2 | R-citalopram_2 | 4 | S-citalopram_3 | 5 | S-citalopram_4 | 6 | S-sertraline_0 | N-desmethylsertraline_0 | 7 | S-sertraline_1 | N-desmethylsertraline_1 | 8 | S-sertraline_2 | N-desmethylsertraline_2 | 9 | S-sertraline_3 | N-desmethylsertraline_3 | 10 | S-sertraline_4 | N-desmethylsertraline_4 | 11 | S-sertraline_5 | N-desmethylsertraline_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ('20 Weeks', 'Dose,  mg/d') | 1 | 40 |  | 2 | 20 |  | 3b | 30 |  | 4 | 10 | 5 | 20 | 6 | NA |  | 7 | 50 |  | 8 | 50 |  | 9 | NA |  | 10 | 100 |  | 11 | 50 |  |
| ('20 Weeks', 'Plasma  Level,  ng/mL') | 1 | 32 | 62 | 2 | 14 | 24 | 3b | 19 | 41 | 4 | 17 | 5 | 58 | 6 | NA | NA | 7 | 10 | 18 | 8 | 4 | 16 | 9 | NA | NA | 10 | 93 | 72 | 11 | 32 | 52 |
| ('30 Weeks', 'Dose,  mg/d') | 1 | 40 |  | 2 | 20 |  | 3b | 30 |  | 4 | 10 | 5 | 20 | 6 | 50 |  | 7 | 50 |  | 8 | 75 |  | 9 | 150 |  | 10 | 100 |  | 11 | 75 |  |
| ('30 Weeks', 'Plasma  Level,  ng/mL') | 1 | 40 | 64 | 2 | 14 | 24 | 3b | 9 | 22 | 4 | 10 | 5 | 70 | 6 | 27 | 44 | 7 | 20 | 40 | 8 | 10 | 17 | 9 | 17 | 141 | 10 | 45 | 92 | 11 | 18 | 77 |
| ('36 Weeks', 'Dose,  mg/d') | 1 | 50 |  | 2 | 40 |  | 3b | 20 |  | 4 | 10 | 5 | 20 | 6 | NA |  | 7 | 100 |  | 8 | 100 |  | 9 | 200 |  | 10 | NA |  | 11 | 75 |  |
| ('36 Weeks', 'Plasma  Level,  ng/mL') | 1 | 25 | 41 | 2 | 18 | 36 | 3b | 5 | 14 | 4 | 13 | 5 | 67 | 6 | NA | NA | 7 | 16 | 66 | 8 | 39 | 74 | 9 | 43 | 147 | 10 | NA | NA | 11 | 23 | 73 |
| ('Delivery', 'Dose,  mg/d') | 1 | 50 |  | 2 | NA | NA | 3b | … | … | 4 | 10 | 5 | NA | 6 | 50 |  | 7 | 50 |  | 8 | NA |  | 9 | 200 |  | 10 | 100 |  | 11 | 100 |  |
| ('Delivery', 'Plasma  Level,  ng/mL') | 1 | 15 | 24 | 2 | NA | NA | 3b | … | … | 4 | 14 | 5 | NA | 6 | 11 | 26 | 7 | 7 | 32 | 8 | NA | NA | 9 | 32 | 120 | 10 | 28 | 59 | 11 | 32 | 97 |
| ('Postpartum 2 Weeks', 'Dose,  mg/d') | 1 | NA | NA | 2 | 40 |  | 3b | … | … | 4 | 10 | 5 | 20 | 6 | NA |  | 7 | 50 |  | 8 | 200 |  | 9 | 200 |  | 10 | NA |  | 11 | NA |  |
| ('Postpartum 2 Weeks', 'Plasma  Level,  ng/mL') | 1 | NA | NA | 2 | 54 | 101 | 3b | … | … | 4 | 24 | 5 | 95 | 6 | NA | NA | 7 | 22 | 41 | 8 | 21 | 59 | 9 | 39 | 149 | 10 | NA | NA | 11 | NA | NA |
| ('Postpartum  4 to 6 Weeksa', 'Dose,  mg/d') | 1 | NA | NA | 2 | NA | NA | 3b | … | … | 4 | NA | 5 | NA | 6 | 50 |  | 7 | 50 |  | 8 | NA |  | 9 | NA |  | 10 | 100 |  | 11 | 100 |  |
| ('Postpartum  4 to 6 Weeksa', 'Plasma  Level,  ng/mL') | 1 | NA | NA | 2 | NA | NA | 3b | … | … | 4 | NA | 5 | NA | 6 | 27 | 52 | 7 | 15 | 29 | 8 | NA | NA | 9 | NA | NA | 10 | 62 | 74 | 11 | 83 | 128 |
| ('Postpartum  12 Weeks', 'Dose,  mg/d') | 1 | 50 |  | 2 | 40 |  | 3b | … | … | 4 | NA | 5 | 20 | 6 | 50 |  | 7 | NA |  | 8 | 200 |  | 9 | NA |  | 10 | NA |  | 11 | 125 |  |
| ('Postpartum  12 Weeks', 'Plasma  Level,  ng/mL') | 1 | 18 | 36 | 2 | 27 | 75 | 3b | … | … | 4 | NA | 5 | 63 | 6 | 21 | 50 | 7 | NA | NA | 8 | 21 | 29 | 9 | NA | NA | 10 | NA | NA | 11 | 99 | 183 |
"""

@pytest.fixture(scope="module")
def md_table_patient_refined_18426260_table_0():
    return """| Population | Pregnancy stage | Pediatric/Gestational age | Subject N |
| --- | --- | --- | --- |
| Maternal | Trimester 2 | N/A | 40 |
| Maternal | Trimester 3 | N/A | 40 |
| Maternal | Trimester 3 | N/A | 50 |
| Maternal | Delivery | N/A | 50 |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | 50 |
| Maternal | Trimester 2 | N/A | 62 |
| Maternal | Trimester 3 | N/A | 64 |
| Maternal | Trimester 3 | N/A | 41 |
| Maternal | Delivery | N/A | 24 |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | 36 |
| Maternal | Trimester 2 | N/A | N/A |
| Maternal | Trimester 3 | N/A | 50 |
| Maternal | Trimester 3 | N/A | N/A |
| Maternal | Delivery | N/A | 50 |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | 50 |
| Maternal | Postpartum | N/A | 50 |
| Maternal | Trimester 2 | N/A | N/A |
| Maternal | Trimester 3 | N/A | 44 |
| Maternal | Trimester 3 | N/A | N/A |
| Maternal | Delivery | N/A | 26 |
| Maternal | Postpartum | N/A | N/A |
| Maternal | Postpartum | N/A | 52 |
| Maternal | Postpartum | N/A | 50 |
"""

@pytest.fixture(scope="module")
def md_table_patine_18426260_table_0():
    return """
| Population | Pregnancy stage | Subject N |
| --- | --- | --- |
| S-citalopram | 20 Weeks | 40 |
| S-citalopram | 30 Weeks | 40 |
| S-citalopram | 36 Weeks | 50 |
| S-citalopram | Delivery | 50 |
| S-citalopram | Postpartum 2 Weeks | N/A |
| S-citalopram | Postpartum 4 to 6 Weeks | N/A |
| S-citalopram | Postpartum 12 Weeks | 50 |
| R-citalopram | 20 Weeks | 62 |
| R-citalopram | 30 Weeks | 64 |
| R-citalopram | 36 Weeks | 41 |
| R-citalopram | Delivery | 24 |
| R-citalopram | Postpartum 2 Weeks | N/A |
| R-citalopram | Postpartum 4 to 6 Weeks | N/A |
| R-citalopram | Postpartum 12 Weeks | 36 |
| S-sertraline | 20 Weeks | N/A |
| S-sertraline | 30 Weeks | 50 |
| S-sertraline | 36 Weeks | N/A |
| S-sertraline | Delivery | 50 |
| S-sertraline | Postpartum 2 Weeks | N/A |
| S-sertraline | Postpartum 4 to 6 Weeks | 50 |
| S-sertraline | Postpartum 12 Weeks | 50 |
| N-desmethylsertraline | 20 Weeks | N/A |
| N-desmethylsertraline | 30 Weeks | 44 |
| N-desmethylsertraline | 36 Weeks | N/A |
| N-desmethylsertraline | Delivery | 26 |
| N-desmethylsertraline | Postpartum 2 Weeks | N/A |
| N-desmethylsertraline | Postpartum 4 to 6 Weeks | 52 |
| N-desmethylsertraline | Postpartum 12 Weeks | 50 |
"""

@pytest.fixture(scope="module")
def caption_18426260_table_0():
    return """Drug Dose and Plasma Levels in Pregnancy and Postpartum\n aValues were available at 4 to 6 weeks postpartum for breast-feeding mother-infant pairs only.
bSubject discontinued medication at delivery.
Abbreviation: NA = not available."""


## ==================================================
## 29100749
## ==================================================

@pytest.fixture(scope="module")
def title_29100749():
    return paper_title_29100749

@pytest.fixture(scope="module")
def abstract_29100749():
    return paper_abstract_29100749

@pytest.fixture(scope="module")
def curated_table_29100749():
    return curated_data_29100749

@pytest.fixture(scope="module")
def source_table_29100749_table_2():
    return data_source_table_29100749_table_2

@pytest.fixture(scope="module")
def md_table_aligned_29100749_table_2():
    return data_md_table_aligned_29100749_table_2

@pytest.fixture(scope="module")
def col_mapping_29100749_table_2():
    return data_col_mapping_29100749_table_2

@pytest.fixture(scope="module")
def caption_29100749_table_2():
    return data_caption_29100749_table_2

@pytest.fixture(scope="module")
def md_table_list_29100749_table_2():
    return data_md_table_list_29100749_table_2

@pytest.fixture(scope="module")
def md_table_drug_29100749_table_2():
    return data_md_table_drug_29100749_table_2

## ==================================================
## 32635742
## ==================================================

@pytest.fixture(scope="module")
def title_32635742():
    return paper_title_32635742

@pytest.fixture(scope="module")
def abstract_32635742():
    return paper_abstract_32635742

@pytest.fixture(scope="module")
def caption_32635742_table_0():
    return data_caption_32635742_table_0

@pytest.fixture(scope="module")
def md_table_32635742_table_0():
    return data_md_table_32635742_table_0

@pytest.fixture(scope="module")
def full_text_32635742():
    fn = Path(__file__).parent / "data" / "32635742_full_text.txt"
    with open(fn, "r") as f:
        return f.read()


# ============================================================================================
# utils
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
        step_reasoning_process: Optional[str | list[str]] = None,
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

@pytest.fixture(scope="module")
def pmid_db():
    pmid_db = PMIDDB(Path(__file__).parent / "data" / "pmid_info.db")
    return pmid_db
