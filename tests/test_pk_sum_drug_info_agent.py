import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek

from extractor.agents.pk_sum_drug_info_agent import (
    PKSumDrugInfoAgent, 
    DrugInfoResult,
    DRUG_INFO_PROMPT,
    INSTRUCTION_PROMPT,
)
from extractor.agents.pk_sum_patient_info_agent import (
    PATIENT_INFO_PROMPT,
    PatientInfoResult,
)
from extractor.agents.pk_sum_param_type_align_agent import (
    PARAMETER_TYPE_ALIGN_PROMPT,
    ParameterTypeAlignResult,
    post_process_parameter_type_align,
)
from extractor.agents.pk_sum_individual_data_del_agent import (
    INDIVIDUAL_DATA_DEL_PROMPT,
    IndividualDataDelResult,
    post_process_individual_del_result,
)
from extractor.agents.pk_sum_header_categorize_agent import (
    HEADER_CATEGORIZE_PROMPT,
    HeaderCategorizeResult,
    HeaderCategorizeJsonSchema,
    post_process_validate_categorized_result,
    get_header_categorize_prompt,
)
from extractor.agents.pk_sum_unit_extract_agent import (
    get_unit_extraction_prompt,
    post_process_validate_matched_tuple,
    UnitExtractionResult,
)
from extractor.agents.pk_sum_param_value_agent import (
    ParameterValueResult,
    get_parameter_value_prompt,
    post_process_matched_list,
)

from extractor.agents.pk_sum_common_agent import PKSumCommonAgent
from extractor.agents.agent_utils import display_md_table
from TabFuncFlow.utils.table_utils import single_html_table_to_markdown
from TabFuncFlow.pipelines.p_pk_summary import p_pk_summary
from TabFuncFlow.steps_pk_summary.s_pk_get_col_mapping import s_pk_get_col_mapping
from TabFuncFlow.steps_pk_summary.s_pk_extract_patient_info import extract_integers

load_dotenv()

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
caption_and_footnote = "Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0-∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life."

md_table_aligned="""
"Parameter type" | "N" | "Range" | "Mean ± s.d." | "Median" |
| --- | --- | --- | --- | --- |
Cmax(ng/mL) | 15 | 29.3–209.6 | 56.1 ± 44.9 | 42.2 |
AUC0−∞ | 15 | 253.3–3202.5 | 822.5 ± 706.1 | 601.5 |
CL(mL/min/kg) | 15 | 3.33–131.50 | 49.33 ± 30.83 | 41.50 |
CL(mL/min/m) | 15 | 5.5–67.5 | 31.95 ± 13.99 | 32.34 |
Vdz(L/kg) | 15 | 0.33–4.05 | 1.92 ± 0.84 | 1.94 |
T1/2(hr) | 15 | 9.5–47.0 | 20.5 ± 10.2 | 18.1 |
"""
col_mapping = {
    "Parameter type": "Parameter type",
    "N": "Uncategorized",
    "Range": "Parameter value",
    "Mean ± s.d.": "Parameter value",
    "Median": "Parameter value"
}
md_table_list = ["""
"Parameter type" | "Range" |
| --- | --- |
Cmax(ng/mL) | 29.3–209.6 |
AUC0−∞ | 253.3–3202.5 |
CL(mL/min/kg) | 3.33–131.50 |
CL(mL/min/m) | 5.5–67.5 |
Vdz(L/kg) | 0.33–4.05 |
T1/2(hr) | 9.5–47.0 |
""", """
"Parameter type" | "Mean ± s.d." |
| --- | --- |
Cmax(ng/mL) | 56.1 ± 44.9 |
AUC0−∞ | 822.5 ± 706.1 |
CL(mL/min/kg) | 49.33 ± 30.83 |
CL(mL/min/m) | 31.95 ± 13.99 |
Vdz(L/kg) | 1.92 ± 0.84 |
T1/2(hr) | 20.5 ± 10.2 |
""", """
"Parameter type" | "Median" |
| --- | --- |
Cmax(ng/mL) | 42.2 |
AUC0−∞ | 601.5 |
CL(mL/min/kg) | 41.50 |
CL(mL/min/m) | 32.34 |
Vdz(L/kg) | 1.94 |
"""]

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

@pytest.mark.skip()
def test_PKSumCommonAgent_patient_info():
    md_table = single_html_table_to_markdown(html_content)
    int_list = extract_integers(md_table + caption_and_footnote)
    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=PATIENT_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption_and_footnote,
            int_list=int_list,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=PatientInfoResult,
    )
    print(res)

@pytest.mark.skip()
def test_PKSumCommonAgent_drug_info():
    the_obj = DrugInfoResult.model_json_schema()
    print(the_obj)

    md_table = single_html_table_to_markdown(html_content=html_content)
    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=DRUG_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table), 
            caption=caption_and_footnote
         ),
         instruction_prompt=INSTRUCTION_PROMPT,
         schema=DrugInfoResult,
    )
    print(res)

@pytest.mark.skip()
def test_PKSumCommonAgent_ind_data_del():
    md_table = single_html_table_to_markdown(html_content)
    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=INDIVIDUAL_DATA_DEL_PROMPT.format(
            processed_md_table=display_md_table(md_table),
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=IndividualDataDelResult,
        post_process=post_process_individual_del_result,
        md_table=md_table,
    )
    print(res)

@pytest.mark.skip()
def test_PKSumCommonAgent_param_type_align():
    md_table = single_html_table_to_markdown(html_content)
    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=PARAMETER_TYPE_ALIGN_PROMPT.format(
            md_table_summary=md_table,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=ParameterTypeAlignResult,
        post_process=post_process_parameter_type_align,
        md_table=md_table,
    )
    print(res)

@pytest.mark.skip()
def test_PKSumCommonAgent_header_categorize():
    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=get_header_categorize_prompt(md_table_aligned),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=HeaderCategorizeJsonSchema, # HeaderCategorizeResult,
        post_process=post_process_validate_categorized_result,
        md_table=md_table_aligned,
    )
    print(res)

@pytest.mark.skip()
def test_PKSumCommonAgent_unit_extraction():
    # test schema
    schema_obj = UnitExtractionResult.model_json_schema()
    print(schema_obj)

    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    res = agent.go(
        system_prompt=get_unit_extraction_prompt(
            md_table_aligned=md_table_aligned,
            md_sub_table=md_table_list[0],
            col_mapping=col_mapping,
            caption=caption_and_footnote,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=UnitExtractionResult,
        post_process=post_process_validate_matched_tuple,
        md_table=md_table_list[0],
    )
    assert len(res) == 2 # matched tuple (parameter types list, parameter valus list)

def test_PKSumCommonAgent_param_value_extraction():
    schema_obj = ParameterValueResult.model_json_schema()
    print(schema_obj)

    llm = get_openai()
    agent = PKSumCommonAgent(llm=llm)
    for md in md_table_list:
        res = agent.go(
            system_prompt=get_parameter_value_prompt(
                md_table_aligned=md_table_aligned,
                md_table_aligned_with_1_param_type_and_value=md,
                caption=caption_and_footnote,
            ),
            instruction_prompt=INSTRUCTION_PROMPT,
            schema=ParameterValueResult,
            post_process=post_process_matched_list,
        )
        print(res)

@pytest.mark.skip()
def test_p_pk_summary_drug_info():
    # md_table = single_html_table_to_markdown(html_content)
    # drug_info = p_pk_summary(
    #     md_table=md_table,
    #     description=caption_and_footnote,
    #     llm="chatgpt_4o",
    # )
    # print(drug_info)

    md_table_aligned = "| Parameter type | N | Range | Mean ± s.d. | Median |\n| --- | --- | --- | --- | --- |\n| Cmax(ng/mL) | 15 | 29.3-209.6 | 56.1 ± 44.9 | 42.2 |\n| AUC0-∞ | 15 | 253.3-3202.5 | 822.5 ± 706.1 | 601.5 |\n| CL(mL/min/kg) | 15 | 3.33-131.50 | 49.33 ± 30.83 | 41.50 |\n| CL(mL/min/m) | 15 | 5.5-67.5 | 31.95 ± 13.99 | 32.34 |\n| Vdz(L/kg) | 15 | 0.33-4.05 | 1.92 ± 0.84 | 1.94 |\n| T1/2(hr) | 15 | 9.5-47.0 | 20.5 ± 10.2 | 18.1 |"
    res = s_pk_get_col_mapping(md_table_aligned, model_name="chatgpt_4o")
    print(res)

