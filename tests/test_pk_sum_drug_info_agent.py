import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from TabFuncFlow.operations.f_split_by_cols import f_split_by_cols
from extractor.agents.pk_summary.pk_sum_drug_info_agent import (
    DrugInfoResult,
    DRUG_INFO_PROMPT,
    INSTRUCTION_PROMPT,
)
from extractor.agents.pk_summary.pk_sum_patient_info_agent import (
    PATIENT_INFO_PROMPT,
    PatientInfoResult,
    post_process_convert_patient_info_to_md_table
)
from extractor.agents.pk_summary.pk_sum_patient_info_refine_agent import (
    get_patient_info_refine_prompt,
    PatientInfoRefinedResult,
    post_process_refined_patient_info,
)
from extractor.agents.pk_summary.pk_sum_param_type_align_agent import (
    PARAMETER_TYPE_ALIGN_PROMPT,
    ParameterTypeAlignResult,
    post_process_parameter_type_align,
)
from extractor.agents.pk_summary.pk_sum_individual_data_del_agent import (
    INDIVIDUAL_DATA_DEL_PROMPT,
    IndividualDataDelResult,
    post_process_individual_del_result,
)
from extractor.agents.pk_summary.pk_sum_header_categorize_agent import (
    HEADER_CATEGORIZE_PROMPT,
    HeaderCategorizeResult,
    HeaderCategorizeJsonSchema,
    post_process_validate_categorized_result,
    get_header_categorize_prompt,
)
from extractor.agents.pk_summary.pk_sum_param_type_unit_extract_agent import (
    get_param_type_unit_extraction_prompt,
    pre_process_param_type_unit,
    post_process_validate_matched_tuple,
    ParamTypeUnitExtractionResult,
)
from extractor.agents.pk_summary.pk_sum_param_value_agent import (
    ParameterValueResult,
    get_parameter_value_prompt,
    post_process_matched_list,
)

from extractor.agents.pk_summary.pk_sum_time_unit_agent import (
    TimeAndUnitResult,
    get_time_and_unit_prompt,
    post_process_time_and_unit,
)

from extractor.agents.pk_summary.pk_sum_split_by_col_agent import (
    get_split_by_columns_prompt,
    SplitByColumnsResult,
    post_process_split_by_columns,
)

from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgent,
    RetryException,
)
from extractor.agents.agent_utils import display_md_table
from TabFuncFlow.utils.table_utils import dataframe_to_markdown, fix_col_name, markdown_to_dataframe, single_html_table_to_markdown
from TabFuncFlow.pipelines.p_pk_summary import p_pk_summary
from TabFuncFlow.steps_pk_summary.s_pk_get_col_mapping import s_pk_get_col_mapping
from TabFuncFlow.steps_pk_summary.s_pk_extract_patient_info import extract_integers

load_dotenv()

md_table_post_processed = """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Subject N | Parameter type | Unit | Value | Summary Statistics | Variation type | Variation value | Interval type | Lower limit | High limit | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | Plasma | N/A | N/A | 15 | Apparent volume of distribution (Vdz) | L/kg | 1.92 | Mean | Standard Deviation (SD) | 0.84 | Range | 0.33 | 4.05 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Apparent volume of distribution (Vdz) | L/kg | 1.94 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Area-under-the-curve to infinity (AUC0−∞) | ng/mL*hr | 822.5 | Mean | Standard Deviation (SD) | 706.1 | Range | 253.3 | 3202.5 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Area-under-the-curve to infinity (AUC0−∞) | ng/mL*hr | 601.5 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Clearance (CL) | mL/min/kg | 49.33 | Mean | Standard Deviation (SD) | 30.83 | Range | 3.33 | 131.50 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Clearance (CL) | mL/min/kg | 41.50 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Clearance (CL) | mL/min/m | 31.95 | Mean | Standard Deviation (SD) | 13.99 | Range | 5.5 | 67.5 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Clearance (CL) | mL/min/m | 32.34 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Half-life (T1/2) | hr | 20.5 | Mean | Standard Deviation (SD) | 10.2 | Range | 9.5 | 47.0 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Half-life (T1/2) | hr | 18.1 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Maximum concentration (Cmax) | ng/mL | 56.1 | Mean | Standard Deviation (SD) | 44.9 | Range | 29.3 | 209.6 | N/A |
| N/A | N/A | Plasma | N/A | N/A | 15 | Maximum concentration (Cmax) | ng/mL | 42.2 | Median | N/A | N/A | N/A | N/A | N/A | N/A |
"""

@pytest.mark.skip()
def test_PKSumCommonAgent_patient_info(llm, md_table, caption):
    int_list = extract_integers(md_table + caption)
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=PATIENT_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table),
            caption=caption,
            int_list=int_list,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=PatientInfoResult,
        post_process=post_process_convert_patient_info_to_md_table,
    )
    assert isinstance(res, PatientInfoResult)
    assert type(processed_res) == str

@pytest.mark.skip()
def test_PKSumCommonAgent_patient_info_refine(llm, md_table, md_table_patient, caption):
    int_list = extract_integers(md_table_patient + caption)
    agent = PKSumCommonAgent(llm=llm)
    res, process_res, token_usage = agent.go(
        system_prompt=get_patient_info_refine_prompt(
            md_table, md_table_patient, caption,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=PatientInfoRefinedResult,
        post_process=post_process_refined_patient_info,
        md_table_patient=md_table_patient,
    )
    assert isinstance(res, PatientInfoRefinedResult)
    assert type(process_res) == str

@pytest.mark.skip()
def test_PKSumCommonAgent_drug_info(llm, md_table, caption):
    the_obj = DrugInfoResult.model_json_schema()
    print(the_obj)
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=DRUG_INFO_PROMPT.format(
            processed_md_table=display_md_table(md_table), 
            caption=caption,
         ),
         instruction_prompt=INSTRUCTION_PROMPT,
         schema=DrugInfoResult,
    )
    assert isinstance(res, DrugInfoResult)
    assert processed_res is None

@pytest.mark.skip()
def test_PKSumCommonAgent_ind_data_del(llm, md_table):
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=INDIVIDUAL_DATA_DEL_PROMPT.format(
            processed_md_table=display_md_table(md_table),
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=IndividualDataDelResult,
        post_process=post_process_individual_del_result,
        md_table=md_table,
    )
    assert isinstance(res, IndividualDataDelResult)
    assert type(processed_res) == str

@pytest.mark.skip()
def test_PKSumCommonAgent_param_type_align(llm, md_table):
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=PARAMETER_TYPE_ALIGN_PROMPT.format(
            md_table_summary=md_table,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=ParameterTypeAlignResult,
        post_process=post_process_parameter_type_align,
        md_table_summary=md_table,
    )
    assert isinstance(res, ParameterTypeAlignResult)
    assert type(processed_res) == str

@pytest.mark.skip()
def test_PKSumCommonAgent_header_categorize(llm, md_table_aligned):
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=get_header_categorize_prompt(md_table_aligned),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=HeaderCategorizeJsonSchema, # HeaderCategorizeResult,
        post_process=post_process_validate_categorized_result,
        md_table_aligned=md_table_aligned,
    )
    assert isinstance(res, dict)
    assert isinstance(processed_res, HeaderCategorizeResult)

@pytest.mark.skip()
def test_PKSumCommonAgent_unit_extraction(llm, md_table_aligned, md_table_list, col_mapping, caption):
    # test schema
    schema_obj = ParamTypeUnitExtractionResult.model_json_schema()
    print(schema_obj)

    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=get_param_type_unit_extraction_prompt(
            md_table_aligned=md_table_aligned,
            md_sub_table=md_table_list[0],
            col_mapping=col_mapping,
            caption=caption,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=ParamTypeUnitExtractionResult,
        pre_process=pre_process_param_type_unit,
        post_process=post_process_validate_matched_tuple,
        md_table=md_table_list[0],
        col_mapping=col_mapping,
    )
    assert isinstance(res, ParamTypeUnitExtractionResult)
    assert type(processed_res) == tuple
    assert len(processed_res) == 2 # matched tuple (parameter types list, parameter valus list)

@pytest.mark.skip()
def test_PKSumCommonAgent_param_value_extraction(llm, md_table_aligned, caption, md_table_list):
    schema_obj = ParameterValueResult.model_json_schema()
    print(schema_obj)

    agent = PKSumCommonAgent(llm=llm)
    for md in md_table_list:
        res, processed_res, token_usage = agent.go(
            system_prompt=get_parameter_value_prompt(
                md_table_aligned=md_table_aligned,
                md_table_aligned_with_1_param_type_and_value=md,
                caption=caption,
            ),
            instruction_prompt=INSTRUCTION_PROMPT,
            schema=ParameterValueResult,
            post_process=post_process_matched_list,
            expected_rows=markdown_to_dataframe(md).shape[0],
        )
        assert isinstance(res, ParameterValueResult)
        assert type(processed_res) == str

@pytest.mark.skip()
def test_PKSumCommonAgent_time_and_unit_extraction(llm, md_table_aligned, caption):
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=get_time_and_unit_prompt(
            md_table_aligned=md_table_aligned,
            md_table_post_processed=md_table_post_processed,
            caption=caption,
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TimeAndUnitResult,
        post_process=post_process_time_and_unit,
        md_table_post_processed=md_table_post_processed,
    )
    assert isinstance(res, TimeAndUnitResult)
    assert type(processed_res) == str
    assert token_usage["total_tokens"] > 0

# @pytest.mark.skip()
def test_PKSumCommonAgent_split_by_col(llm, col_mapping, md_table_aligned):
    agent = PKSumCommonAgent(llm=llm)
    res, processed_res, token_usage = agent.go(
        system_prompt=get_split_by_columns_prompt(
            md_table=md_table_aligned, col_mapping=col_mapping
        ),
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=SplitByColumnsResult,
        post_process=post_process_split_by_columns,
        md_table_aligned=md_table_aligned,
    )
    assert isinstance(res, SplitByColumnsResult)

@pytest.mark.skip()
def test_p_pk_summary_drug_info(md_table_aligned):
    # md_table = single_html_table_to_markdown(html_content)
    # drug_info = p_pk_summary(
    #     md_table=md_table,
    #     description=caption,
    #     llm="chatgpt_4o",
    # )
    # print(drug_info)

    res = s_pk_get_col_mapping(md_table_aligned, model_name="chatgpt_4o")
    print(res)

@pytest.mark.skip()
def test_post_process_split_by_columns(md_table_aligned):
    col_groups = [['Parameter type', 'N', 'Range', 'Mean ± s.d.', 'Median']]
    processed_col_groups = [[fix_col_name(item, md_table_aligned) for item in group] for group in col_groups]
    df_table = f_split_by_cols(processed_col_groups, markdown_to_dataframe(md_table_aligned))

    return_md_table_list = [dataframe_to_markdown(d) for d in df_table]