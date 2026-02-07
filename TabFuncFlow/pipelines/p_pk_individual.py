from TabFuncFlow.steps_pk_individual.s_pk_delete_summary import *
from TabFuncFlow.steps_pk_individual.s_pk_align_parameter import *
from TabFuncFlow.steps_pk_individual.s_pk_extract_drug_info import *
from TabFuncFlow.steps_pk_individual.s_pk_extract_patient_info import *
from TabFuncFlow.steps_pk_individual.s_pk_extract_time_and_unit import *
from TabFuncFlow.steps_pk_individual.s_pk_get_col_mapping import *
from TabFuncFlow.steps_pk_individual.s_pk_get_parameter_type_unit_value import *
from TabFuncFlow.steps_pk_individual.s_pk_match_drug_info import *
from TabFuncFlow.steps_pk_individual.s_pk_match_patient_info import *
from TabFuncFlow.steps_pk_individual.s_pk_split_by_cols import *
# from TabFuncFlow.steps_pk_individual.s_pk_get_parameter_value import *
from TabFuncFlow.steps_pk_individual.s_pk_refine_patient_info import *
import re
import itertools
from difflib import get_close_matches


def clean_llm_reasoning(text: str) -> str:
    """
    Cleans the LLM inference string by removing content after the last occurrence
    of '<<' or '[[END]]', keeping only the portion up to the last preceding period.
    If the last period is too close to the previous newline, truncate at the newline instead.
    Additionally, if the text contains two occurrences of '```', truncate before the first occurrence.
    Ensures the output ends with a newline.

    :param text: The input inference string.
    :return: The cleaned inference string, ensuring it ends with a newline.
    """
    # Check for two occurrences of '```'
    first_triple_backtick = text.find("```")
    second_triple_backtick = text.find("```", first_triple_backtick + 3)

    if first_triple_backtick != -1 and second_triple_backtick != -1:
        result = text[:first_triple_backtick]
        return result if result.endswith("\n") else result + "\n"

    # Try finding '<<' first
    last_double_angle = text.rfind("<<")

    if last_double_angle != -1:
        cutoff_index = last_double_angle
    else:
        # If '<<' is not found, look for '[[END]]'
        last_end_marker = text.rfind("[[END]]")
        if last_end_marker != -1:
            cutoff_index = last_end_marker
        else:
            # If neither is found, return original text (ensuring newline)
            return text if text.endswith("\n") else text + "\n"

    # Find the last period before the cutoff index
    last_period = text.rfind(".", 0, cutoff_index)

    if last_period != -1:
        # Find the last newline before the period
        last_newline = text.rfind("\n", 0, last_period)

        # If newline is found and period is too close to it, truncate at newline
        if last_newline != -1 and (last_period - last_newline) < 5:
            result = text[:last_newline + 1]
        else:
            result = text[:last_period + 1]
    else:
        result = text  # No period found, return full text

    # Ensure the output ends with a newline
    return result if result.endswith("\n") else result + "\n"


def p_pk_individual(md_table, description, llm="gemini_15_pro", max_retries=10, initial_wait=2, use_color=True, clean_reasoning=False):
    """
    PK Individual Pipeline 250312
    Summarizes pharmacokinetic (PK) data from a given markdown table.

    :param md_table: The markdown representation of a SINGLE HTML table containing PK data.
    :param description: Additional contextual information, including captions, footnotes, etc.
    :param llm: The language model to use for processing ("gemini_15_pro" or "chatgpt_4o").
    :param max_retries: Maximum number of retries before failing.
    :param initial_wait: Initial delay before retrying (in seconds), which doubles on each failure.
    :param use_color: Better-looking print output.
    :param clean_reasoning: When printing output, hide the parsing-related parts in the reasoning.
    :return:
    """
    if use_color:
        # COLOR_START = "\033[31m"  # red
        COLOR_START = "\033[32m"  # green
        # COLOR_START = "\033[33m"  # yellow
        COLOR_END = "\033[0m"
    else:
        COLOR_START = ""
        COLOR_END = ""

    step_list = []
    res_list = []
    content_list = []
    content_list_clean = []
    usage_list = []
    truncated_list = []
    """
    Step 0: Pre-launch Inspection
    """
    print("=" * 64)
    step_name = "Pre-launch Inspection"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    print("Markdown Table:")
    print(display_md_table(md_table))
    print("Description:")
    print(description)
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 1: Drug Information Extraction
    """
    print("=" * 64)
    step_name = "Drug Information Extraction"
    print(COLOR_START+step_name+COLOR_END)
    drug_info = s_pk_extract_drug_info(md_table, description, llm, max_retries, initial_wait)
    if drug_info is None:
        return None
    md_table_drug, res_drug, content_drug, usage_drug, truncated_drug = drug_info
    step_list.append(step_name)
    res_list.append(res_drug)
    content_list.append(content_drug)
    content_list_clean.append(clean_llm_reasoning(content_drug))
    usage_list.append(usage_drug)
    truncated_list.append(truncated_drug)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_drug))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 2: Population Information Extraction
    """
    print("=" * 64)
    step_name = "Population Information Extraction"
    print(COLOR_START+step_name+COLOR_END)
    patient_info = s_pk_extract_patient_info(md_table, description, llm, max_retries, initial_wait)
    if patient_info is None:
        return None
    md_table_patient, res_patient, content_patient, usage_patient, truncated_patient = patient_info
    step_list.append(step_name)
    res_list.append(res_patient)
    content_list.append(content_patient)
    content_list_clean.append(clean_llm_reasoning(content_patient))
    usage_list.append(usage_patient)
    truncated_list.append(truncated_patient)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_patient))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 3: Population Information Refinement
    """
    print("=" * 64)
    step_name = "Population Information Refinement"
    print(COLOR_START+step_name+COLOR_END)
    patient_info_refined = s_pk_refine_patient_info(md_table, description, md_table_patient, llm, max_retries, initial_wait)
    if patient_info_refined is None:
        return None
    md_table_patient_refined, res_patient_refined, content_patient_refined, usage_patient_refined, truncated_patient_refined = patient_info_refined
    step_list.append(step_name)
    res_list.append(res_patient_refined)
    content_list.append(content_patient_refined)
    content_list_clean.append(clean_llm_reasoning(content_patient_refined))
    usage_list.append(usage_patient_refined)
    truncated_list.append(truncated_patient_refined)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_patient_refined))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 4: Summary Data Deletion
    """
    print("=" * 64)
    step_name = "Summary Data Deletion"
    print(COLOR_START+step_name+COLOR_END)
    individual_only_info = s_pk_delete_summary(md_table, llm, max_retries, initial_wait)
    if individual_only_info is None:
        return None
    md_table_individual, res_individual, content_individual, usage_individual, truncated_individual = individual_only_info
    step_list.append(step_name)
    res_list.append(res_individual)
    content_list.append(content_individual)
    content_list_clean.append(clean_llm_reasoning(content_individual))
    usage_list.append(usage_individual)
    truncated_list.append(truncated_individual)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_individual))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 5: Parameter Type Alignment
    """
    print("=" * 64)
    step_name = "Parameter Type Alignment"
    print(COLOR_START+step_name+COLOR_END)
    aligned_info = s_pk_align_parameter(md_table_individual, llm, max_retries, initial_wait)
    if aligned_info is None:
        return None
    md_table_aligned, res_aligned, content_aligned, usage_aligned, truncated_aligned = aligned_info
    step_list.append(step_name)
    res_list.append(res_aligned)
    content_list.append(content_aligned)
    content_list_clean.append(clean_llm_reasoning(content_aligned))
    usage_list.append(usage_aligned)
    truncated_list.append(truncated_aligned)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_aligned))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 6: Column Header Categorization
    """
    print("=" * 64)
    step_name = "Column Header Categorization"
    print(COLOR_START+step_name+COLOR_END)
    mapping_info = s_pk_get_col_mapping(md_table_aligned, llm, max_retries, initial_wait)
    if mapping_info is None:
        return None
    col_mapping, res_mapping, content_mapping, usage_mapping, truncated_mapping = mapping_info
    step_list.append(step_name)
    res_list.append(res_mapping)
    content_list.append(content_mapping)
    content_list_clean.append(clean_llm_reasoning(content_mapping))
    usage_list.append(usage_mapping)
    truncated_list.append(truncated_mapping)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(col_mapping)
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 7: Rough Task Allocation (Preferably hidden from Users)
    """
    parameter_count = list(col_mapping.values()).count("Parameter")
    patient_id_count = list(col_mapping.values()).count("Patient ID")
    need_split_col = False
    need_match_drug = True
    need_match_patient = True
    if parameter_count == 0:
        return 0
    if patient_id_count == 0:
        return 0
    if patient_id_count == 1:
        need_split_col = False
    if patient_id_count > 1:
        need_split_col = True
    if markdown_to_dataframe(md_table_drug).shape[0] == 1:
        need_match_drug = False
    if markdown_to_dataframe(md_table_patient).shape[0] == 1:
        need_match_patient = False
    print("=" * 64)
    step_name = "Rough Task Allocation"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    tasks = ["Sub-table Creation", "Drug Matching", "Patient Matching"]
    statuses = [need_split_col, need_match_drug, need_match_patient]
    active_tasks = [task for task, status in zip(tasks, statuses) if status]
    canceled_tasks = [task for task, status in zip(tasks, statuses) if not status]
    print(f"LLM Execution: {', '.join(active_tasks) if active_tasks else 'None'}")
    print(f"Auto Execution: {', '.join(canceled_tasks) if canceled_tasks else 'None'}")
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 8: Sub-table Creation
    """
    print("=" * 64)
    step_name = "Sub-table Creation"
    print(COLOR_START + step_name + COLOR_END)
    if need_split_col:
        split_returns = s_pk_split_by_cols(md_table_aligned, col_mapping, llm, max_retries, initial_wait)
        if split_returns is None:
            return None
        md_table_list, res_split, content_split, usage_split, truncated_split = split_returns
        step_list.append(step_name)
        res_list.append(res_split)
        content_list.append(content_split)
        content_list_clean.append(clean_llm_reasoning(content_split))
        usage_list.append(usage_split)
        truncated_list.append(truncated_split)
    else:
        usage_split = 0
        content_split = "Automatic execution.\n"
        md_table_list = [md_table_aligned, ]

    _md_table_list = []
    for md in md_table_list:
        df = markdown_to_dataframe(md)
        cols_to_drop = [col for col in df.columns if col_mapping.get(col) == "Uncategorized"]
        df.drop(columns=cols_to_drop, inplace=True)
        _md_table_list.append(dataframe_to_markdown(df))
    # md_table_list = _md_table_list
    __md_table_list = []
    for md in _md_table_list:
        df = markdown_to_dataframe(md)
        cols_to_split = [col for col in df.columns if col_mapping.get(col) == "Parameter"]
        common_cols = [col for col in df.columns if col not in cols_to_split]
        for col in cols_to_split:
            if col in df.columns:
                selected_cols = [c for c in df.columns if c in common_cols or c == col]
                __md_table_list.append(dataframe_to_markdown(df[selected_cols].copy()))
    _md_table_list = __md_table_list
    __md_table_list = []
    for md in _md_table_list:
        df = markdown_to_dataframe(md)
        parameter_col = [col for col in df.columns if col_mapping.get(col) == "Parameter"][0]
        patient_col = [col for col in df.columns if col_mapping.get(col) == "Patient ID"][0]
        df_reshaped = df.rename(columns={patient_col: patient_col, parameter_col: "Parameter value"})
        df_reshaped["Parameter type (rough)"] = parameter_col
        df_reshaped = df_reshaped[[patient_col, "Parameter type (rough)", "Parameter value"]]
        __md_table_list.append(dataframe_to_markdown(df_reshaped))
    md_table_list = __md_table_list
    print(COLOR_START + "Usage:" + COLOR_END)
    print(usage_split)
    print(COLOR_START + "Result:" + COLOR_END)
    for i in range(len(md_table_list)):
        print(f"Index [{i}]:")
        print(display_md_table(md_table_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_split)
    """
    Step 9: Parameter Value Extraction
    """
    print("=" * 64)
    step_name = "Parameter Value Extraction"
    print(COLOR_START + step_name + COLOR_END)
    df = markdown_to_dataframe(md_table_aligned)
    col_name_of_parameter_type_list = [col for col in df.columns if col_mapping.get(col) == "Parameter"]
    # print(col_name_of_parameter_type_list)
    unit_info = s_pk_get_parameter_type_unit_value(md_table_aligned, col_name_of_parameter_type_list, description, llm, max_retries, initial_wait)
    if unit_info is None:
        return None
    tuple_type_unit, res_type_unit, content_type_unit, usage_type_unit, truncated_type_unit = unit_info
    type_unit_value_list = []
    for md_table in md_table_list:
        df = markdown_to_dataframe(md_table)
        df.rename(columns={'Parameter type (rough)': 'Parameter type'}, inplace=True)
        df['Parameter unit'] = ""
        for idx, value in df['Parameter type'].items():
            if value in col_name_of_parameter_type_list:
                position = col_name_of_parameter_type_list.index(value)
                df.at[idx, 'Parameter type'] = tuple_type_unit[0][position]
                df.at[idx, 'Parameter unit'] = tuple_type_unit[1][position]
        df_filtered = df[['Parameter type', 'Parameter value', 'Parameter unit']]
        type_unit_value_list.append(dataframe_to_markdown(df_filtered))
    step_list.append(step_name)
    res_list.append(res_type_unit)
    content_list.append(content_type_unit)
    content_list_clean.append(clean_llm_reasoning(content_type_unit))
    usage_list.append(usage_type_unit)
    truncated_list.append(truncated_type_unit)
    print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
    print(COLOR_START + "Result:" + COLOR_END)
    for i in range(len(type_unit_value_list)):
        print(f"Index [{i}]:")
        print(display_md_table(type_unit_value_list[i]))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 10: Drug Matching
    """
    drug_list = []
    round = 0
    if need_match_drug is False:
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            df_expanded = pd.concat([markdown_to_dataframe(md_table_drug)] * row_num, ignore_index=True)
            drug_list.append(dataframe_to_markdown(df_expanded))
    else:
        for md in md_table_list:
            print("=" * 64)
            step_name = "Drug Matching" + f" (Trial {str(round)})"
            round += 1
            print(COLOR_START + step_name + COLOR_END)
            drug_match_info = s_pk_match_drug_info(md_table_aligned, description, md, md_table_drug, llm, max_retries, initial_wait)
            if drug_match_info is None:
                return None
            drug_match_list, res_drug_match, content_drug_match, usage_drug_match, truncated_drug_match = drug_match_info
            df_table_drug = markdown_to_dataframe(md_table_drug)
            df_table_drug = pd.concat(
                [df_table_drug, pd.DataFrame([{'Drug name': 'ERROR', 'Analyte': 'ERROR', 'Specimen': 'ERROR'}])],
                ignore_index=True)
            df_table_drug_reordered = df_table_drug.iloc[drug_match_list].reset_index(drop=True)
            drug_list.append(dataframe_to_markdown(df_table_drug_reordered))
            # type_unit_list.append(md_type_unit)
            step_list.append(step_name)
            res_list.append(res_drug_match)
            content_list.append(content_drug_match)
            content_list_clean.append(clean_llm_reasoning(content_drug_match))
            usage_list.append(usage_drug_match)
            truncated_list.append(truncated_drug_match)
            print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
            print(COLOR_START + "Result:" + COLOR_END)
            print(display_md_table(drug_list[-1]))
            content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
            print(COLOR_START + "Reasoning:" + COLOR_END)
            print(content_to_print)

    print("=" * 64)
    step_name = "Drug Matching (Final)"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    for i in range(len(drug_list)):
        print(f"Index [{i}]:")
        print(display_md_table(drug_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 11: Population Matching
    """
    patient_list = []
    round = 0
    if need_match_patient is False:
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            # df_expanded = pd.concat([markdown_to_dataframe(md_table_patient)] * row_num, ignore_index=True)
            df_expanded = pd.concat([markdown_to_dataframe(md_table_patient_refined)] * row_num, ignore_index=True)
            patient_list.append(dataframe_to_markdown(df_expanded))
    else:
        for md in md_table_list:
            print("=" * 64)
            step_name = "Population Matching" + f" (Trial {str(round)})"
            round += 1
            print(COLOR_START + step_name + COLOR_END)
            patient_match_info = s_pk_match_patient_info(md_table_aligned, description, md, md_table_patient, llm, max_retries, initial_wait)
            if patient_match_info is None:
                return None
            patient_match_list, res_patient_match, content_patient_match, usage_patient_match, truncated_patient_match = patient_match_info
            df_table_patient = markdown_to_dataframe(md_table_patient_refined)
            df_table_patient = pd.concat(
                [df_table_patient,
                 pd.DataFrame([{'Patient ID': 'ERROR', 'Population': 'ERROR', 'Pregnancy stage': 'ERROR', 'Pediatric/Gestational age': 'ERROR'}])],
                ignore_index=True)
            df_table_patient_reordered = df_table_patient.iloc[patient_match_list].reset_index(drop=True)
            patient_list.append(dataframe_to_markdown(df_table_patient_reordered))
            step_list.append(step_name)
            res_list.append(res_patient_match)
            content_list.append(content_patient_match)
            content_list_clean.append(clean_llm_reasoning(content_patient_match))
            usage_list.append(usage_patient_match)
            truncated_list.append(truncated_patient_match)
            print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
            print(COLOR_START + "Result:" + COLOR_END)
            print(display_md_table(patient_list[-1]))
            content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
            print(COLOR_START + "Reasoning:" + COLOR_END)
            print(content_to_print)

    print("=" * 64)
    step_name = "Population Matching (Final)"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    for i in range(len(patient_list)):
        print(f"Index [{i}]:")
        print(display_md_table(patient_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    # """
    # Step 11: Population Matching (with cache)
    # """
    # patient_list = []
    # patient_cache = {}
    # round = 0
    # if need_match_patient is False:
    #     for md in md_table_list:
    #         df = markdown_to_dataframe(md)
    #         row_num = df.shape[0]
    #         df_expanded = pd.concat([markdown_to_dataframe(md_table_patient_refined)] * row_num, ignore_index=True)  # 这
    #         patient_list.append(dataframe_to_markdown(df_expanded))
    # else:
    #     for md in md_table_list:
    #         df = markdown_to_dataframe(md)
    #         col_name_patient_id = [col for col in df.columns if col_mapping.get(col) == "Patient ID"][0]
    #         if col_name_patient_id in patient_cache.keys():
    #             patient_list.append(patient_cache[col_name_patient_id])
    #         else:
    #             print("=" * 64)
    #             step_name = "Population Matching" + f" (Trial {str(round)})"
    #             round += 1
    #             print(COLOR_START + step_name + COLOR_END)
    #             patient_match_info = s_pk_match_patient_info(md_table_aligned, description, md, md_table_patient, llm, max_retries, initial_wait)
    #             if patient_match_info is None:
    #                 return None
    #             patient_match_list, res_patient_match, content_patient_match, usage_patient_match, truncated_patient_match = patient_match_info
    #             df_table_patient = markdown_to_dataframe(md_table_patient_refined)  # 这
    #             df_table_patient = pd.concat(
    #                 [df_table_patient,
    #                  pd.DataFrame([{'Population': 'ERROR', 'Pregnancy stage': 'ERROR', 'Pediatric/Gestational age': 'ERROR'}])],
    #                 ignore_index=True)
    #             df_table_patient_reordered = df_table_patient.iloc[patient_match_list].reset_index(drop=True)
    #             patient_list.append(dataframe_to_markdown(df_table_patient_reordered))
    #             # type_unit_list.append(md_type_unit)
    #             step_list.append(step_name)
    #             res_list.append(res_patient_match)
    #             content_list.append(content_patient_match)
    #             content_list_clean.append(clean_llm_reasoning(content_patient_match))
    #             usage_list.append(usage_patient_match)
    #             truncated_list.append(truncated_patient_match)
    #             print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
    #             print(COLOR_START + "Result:" + COLOR_END)
    #             print(display_md_table(patient_list[-1]))
    #             content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    #             print(COLOR_START + "Reasoning:" + COLOR_END)
    #             print(content_to_print)
    #             patient_cache[col_name_patient_id] = dataframe_to_markdown(df_table_patient_reordered)
    #
    # print("=" * 64)
    # step_name = "Population Matching (Final)"
    # print(COLOR_START+step_name+COLOR_END)
    # print(COLOR_START+"Usage:"+COLOR_END, 0)
    # print(COLOR_START+"Result:"+COLOR_END)
    # for i in range(len(patient_list)):
    #     print(f"Index [{i}]:")
    #     print(display_md_table(patient_list[i]))
    # print(COLOR_START + "Reasoning:" + COLOR_END)
    # print("Automatic execution.\n")
    # step_list.append(step_name)
    # res_list.append(True)
    # content_list.append("Automatic execution.\n")
    # content_list_clean.append("Automatic execution.\n")
    # usage_list.append(0)
    # truncated_list.append(False)
    """
    Step 12: Time Extraction
    """
    time_list = []
    round = 0
    for md in md_table_list:
        print("=" * 64)
        step_name = "Time Extraction" + f" (Trial {str(round)})"
        round += 1
        print(COLOR_START + step_name + COLOR_END)
        time_info = s_pk_extract_time_and_unit(md_table_aligned, description, md, llm, max_retries, initial_wait)
        if time_info is None:
            return None
        md_time, res_time, content_time, usage_time, truncated_time = time_info
        time_list.append(md_time)
        step_list.append(step_name)
        res_list.append(res_time)
        content_list.append(content_time)
        content_list_clean.append(clean_llm_reasoning(content_time))
        usage_list.append(usage_time)
        truncated_list.append(truncated_time)
        print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
        print(COLOR_START + "Result:" + COLOR_END)
        print(display_md_table(md_time))
        content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
        print(COLOR_START + "Reasoning:" + COLOR_END)
        print(content_to_print)
    """
    Step 13: Time Extraction (Final)
    """
    print("=" * 64)
    step_name = "Time Extraction (Final)"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    for i in range(len(time_list)):
        print(f"Index [{i}]:")
        print(display_md_table(time_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 14: Assembly
    """
    df_list = []
    assert len(drug_list) == len(patient_list) == len(type_unit_value_list) == len(time_list)
    for i in range(len(drug_list)):
        df_drug = markdown_to_dataframe(drug_list[i])
        df_table_patient = markdown_to_dataframe(patient_list[i])
        df_type_unit_value = markdown_to_dataframe(type_unit_value_list[i])
        df_time = markdown_to_dataframe(time_list[i])
        df_combined = pd.concat([df_table_patient, df_drug, df_type_unit_value, df_time], axis=1)
        df_list.append(df_combined)
    df_combined = pd.concat(df_list, ignore_index=True)
    print("=" * 64)
    step_name = "Assembly"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(dataframe_to_markdown(df_combined)))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 15: Row cleanup
    """
    df_combined["original_index"] = df_combined.index

    """fix col name"""
    expected_columns = ["Patient ID", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Drug name", "Analyte", "Specimen", "Parameter type", "Parameter unit", "Parameter value", "Time value", "Time unit"]

    def rename_columns(df, expected_columns):
        renamed_columns = {}
        for col in df.columns:
            matches = get_close_matches(col, expected_columns, n=1, cutoff=0.8)
            if matches:
                renamed_columns[col] = matches[0]
            else:
                renamed_columns[col] = col

        df.rename(columns=renamed_columns, inplace=True)
        return df

    df_combined = rename_columns(df_combined, expected_columns)

    """Delete ERROR rows"""
    df_combined = df_combined[df_combined.ne("ERROR").all(axis=1)]
    """if Time unit == "weeks", Time value must be "N/A" ...... so it can be removed"""
    df_combined.loc[
        (df_combined["Time unit"] == "Weeks"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "weeks"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Week"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "week"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Wks"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "wks"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Wk"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "wk"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "W"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "w"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Months"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "months"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Month"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "month"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Years"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "years"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "Year"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "year"), "Time value"] = "N/A"

    """if Time == "N/A", Time unit must be "N/A" ......"""
    df_combined.loc[
        (df_combined["Time value"] == "N/A"), "Time unit"] = "N/A"
    df_combined.loc[
        (df_combined["Time unit"] == "N/A"), "Time value"] = "N/A"
    """if Value == "N/A", type and unit must be "N/A"。"""
    df_combined.loc[
        (df_combined["Parameter value"] == "N/A"), "Parameter type"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter value"] == "N/A"), "Parameter unit"] = "N/A"
    """if Cmax, Cavg, Tmax, time value and unit must be "N/A"。"""
    df_combined.loc[
        (df_combined["Parameter type"] == "Cmax"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter value"] == "Cmax"), "Time unit"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter type"] == "Tmax"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter value"] == "Tmax"), "Time unit"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter type"] == "Cavg"), "Time value"] = "N/A"
    df_combined.loc[
        (df_combined["Parameter value"] == "Cavg"), "Time unit"] = "N/A"
    """replace empty by N/A"""
    df_combined.replace(r'^\s*$', 'N/A', regex=True, inplace=True)
    """replace n/a by N/A"""
    df_combined.replace("n/a", "N/A", inplace=True)
    """replace unknown by N/A"""
    df_combined.replace("unknown", "N/A", inplace=True)
    df_combined.replace("Unknown", "N/A", inplace=True)
    """replace nan by N/A"""
    df_combined.replace("nan", "N/A", inplace=True)
    """replace , by empty"""
    df_combined.replace(",", " ", inplace=True)

    """Remove non-digit rows"""
    # columns_to_check = ["Parameter value",]
    #
    # def contains_number(s):
    #     return any(char.isdigit() for char in s)
    #
    # df_combined = df_combined[df_combined[columns_to_check].apply(lambda row: any(contains_number(str(cell)) for cell in row), axis=1)]
    # df_combined = df_combined.reset_index(drop=True)
    columns_to_check = ["Parameter value", ]

    df_combined = df_combined[
        ~df_combined[columns_to_check].apply(lambda row: any(str(cell) == 'N/A' for cell in row), axis=1)]
    df_combined = df_combined.reset_index(drop=True)

    """Remove duplicate"""
    df_combined = df_combined.drop_duplicates()
    df_combined = df_combined.reset_index(drop=True)

    """Final"""
    df_combined.sort_values(by="original_index", inplace=True)
    df_combined.drop(columns=["original_index"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    df_combined = df_combined.sort_values(
        by="Patient ID",
        key=lambda x: pd.Categorical(x, categories=df_combined["Patient ID"].unique(), ordered=True)
    ).reset_index(drop=True)

    print("=" * 64)
    step_name = "Row cleanup"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(dataframe_to_markdown(df_combined)))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)

    """
    Step 16: Post-operation Inspection
    """

    """Rename col names"""
    # column_mapping = {
    #     # "Parameter unit": "Unit",
    #     # "Main value": "Value",
    #     # "Statistics type": "Summary Statistics",
    #     # "Lower bound": "Lower limit",
    #     # "Upper bound": "High limit",
    #     "Main value": "Parameter value",
    #     "Statistics type": "Parameter statistic",
    #     "Lower bound": "Lower limit",
    #     "Upper bound": "High limit",
    # }
    # df_combined = df_combined.rename(columns=column_mapping)

    print("=" * 64)
    step_name = "Column Renaming & Final Inspection"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(dataframe_to_markdown(df_combined)))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)

    return df_combined, step_list, res_list, content_list, content_list_clean, usage_list, truncated_list


