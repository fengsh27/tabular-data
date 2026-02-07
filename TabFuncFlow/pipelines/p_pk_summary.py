from TabFuncFlow.steps_pk_summary.s_pk_delete_individual import *
from TabFuncFlow.steps_pk_summary.s_pk_align_parameter import *
from TabFuncFlow.steps_pk_summary.s_pk_extract_drug_info import *
from TabFuncFlow.steps_pk_summary.s_pk_extract_patient_info import *
from TabFuncFlow.steps_pk_summary.s_pk_extract_time_and_unit import *
from TabFuncFlow.steps_pk_summary.s_pk_get_col_mapping import *
from TabFuncFlow.steps_pk_summary.s_pk_get_parameter_type_and_unit import *
from TabFuncFlow.steps_pk_summary.s_pk_match_drug_info import *
from TabFuncFlow.steps_pk_summary.s_pk_match_patient_info import *
from TabFuncFlow.steps_pk_summary.s_pk_split_by_cols import *
from TabFuncFlow.steps_pk_summary.s_pk_get_parameter_value import *
from TabFuncFlow.steps_pk_summary.s_pk_refine_patient_info import *
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


def p_pk_summary(md_table, description, llm="gemini_15_pro", max_retries=5, initial_wait=2, use_color=True, clean_reasoning=False):
    """
    PK Summary Pipeline 250227
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
    # """
    # Step 2-1: Population Information Extraction (Trial 1)
    # """
    # print("=" * 64)
    # step_name = "Population Information Extraction (Trial 1)"
    # print(COLOR_START+step_name+COLOR_END)
    # patient_info_1 = s_pk_extract_patient_info(md_table, description, llm, max_retries, initial_wait)
    # if patient_info_1 is None:
    #     return None
    # md_table_patient_1, res_patient_1, content_patient_1, usage_patient_1, truncated_patient_1 = patient_info_1
    # step_list.append(step_name)
    # res_list.append(res_patient_1)
    # content_list.append(content_patient_1)
    # content_list_clean.append(clean_llm_reasoning(content_patient_1))
    # usage_list.append(usage_patient_1)
    # truncated_list.append(truncated_patient_1)
    # print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    # print(COLOR_START+"Result:"+COLOR_END)
    # print(display_md_table(md_table_patient_1))
    # content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    # print(COLOR_START + "Reasoning:" + COLOR_END)
    # print(content_to_print)
    # """
    # Step 2-2: Population Information Extraction (Trial 2)
    # """
    # print("=" * 64)
    # step_name = "Population Information Extraction (Trial 2)"
    # print(COLOR_START+step_name+COLOR_END)
    # patient_info_2 = s_pk_extract_patient_info(md_table, description, llm, max_retries, initial_wait)
    # if patient_info_2 is None:
    #     return None
    # md_table_patient_2, res_patient_2, content_patient_2, usage_patient_2, truncated_patient_2 = patient_info_2
    # step_list.append(step_name)
    # res_list.append(res_patient_2)
    # content_list.append(content_patient_2)
    # content_list_clean.append(clean_llm_reasoning(content_patient_2))
    # usage_list.append(usage_patient_2)
    # truncated_list.append(truncated_patient_2)
    # print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    # print(COLOR_START+"Result:"+COLOR_END)
    # print(display_md_table(md_table_patient_2))
    # content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    # print(COLOR_START + "Reasoning:\n" + COLOR_END)
    # print(content_to_print)
    # """
    # Step 2-3: Population Information Extraction (Keep The Longest Table)
    # """
    # if len(md_table_patient_1) >= len(md_table_patient_2):
    #     patient_info = patient_info_1
    # else:
    #     patient_info = patient_info_2
    # print("=" * 64)
    # step_name = "Population Information Extraction (Final)"
    # print(COLOR_START+step_name+COLOR_END)
    # print(COLOR_START+"Usage:"+COLOR_END, 0)
    # print(COLOR_START+"Result:"+COLOR_END)
    # print(display_md_table(patient_info[0]))
    # print(COLOR_START + "Reasoning:" + COLOR_END)
    # print("Automatic execution.\n")
    # md_table_patient, res_patient, content_patient, usage_patient, truncated_patient = patient_info
    # step_list.append(step_name)
    # res_list.append(True)
    # content_list.append("Automatic execution.\n")
    # content_list_clean.append("Automatic execution.\n")
    # usage_list.append(0)
    # truncated_list.append(False)
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
    Step 4: Individual Data Deletion
    """
    print("=" * 64)
    step_name = "Individual Data Deletion"
    print(COLOR_START+step_name+COLOR_END)
    summary_only_info = s_pk_delete_individual(md_table, llm, max_retries, initial_wait)
    if summary_only_info is None:
        return None
    md_table_summary, res_summary, content_summary, usage_summary, truncated_summary = summary_only_info
    step_list.append(step_name)
    res_list.append(res_summary)
    content_list.append(content_summary)
    content_list_clean.append(clean_llm_reasoning(content_summary))
    usage_list.append(usage_summary)
    truncated_list.append(truncated_summary)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_summary))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)
    """
    Step 5: Parameter Type Alignment
    """
    print("=" * 64)
    step_name = "Parameter Type Alignment"
    print(COLOR_START+step_name+COLOR_END)
    aligned_info = s_pk_align_parameter(md_table_summary, llm, max_retries, initial_wait)
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
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    parameter_unit_count = list(col_mapping.values()).count("Parameter unit")
    parameter_value_count = list(col_mapping.values()).count("Parameter value")
    parameter_pvalue_count = list(col_mapping.values()).count("P value")
    need_get_unit = True
    need_split_col = False
    need_match_drug = True
    need_match_patient = True
    if parameter_value_count == 0:
        return
    if parameter_type_count == 0:
        return
    if parameter_type_count > 1 or parameter_pvalue_count > 1:
        need_split_col = True
    if parameter_unit_count == 1 and parameter_type_count == 1:
        need_get_unit = False
    if markdown_to_dataframe(md_table_drug).shape[0] == 1:
        need_match_drug = False
    if markdown_to_dataframe(md_table_patient).shape[0] == 1:
        need_match_patient = False
    print("=" * 64)
    step_name = "Rough Task Allocation"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    tasks = ["Unit Extraction", "Sub-table Creation", "Drug Matching", "Population Matching"]
    statuses = [need_get_unit, need_split_col, need_match_drug, need_match_patient]
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
        cols_to_split = [col for col in df.columns if col_mapping.get(col) == "Parameter value"]
        common_cols = [col for col in df.columns if col not in cols_to_split]
        for col in cols_to_split:
            if col in df.columns:
                selected_cols = [c for c in df.columns if c in common_cols or c == col]
                __md_table_list.append(dataframe_to_markdown(df[selected_cols].copy()))
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
    Step 9: Unit Extraction
    """
    type_unit_list = []
    type_unit_cache = {}
    round = 0
    for md in md_table_list:
        df = markdown_to_dataframe(md)
        col_name_of_parameter_type = [col for col in df.columns if col_mapping.get(col) == "Parameter type"][0]
        col_name_of_parameter_unit_list = [col for col in df.columns if col_mapping.get(col) == "Parameter unit"]
        if col_name_of_parameter_type in type_unit_cache.keys():
            type_unit_list.append(type_unit_cache[col_name_of_parameter_type])
        else:
            if len(col_name_of_parameter_unit_list) == 1:
                # print(col_name_of_parameter_unit_list)
                selected_cols = [col_name_of_parameter_type, col_name_of_parameter_unit_list[0]]
                df_selected = df[selected_cols].copy()
                if df_selected is None or df_selected.empty:
                    raise ValueError(
                        "df_selected is None or empty. Please check the input DataFrame and selected columns.")
                df_selected = df_selected.rename(
                    columns={col_name_of_parameter_type: "Parameter type",
                             col_name_of_parameter_unit_list[0]: "Parameter unit"}
                )

                type_unit_list.append(dataframe_to_markdown(df_selected))
            else:
                print("=" * 64)
                step_name = "Unit Extraction" + f" (Trial {str(round)})"
                round += 1
                print(COLOR_START + step_name + COLOR_END)
                unit_info = s_pk_get_parameter_type_and_unit(md_table_aligned, col_mapping, md, description, llm, max_retries, initial_wait)
                if unit_info is None:
                    return None
                tuple_type_unit, res_type_unit, content_type_unit, usage_type_unit, truncated_type_unit = unit_info
                md_type_unit = dataframe_to_markdown(pd.DataFrame([tuple_type_unit[0], tuple_type_unit[1]], index=["Parameter type", "Parameter unit"]).T)
                type_unit_list.append(md_type_unit)
                step_list.append(step_name)
                res_list.append(res_type_unit)
                content_list.append(content_type_unit)
                content_list_clean.append(clean_llm_reasoning(content_type_unit))
                usage_list.append(usage_type_unit)
                truncated_list.append(truncated_type_unit)
                print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
                print(COLOR_START + "Result:" + COLOR_END)
                print(display_md_table(md_type_unit))
                content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
                print(COLOR_START + "Reasoning:" + COLOR_END)
                print(content_to_print)
            type_unit_cache[col_name_of_parameter_type] = type_unit_list[-1]
    """
    Step 10: Unit Extraction (Final)
    """
    print("=" * 64)
    step_name = "Unit Extraction (Final)"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    for i in range(len(type_unit_list)):
        print(f"Index [{i}]:")
        print(display_md_table(type_unit_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 11: Drug Matching
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
    Step 12: Population Matching
    """
    patient_list = []
    round = 0
    if need_match_patient is False:
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            # df_expanded = pd.concat([markdown_to_dataframe(md_table_patient)] * row_num, ignore_index=True)
            df_expanded = pd.concat([markdown_to_dataframe(md_table_patient_refined)] * row_num, ignore_index=True)  # 这
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
            df_table_patient = markdown_to_dataframe(md_table_patient_refined)  # 这
            # df_table_patient = pd.concat(
            #     [df_table_patient,
            #      pd.DataFrame([{'Population': 'ERROR', 'Pregnancy stage': 'ERROR', 'Subject N': 'ERROR'}])],
            #     ignore_index=True)
            df_table_patient = pd.concat(
                [df_table_patient,
                 pd.DataFrame([{'Population': 'ERROR', 'Pregnancy stage': 'ERROR', 'Pediatric/Gestational age': 'ERROR', 'Subject N': 'ERROR'}])],
                ignore_index=True)
            df_table_patient_reordered = df_table_patient.iloc[patient_match_list].reset_index(drop=True)
            patient_list.append(dataframe_to_markdown(df_table_patient_reordered))
            # type_unit_list.append(md_type_unit)
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
    """
    Step 13: Parameter Value Extraction
    """
    value_list = []
    round = 0
    for md in md_table_list:
        print("=" * 64)
        step_name = "Parameter Value Extraction" + f" (Trial {str(round)})"
        round += 1
        print(COLOR_START + step_name + COLOR_END)
        value_info = s_pk_get_parameter_value(md_table_aligned, description, md, llm, max_retries, initial_wait)
        if value_info is None:
            return None
        md_value, res_value, content_value, usage_value, truncated_value = value_info
        value_list.append(md_value)
        step_list.append(step_name)
        res_list.append(res_value)
        content_list.append(content_value)
        content_list_clean.append(clean_llm_reasoning(content_value))
        usage_list.append(usage_value)
        truncated_list.append(truncated_value)
        print(COLOR_START + "Usage:" + COLOR_END, usage_list[-1])
        print(COLOR_START + "Result:" + COLOR_END)
        print(display_md_table(md_value))
        content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
        print(COLOR_START + "Reasoning:" + COLOR_END)
        print(content_to_print)
    """
    Step 14: Parameter Value Extraction (Final)
    """
    print("=" * 64)
    step_name = "Parameter Value Extraction (Final)"
    print(COLOR_START+step_name+COLOR_END)
    print(COLOR_START+"Usage:"+COLOR_END, 0)
    print(COLOR_START+"Result:"+COLOR_END)
    for i in range(len(value_list)):
        print(f"Index [{i}]:")
        print(display_md_table(value_list[i]))
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print("Automatic execution.\n")
    step_list.append(step_name)
    res_list.append(True)
    content_list.append("Automatic execution.\n")
    content_list_clean.append("Automatic execution.\n")
    usage_list.append(0)
    truncated_list.append(False)
    """
    Step 15: Assembly
    """
    df_list = []
    assert len(drug_list) == len(patient_list) == len(type_unit_list) == len(value_list)# == len(time_list)
    for i in range(len(drug_list)):
        df_drug = markdown_to_dataframe(drug_list[i])
        df_table_patient = markdown_to_dataframe(patient_list[i])
        df_type_unit = markdown_to_dataframe(type_unit_list[i])
        df_value = markdown_to_dataframe(value_list[i])
        # df_time = markdown_to_dataframe(time_list[i])
        # df_combined = pd.concat([df_drug, df_table_patient, df_time, df_type_unit, df_value], axis=1)
        df_combined = pd.concat([df_drug, df_table_patient, df_type_unit, df_value], axis=1)
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
    Step 16: Row cleanup
    """
    # df_combined["original_index"] = df_combined.index

    """fix col name"""
    # expected_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type", "Parameter unit", "Main value", "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound", "Upper bound", "P value"]
    expected_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N", "Parameter type", "Parameter unit", "Main value", "Statistics type", "Variation type", "Variation value", "Interval type", "Lower bound", "Upper bound", "P value"]

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
    # """if Time == "N/A", Time unit must be "N/A"。"""
    # df_combined.loc[
    #     (df_combined["Time value"] == "N/A"), "Time unit"] = "N/A"
    """if Statistics type == Interval type or N/A, and (Main value == Lower bound or Main value == Upper bound), set Main value and Statistics type = N/A"""
    condition = (
            (
                    (df_combined["Statistics type"] == df_combined["Interval type"]) |
                    (df_combined["Statistics type"] == 'N/A')
            ) &
            (
                    (df_combined["Main value"] == df_combined["Lower bound"]) |
                    (df_combined["Main value"] == df_combined["Upper bound"])
            )
    )
    df_combined.loc[condition, ["Main value", "Statistics type"]] = "N/A"
    """if Lower bound and Upper bound are both in Main value (string), Main value = N/A"""
    def contains_bounds(row):
        main_value = str(row["Main value"])
        lower = str(row["Lower bound"])
        upper = str(row["Upper bound"])
        if lower.strip() != "N/A" and upper.strip() != "N/A":
            return lower in main_value and upper in main_value
        return False
    mask = df_combined.apply(contains_bounds, axis=1)
    df_combined.loc[mask, "Main value"] = "N/A"
    """if Value == "N/A", Summary Statistics must be "N/A"。"""
    df_combined.loc[
        (df_combined["Main value"] == "N/A"), "Statistics type"] = "N/A"
    """if Lower limit & High limit == "N/A", Interval type must be "N/A"。"""
    df_combined.loc[
        (df_combined["Lower bound"] == "N/A") & (df_combined["Upper bound"] == "N/A"), "Interval type"] = "N/A"
    """if Lower limit & High limit != "N/A", Interval type set as default "Range" """
    df_combined.loc[
        (df_combined["Lower bound"] != "N/A") & (df_combined["Upper bound"] != "N/A"), "Interval type"] = "Range"
    """if Variation value == "N/A", Variation type must be "N/A"。"""
    df_combined.loc[
        (df_combined["Variation value"] == "N/A"), "Variation type"] = "N/A"
    # df_combined = df_combined.reset_index(drop=True)
    """replace empty by N/A"""
    df_combined.replace(r'^\s*$', 'N/A', regex=True, inplace=True)
    """replace n/a by N/A"""
    df_combined.replace("n/a", "N/A", inplace=True)
    """replace unknown by N/A"""
    df_combined.replace("unknown", "N/A", inplace=True)
    df_combined.replace("Unknown", "N/A", inplace=True)
    """replace nan by N/A"""
    df_combined.replace("nan", "N/A", inplace=True)
    """replace Standard Deviation (SD) by SD"""
    df_combined.replace("Standard Deviation (SD)", "SD", inplace=True)
    df_combined.replace("s.d.", "SD", inplace=True)
    df_combined.replace("S.D.", "SD", inplace=True)

    """replace , by empty"""
    df_combined.replace(",", " ", inplace=True)

    """Remove non-digit rows"""
    columns_to_check = ["Main value", "Variation value", "Lower bound", "Upper bound"]

    def contains_number(s):
        return any(char.isdigit() for char in s)

    df_combined = df_combined[df_combined[columns_to_check].apply(lambda row: any(contains_number(str(cell)) for cell in row), axis=1)]
    # df_combined = df_combined.reset_index(drop=True)

    """ Merge """

    df = df_combined.copy()

    df.replace("N/A", pd.NA, inplace=True)

    # group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type",
    #                  "Parameter unit"]
    group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N", "Parameter type",
                     "Parameter unit"]
    grouped = df.groupby(group_columns, dropna=False)

    merged_rows = []
    for _, group in grouped:
        # group = group.reset_index(drop=True)
        used_indices = set()

        for i, j in itertools.combinations(range(len(group)), 2):
            if i in used_indices or j in used_indices:
                continue

            row1, row2 = group.iloc[i].copy(), group.iloc[j].copy()
            can_merge = True

            for col in df.columns:
                val1, val2 = row1[col], row2[col]
                if pd.notna(val1) and pd.notna(val2) and val1 != val2:
                    can_merge = False
                    break
                elif pd.isna(val1) and pd.notna(val2):
                    row1[col] = val2
                elif pd.notna(val1) and pd.isna(val2):
                    row2[col] = val1

            if can_merge:
                used_indices.add(i)
                used_indices.add(j)
                merged_rows.append(row1)
            # else:
            #     merged_rows.append(row1)
            #     merged_rows.append(row2)

        for i in range(len(group)):
            if i not in used_indices:
                merged_rows.append(group.iloc[i])

    df_merged = pd.DataFrame(merged_rows, columns=df.columns)
    df_merged.fillna("N/A", inplace=True)

    df_combined = df_merged
    # df_combined = df_combined.reset_index(drop=True)

    """Remove duplicate"""
    df_combined = df_combined.drop_duplicates()
    # df_combined = df_combined.reset_index(drop=True)

    """delete 'fill in subject N as value error', this implementation is bad, still looking for better solutions"""
    df_combined = df_combined[df_combined["Subject N"] != df_combined["Main value"]]
    # df_combined = df_combined[~df_combined["Value"].isin(markdown_to_dataframe(md_table_patient)["Subject N"].to_list())]
    # df_combined = df_combined.reset_index(drop=True)

    """fix put range only in lower limit/high limit"""
    float_pattern = re.compile(r"-?\d+\.\d+")

    def extract_limits(row):
        if row["Upper bound"] == "N/A":
            numbers = float_pattern.findall(str(row["Lower bound"]))
            if len(numbers) == 2:
                return pd.Series([str(numbers[0]), str(numbers[1])])
        if row["Lower bound"] == "N/A":
            numbers = float_pattern.findall(str(row["Upper bound"]))
            if len(numbers) == 2:
                return pd.Series([str(numbers[0]), str(numbers[1])])
        if row["Upper bound"] == row["Lower bound"]:
            numbers = float_pattern.findall(str(row["Upper bound"]))
            if len(numbers) == 2:
                return pd.Series([str(numbers[0]), str(numbers[1])])
        return pd.Series([row["Lower bound"], row["Upper bound"]])

    df_combined[["Lower bound", "Upper bound"]] = df_combined.apply(extract_limits, axis=1)
    # df_combined = df_combined.reset_index(drop=True)

    """remove inclusive rows"""
    def remove_contained_rows(df):
        df_cleaned = df

        rows_to_drop = set()
        for i in range(len(df_cleaned)):
            for j in range(i + 1, len(df_cleaned)):
                row1 = df_cleaned.iloc[i]
                row2 = df_cleaned.iloc[j]

                if all((r1 == r2) or (r1 == "N/A") for r1, r2 in zip(row1, row2)):
                    rows_to_drop.add(i)  # row1 included by row2
                elif all((r2 == r1) or (r2 == "N/A") for r1, r2 in zip(row1, row2)):
                    rows_to_drop.add(j)

        df_cleaned = df_cleaned.drop(index=rows_to_drop)  # .reset_index(drop=True)
        return df_cleaned

    df_combined = remove_contained_rows(df_combined)
    df_combined = remove_contained_rows(df_combined)
    df_combined = remove_contained_rows(df_combined)
    df_combined = remove_contained_rows(df_combined)
    df_combined = remove_contained_rows(df_combined)
    # df_combined = df_combined.reset_index(drop=True)

    """col exchange"""
    cols = list(df_combined.columns)
    i, j = cols.index('Main value'), cols.index('Statistics type')
    cols[i], cols[j] = cols[j], cols[i]
    df_combined = df_combined[cols]
    # df_combined = df_combined.reset_index(drop=True)

    """give range to median"""
    # group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type",
    #                  "Parameter unit"]
    group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Pediatric/Gestational age", "Subject N", "Parameter type",
                     "Parameter unit"]

    # Finding pairs of rows that match on group_columns
    grouped = df_combined.groupby(group_columns)

    # Processing each group
    for _, group in grouped:
        if len(group) == 2:  # Only process if there are exactly two rows in the group
            median_row = group[group["Statistics type"] == "Median"]
            non_median_row = group[group["Statistics type"] != "Median"]

            if not median_row.empty and not non_median_row.empty:
                # Check if non-median row has "Range"
                if "Range" in non_median_row["Interval type"].values:
                    # Assign range values to the median row
                    df_combined.loc[median_row.index, ["Interval type", "Lower bound", "Upper bound"]] = \
                        non_median_row[["Interval type", "Lower bound", "Upper bound"]].values

                    # Remove range information from the non-median row
                    df_combined.loc[non_median_row.index, ["Interval type", "Lower bound", "Upper bound"]] = ["N/A", "N/A", "N/A"]

    # df_combined = df_combined.reset_index(drop=True)

    # df_combined.sort_values(by="original_index", inplace=True)
    # df_combined.drop(columns=["original_index"], inplace=True)
    # df_combined.reset_index(drop=True, inplace=True)
    df_combined["original_order"] = df_combined.index
    df_combined = df_combined.sort_values(by="original_order").drop(columns=["original_order"]).reset_index(drop=True)

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

    # """
    # Step 17: Time Extraction
    # """
    print("=" * 64)
    step_name = "Time Appending"
    print(COLOR_START+step_name+COLOR_END)
    md_data_lines_after_post_process = dataframe_to_markdown(df_combined[["Main value", "Statistics type", "Variation type", "Variation value",
                        "Interval type", "Lower bound", "Upper bound", "P value"]])
    time_info = s_pk_extract_time_and_unit(md_table_aligned, description, md_data_lines_after_post_process, llm, max_retries, initial_wait)
    if time_info is None:
        return None
    md_table_time, res_time, content_time, usage_time, truncated_time = time_info
    step_list.append(step_name)
    res_list.append(res_time)
    content_list.append(content_time)
    content_list_clean.append(clean_llm_reasoning(content_time))
    usage_list.append(usage_time)
    truncated_list.append(truncated_time)
    print(COLOR_START+"Usage:"+COLOR_END, usage_list[-1])
    print(COLOR_START+"Result:"+COLOR_END)
    print(display_md_table(md_table_time))
    content_to_print = content_list_clean[-1] if clean_reasoning else content_list[-1]
    print(COLOR_START + "Reasoning:" + COLOR_END)
    print(content_to_print)

    # insert_pos = df_combined.columns.get_loc("Parameter type")  # before Parameter type
    # # split df_combined and insert df of time
    # df_combined = pd.concat([df_combined.iloc[:, :insert_pos], markdown_to_dataframe(md_table_time), df_combined.iloc[:, insert_pos:]], axis=1)
    df_combined = pd.concat([df_combined, markdown_to_dataframe(md_table_time)], axis=1)
    df_combined = df_combined.reset_index(drop=True)
    """
    Step 17: Post-operation Inspection
    """

    """Rename col names"""
    column_mapping = {
        # "Parameter unit": "Unit",
        # "Main value": "Value",
        # "Statistics type": "Summary Statistics",
        # "Lower bound": "Lower limit",
        # "Upper bound": "High limit",
        "Main value": "Parameter value",
        "Statistics type": "Parameter statistic",
        # "Lower bound": "Lower limit",
        # "Upper bound": "High limit",
    }
    df_combined = df_combined.rename(columns=column_mapping)

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


