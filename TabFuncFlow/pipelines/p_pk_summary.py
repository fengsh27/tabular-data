import time
from TabFuncFlow.steps_pk_summary.s_pk_delete_individual import *
from TabFuncFlow.steps_pk_summary.s_pk_align_parameter import *
from TabFuncFlow.steps_pk_summary.s_pk_extract_drug_info import *
from TabFuncFlow.steps_pk_summary.s_pk_extract_patient_info import *
from TabFuncFlow.steps_pk_summary.s_pk_get_col_mapping import *
from TabFuncFlow.steps_pk_summary.s_pk_get_parameter_type_and_unit import *
from TabFuncFlow.steps_pk_summary.s_pk_match_drug_info import *
from TabFuncFlow.steps_pk_summary.s_pk_match_patient_info import *
from TabFuncFlow.steps_pk_summary.s_pk_split_by_cols import *
from TabFuncFlow.steps_pk_summary.s_pk_get_parameter_value import *
import re
import itertools


def clean_llm_reasoning(text: str) -> str:
    """
    Cleans the LLM inference string by removing content after the last occurrence
    of '<<' or '[[END]]', keeping only the portion up to the last preceding period.
    If the last period is too close to the previous newline, truncate at the newline instead.
    Ensures the output ends with a newline.

    :param text: The input inference string.
    :return: The cleaned inference string, ensuring it ends with a newline.
    """
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


# def run_with_retry(func, *args, max_retries=5, base_delay=10, **kwargs):
#     delay = base_delay
#     for attempt in range(max_retries):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#                 delay *= 2
#             else:
#                 print("Max retries reached. Returning None.")
#                 return None
#
#     # it will not get here
#     return None


def run_with_retry(func, *args, max_retries=5, base_delay=5, **kwargs):
    delay = base_delay
    last_matched_number = 0

    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                modified_result = list(result)
                modified_result[-2] += last_matched_number
                return tuple(modified_result) if isinstance(result, tuple) else modified_result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            matches = re.findall(r'<<(\d+)>>', str(e))
            if matches:
                last_matched_number += int(matches[-1])

            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print("Max retries reached. Returning None.")
                return None
    return None


def p_pk_summary(md_table, description, llm="gemini_15_pro", max_retries=5, base_delay=1, use_color=True, clean_reasoning=False):
    """
    PK Summary Pipeline 250227
    Summarizes pharmacokinetic (PK) data from a given markdown table.

    :param md_table: The markdown representation of a SINGLE HTML table containing PK data.
    :param description: Additional contextual information, including captions, footnotes, etc.
    :param llm: The language model to use for processing ("gemini_15_pro" or "chatgpt_4o").
    :param max_retries: Maximum number of retries before failing.
    :param base_delay: Initial delay before retrying (in seconds), which doubles on each failure.
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
    drug_info = run_with_retry(
        s_pk_extract_drug_info,
        md_table,
        description,
        llm,
        max_retries=max_retries,
        base_delay=base_delay,
    )
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
    # print("\n"*1)
    """
    Step 2: Population Information Extraction
    """
    print("=" * 64)
    step_name = "Population Information Extraction"
    print(COLOR_START+step_name+COLOR_END)
    patient_info = run_with_retry(
        s_pk_extract_patient_info,
        md_table,
        description,
        llm,
        max_retries=max_retries,
        base_delay=base_delay,
    )
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
    Step 2-1: Population Information Extraction (Trial 1)
    """
    # print("=" * 64)
    # step_name = "Population Information Extraction (Trial 1)"
    # print(COLOR_START+step_name+COLOR_END)
    # patient_info_1 = run_with_retry(
    #     s_pk_extract_patient_info,
    #     md_table,
    #     description,
    #     llm,
    #     max_retries=max_retries,
    #     base_delay=base_delay,
    # )
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
    """
    Step 2-2: Population Information Extraction (Trial 2)
    """
    # print("=" * 64)
    # step_name = "Population Information Extraction (Trial 2)"
    # print(COLOR_START+step_name+COLOR_END)
    # patient_info_2 = run_with_retry(
    #     s_pk_extract_patient_info,
    #     md_table,
    #     description,
    #     llm,
    #     max_retries=max_retries,
    #     base_delay=base_delay,
    # )
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
    """
    Step 2-3: Population Information Extraction (Keep The Longest Table)
    """
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
    Step 3: Individual Data Deletion
    """
    print("=" * 64)
    step_name = "Individual Data Deletion"
    print(COLOR_START+step_name+COLOR_END)
    summary_only_info = run_with_retry(
        s_pk_delete_individual,
        md_table,
        llm,
        max_retries=max_retries,
        base_delay=base_delay,
    )
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
    # print("\n"*1)
    """
    Step 4: Parameter Type Alignment
    """
    print("=" * 64)
    step_name = "Parameter Type Alignment"
    print(COLOR_START+step_name+COLOR_END)
    aligned_info = run_with_retry(
        s_pk_align_parameter,
        md_table_summary,
        llm,
        max_retries=max_retries,
        base_delay=base_delay,
    )
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
    # print("\n"*1)
    """
    Step 5: Column Header Categorization
    """
    print("=" * 64)
    step_name = "Column Header Categorization"
    print(COLOR_START+step_name+COLOR_END)
    mapping_info = run_with_retry(
        s_pk_get_col_mapping,
        md_table_aligned,
        llm,
        max_retries=max_retries,
        base_delay=base_delay,
    )
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
    # print("\n"*1)
    """
    Step 6: Rough Task Allocation (Preferably hidden from Users)
    """
    parameter_type_count = list(col_mapping.values()).count("Parameter type")
    parameter_unit_count = list(col_mapping.values()).count("Parameter unit")
    parameter_value_count = list(col_mapping.values()).count("Parameter value")
    parameter_pvalue_count = list(col_mapping.values()).count("P value")
    need_get_unit = True
    need_split_col = False
    need_match_drug = True
    need_match_patient = True
    unit_auto_parse = False
    # 0302 unit parse speedup!
    # if parameter_unit_count == 0 and parameter_type_count == 1:
    #     parameter_type_key = None
    #     for key, value in col_mapping.items():
    #         if value == "Parameter type":
    #             parameter_type_key = key
    #             break
    #     df_table_aligned = markdown_to_dataframe(md_table_aligned)
    #     df_table_aligned["Parameter unit"] = df_table_aligned[parameter_type_key].str.extract(r'\((.*?)\)')[0].fillna(
    #         "N/A")
    #     df_table_aligned[parameter_type_key] = df_table_aligned[parameter_type_key].str.replace(r'\(.*?\)', '',
    #                                                                                             regex=True).str.strip()
    #     matched_count = (df_table_aligned["Parameter unit"] != "N/A").sum()
    #     total_count = len(df_table_aligned)
    #     if matched_count <= total_count / 2:
    #         df_table_aligned.drop(columns=["Parameter unit"], inplace=True)
    #     else:
    #         col_mapping["Parameter unit"] = "Parameter unit"
    #         parameter_unit_count += 1
    #         unit_auto_parse = True
    #     md_table_aligned = dataframe_to_markdown(df_table_aligned)
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
    if unit_auto_parse:
        print("Auto unit parsing is complete!")
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
    Step 7: Sub-table Creation
    """
    print("=" * 64)
    step_name = "Sub-table Creation"
    print(COLOR_START + step_name + COLOR_END)
    if need_split_col:
        split_returns = run_with_retry(
            s_pk_split_by_cols,
            md_table_aligned,
            col_mapping,
            llm,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        if split_returns is None:
            return None
        md_table_list, res_split, content_split, usage_split, truncated_split = split_returns
        step_list.append(step_name)
        res_list.append(res_split)
        content_list.append(content_split)
        content_list_clean.append(clean_llm_reasoning(content_split))
        usage_list.append(usage_split)
        truncated_list.append(truncated_split)
        # print("\n"*1)
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
    Step 8: Unit Extraction
    """
    type_unit_list = []
    type_unit_cache = {}
    round = 1
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
                unit_info = run_with_retry(
                    s_pk_get_parameter_type_and_unit,
                    md_table_aligned,
                    col_mapping,
                    md,
                    description,
                    llm,
                    max_retries=max_retries,
                    base_delay=base_delay,
                )
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
    Step 9: Unit Extraction (Final)
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
    Step 10: Drug Matching
    """
    drug_list = []
    round = 1
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
            drug_match_info = run_with_retry(
                s_pk_match_drug_info,
                md_table_aligned,
                description,
                md,
                md_table_drug,
                llm,
                max_retries=max_retries,
                base_delay=base_delay,
            )
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
    round = 1
    if need_match_patient is False:
        for md in md_table_list:
            df = markdown_to_dataframe(md)
            row_num = df.shape[0]
            df_expanded = pd.concat([markdown_to_dataframe(md_table_patient)] * row_num, ignore_index=True)
            patient_list.append(dataframe_to_markdown(df_expanded))
    else:
        for md in md_table_list:
            print("=" * 64)
            step_name = "Population Matching" + f" (Trial {str(round)})"
            round += 1
            print(COLOR_START + step_name + COLOR_END)
            patient_match_info = run_with_retry(
                s_pk_match_patient_info,
                md_table_aligned,
                description,
                md,
                md_table_patient,
                llm,
                max_retries=max_retries,
                base_delay=base_delay,
            )
            if patient_match_info is None:
                return None
            patient_match_list, res_patient_match, content_patient_match, usage_patient_match, truncated_patient_match = patient_match_info
            df_table_patient = markdown_to_dataframe(md_table_patient)
            df_table_patient = pd.concat(
                [df_table_patient, pd.DataFrame([{'Population': 'ERROR', 'Pregnancy stage': 'ERROR', 'Subject N': 'ERROR'}])],
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
    Step 12: Parameter Value Extraction
    """
    value_list = []
    round = 1
    for md in md_table_list:
        print("=" * 64)
        step_name = "Parameter Value Extraction" + f" (Trial {str(round)})"
        round += 1
        print(COLOR_START + step_name + COLOR_END)
        value_info = run_with_retry(
            s_pk_get_parameter_value,
            md_table_aligned,
            description,
            md,
            llm,
            max_retries=max_retries,
            base_delay=base_delay,
        )
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
    Step 13: Parameter Value Extraction (Final)
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
    Step 14: Assembly
    """
    df_list = []
    assert len(drug_list) == len(patient_list) == len(type_unit_list) == len(value_list)
    for i in range(len(drug_list)):
        df_drug = markdown_to_dataframe(drug_list[i])
        df_table_patient = markdown_to_dataframe(patient_list[i])
        df_type_unit = markdown_to_dataframe(type_unit_list[i])
        df_value = markdown_to_dataframe(value_list[i])
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
    Step 15: Post-Processing
    """

    """Delete ERROR rows"""
    df_combined = df_combined[df_combined.ne("ERROR").all(axis=1)]
    """if Value == "N/A", Summary Statistics must be "N/A"。"""
    df_combined.loc[
        (df_combined["Main value"] == "N/A"), "Statistics type"] = "N/A"
    """if Lower limit & High limit == "N/A", Interval type must be "N/A"。"""
    df_combined.loc[
        (df_combined["Lower bound"] == "N/A") & (df_combined["Upper bound"] == "N/A"), "Interval type"] = "N/A"
    """if Variation value == "N/A", Variation type must be "N/A"。"""
    df_combined.loc[
        (df_combined["Variation value"] == "N/A"), "Variation type"] = "N/A"
    df_combined = df_combined.reset_index(drop=True)
    """replace empty by N/A"""
    df_combined.replace(r'^\s*$', 'N/A', regex=True, inplace=True)
    """replace n/a by N/A"""
    df_combined.replace("n/a", "N/A", inplace=True)

    """Remove non-digit rows"""
    columns_to_check = ["Main value", "Statistics type", "Variation type", "Variation value",
                        "Interval type", "Lower bound", "Upper bound", "P value"]

    def contains_number(s):
        return any(char.isdigit() for char in s)

    df_combined = df_combined[df_combined[columns_to_check].apply(lambda row: any(contains_number(str(cell)) for cell in row), axis=1)]
    df_combined = df_combined.reset_index(drop=True)

    """ Merge """

    df = df_combined.copy()

    df.replace("N/A", pd.NA, inplace=True)

    group_columns = ["Drug name", "Analyte", "Specimen", "Population", "Pregnancy stage", "Subject N", "Parameter type",
                     "Parameter unit"]
    grouped = df.groupby(group_columns, dropna=False)

    merged_rows = []
    for _, group in grouped:
        group = group.reset_index(drop=True)
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
            else:
                merged_rows.append(row1)
                merged_rows.append(row2)

        for i in range(len(group)):
            if i not in used_indices:
                merged_rows.append(group.iloc[i])

    df_merged = pd.DataFrame(merged_rows, columns=df.columns)
    df_merged.fillna("N/A", inplace=True)

    df_combined = df_merged
    df_combined = df_combined.reset_index(drop=True)

    """Remove duplicate"""
    df_combined = df_combined.drop_duplicates()
    df_combined = df_combined.reset_index(drop=True)

    """delete 'fill in subject N as value error', this implementation is bad, still looking for better solutions"""
    df_combined = df_combined[df_combined["Subject N"] != df_combined["Main value"]]
    # df_combined = df_combined[~df_combined["Value"].isin(markdown_to_dataframe(md_table_patient)["Subject N"].to_list())]
    df_combined = df_combined.reset_index(drop=True)

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
    df_combined = df_combined.reset_index(drop=True)

    """Rename col names"""
    column_mapping = {
        "Parameter unit": "Unit",
        "Main value": "Value",
        "Statistics type": "Summary Statistics",
        "Lower bound": "Lower limit",
        "Upper bound": "High limit",
    }
    df_combined = df_combined.rename(columns=column_mapping)

    print("=" * 64)
    step_name = "Post-Processing"
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


