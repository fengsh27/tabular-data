from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd
import logging

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import display_md_table
from extractor.agents.pk_specimen_individual.pk_spec_ind_common_agent import (
    RetryException,
    PKSpecIndCommonAgentResult,
)

logger = logging.getLogger(__name__)

TIME_AND_UNIT_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

From the article above, I have extracted the following information to create Subtable 1, where each row represents a unique combination of "Patient ID" - "Specimen" - "Sample N" - "Sample time" as follows:
{processed_md_table_specimen}

Carefully analyze the table and follow these steps:  
(1) For each row in Subtable 1, add two more columns [Sample time, Time unit, Source text].  
    - **Sample time:** The specific moment (numerical or time range) when the specimen is sampled. **Keep the value(s) numerical! Keep the value(s) numerical! Keep the value(s) numerical!**  
    e.g., "0", "24", "0, 2, 4", "0-2", "0-2, 2-4, 4-6"
    - **Time Unit:** The unit corresponding to the sample time (e.g., "Second", "Minute", "Hour", "Day").  
    - **Source text**: The original sentence or excerpt from the source document where the data was reported. This field provides context and traceability, ensuring that each data point can be verified against its original description in the literature. Use "N/A" if no source can be found.
(2) List each unique combination in the format of a list of lists, using Python string syntax. Your answer should be like this:  
`[["0-1", "Hour", "... the sentence from the article ..."], ["10", "Minute", "... the sentence from the article ..."], ["0, 2, 4, 6", "Hour", "... the sentence from the article ..."], ["0-2, 2-4, 4-6, 6-8", "Hour", "... the sentence from the article ..."], ["N/A", "N/A", "... the sentence from the article ..."]]` (example)
    - If a field contains multiple values separated by commas, preserve the comma-separated string as a single element.  
    - If a range of time is given (like "0-1"), treat the entire range string as one item.  
    - If no valid data is available, use "N/A".
(3) Strictly ensure that you process only rows 0 to {md_table_specimen_max_row_index} from the Subtable 1 (which has {md_table_specimen_row_num} rows in total). 
    - The number of processed rows must **exactly match** the number of rows in the Subtable 1—no more, no less.  
(4) Verify the source of each [Sample time, Time unit, Source text] combination before including it in your answer.  
(5) **Absolutely no calculations are allowed—every value must be taken directly from the table without any modifications.** 
(6) **If no valid [Sample time, Time unit, Source text] combinations are found, return the default output:**  
`[["N/A", "N/A", "N/A"]]`
""")


def get_time_and_unit_prompt(title: str, full_text: str, md_table_specimen: str):
    row_num = markdown_to_dataframe(md_table_specimen).shape[0]
    return TIME_AND_UNIT_PROMPT.format(
        title=title,
        full_text=full_text,
        processed_md_table_specimen=display_md_table(md_table_specimen),
        md_table_specimen_max_row_index=row_num - 1,
        md_table_specimen_row_num=row_num,
    )


class TimeAndUnitResult(PKSpecIndCommonAgentResult):
    """Time and Unit Extraction Result"""

    times_and_units: list[list[str]] = Field(
        description="a list of lists, where each inner list represents [Sample time, Time unit, Source text]"
    )


def post_process_time_and_unit(
    res: TimeAndUnitResult,
    md_table_specimen: str,
) -> str:
    match_list = res.times_and_units
    expected_rows = markdown_to_dataframe(md_table_specimen).shape[0]
    if all(x == match_list[0] for x in match_list):
        # expand to expect_rows
        match_list = [match_list[0]] * expected_rows

    df_table = pd.DataFrame(match_list, columns=["Sample time", "Time unit", "Source text"])
    if df_table.shape[0] != expected_rows:
        if expected_rows == 1:
            if len(df_table["Time unit"].unique()) == 1:
                combined_time = ", ".join(df_table["Sample time"].astype(str))
                time_unit = df_table["Time unit"].iloc[0]
                unique_sources = df_table["Source text"].dropna().astype(str).unique()
                combined_source = "\n".join(unique_sources)
                df_table = pd.DataFrame([[combined_time, time_unit, combined_source]], columns=["Sample time", "Time unit", "Source text"])
                return dataframe_to_markdown(df_table)
        error_msg = (
            "Wrong answer example:\n"
            + str(match_list)
            + f"\nWhy it's wrong:\nMismatch: Expected {expected_rows} rows, but got {df_table.shape[0]} extracted matches."
        )
        logger.error(error_msg)
        raise RetryException(error_msg)

    if "|" in dataframe_to_markdown(df_table):
        for row_idx in df_table.index:
            for col in df_table.columns:
                cell = df_table.at[row_idx, col]
                if isinstance(cell, str) and "|" in cell:
                    updated = "Content from Table: " + cell.replace("|", "-")
                    df_table.at[row_idx, col] = updated

    return dataframe_to_markdown(df_table)
