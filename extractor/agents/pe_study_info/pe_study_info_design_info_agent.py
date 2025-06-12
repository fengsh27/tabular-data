from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import pandas as pd

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.pe_study_info.pe_study_info_common_agent import (
    PEStudyInfoCommonAgentResult,
    RetryException,
)

STUDY_DESIGN_PROMPT = ChatPromptTemplate.from_template("""
{title}
{full_text}
Read the article and answer the following:

(1) Summarize the Article in the format of [[Study type, Study design, Data source]]. 
    - **Study type**: This deals with the area of interest related to a particular article; 
        choose from Pharmacoepidemiology / Clinical Trials / Pharmacokinetics / Pharmacodynamics / Pharmacogenetics. 
    - **Study design**: Identify the study design described in the article. Common examples include:
        - *Prospective cohort study*
        - *Retrospective cohort study*
        - *Randomized controlled trial (RCT)*
        - *Double-blind randomized trial*
        - *Case-control study*
        - *Cross-sectional study*
        - *Systematic review and meta-analysis*
        - *Open-label study*
        - *Nested case-control study*
        - *Pilot study*
        - *Chart review*
        - *Observational study*  
        If multiple designs are mentioned (e.g., "prospective, randomized, double-blind"), list them as one string in the same order.
    - **Data source**: The primary locations(s) where data is accessed from or primary site where a study was conducted (ex: hospitals, database, geographic locations etc.)
(2) List the combination in Python list-of-lists syntax, like this:  
    [["Pharmacoepidemiology", "Prospective Randomized Double-blinded Invesigation", "OSUMC"]] (example)  
""")


INSTRUCTION_PROMPT = "Do not give the final result immediately. First, explain your thought process, then provide the answer."


class DesignInfoResult(PEStudyInfoCommonAgentResult):
    """Design Information Result"""

    study_design_combinations: list[list[str]] = Field(
        description="a list of lists, but only has one combination of [Study type, Study design, Data source]"
    )


def post_process_study_design_info(
    res: DesignInfoResult,
):
    if res.study_design_combinations is None:
        raise ValueError("Empty study design combinations")

#     if type(res.study_design_combinations) != list or len(res.study_design_combinations) == 0:
#         raise RetryException(f"""
# Wrong answer: {res.study_design_combinations}, if the table does not explicitly mention any [Study type, Study Design, Data source], please leave it with [["N/A", "N/A", "N/A"]].
# """)

    df_table = pd.DataFrame(
        res.study_design_combinations, columns=["Study type", "Study design", "Data source"]
    )

    if "|" in dataframe_to_markdown(df_table):
        for row_idx in df_table.index:
            for col in df_table.columns:
                cell = df_table.at[row_idx, col]
                if isinstance(cell, str) and "|" in cell:
                    updated = "Content from Table: " + cell.replace("|", "-")
                    df_table.at[row_idx, col] = updated

    return dataframe_to_markdown(df_table)
