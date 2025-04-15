from typing import List
from pandas import DataFrame
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
import logging

from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
    PKSumCommonAgent,
    RetryException,
)
from extractor.prompts_utils import generate_tables_prompts

logger = logging.getLogger(__name__)

SELECT_PK_TABLES_PROMPT = ChatPromptTemplate.from_template("""
Analyze the provided content and identify all tables related to pharmacokinetics (PK), including but not limited to tables covering Absorption, Distribution, Metabolism, and Excretion (ADME) properties. 

Focus particularly on tables that report:
- Drug concentrations in body fluids (plasma, urine, cord blood, etc.)
- PK parameters (e.g., AUC, Cmax, t½)
- ADME characteristics
                                                           
**Exclude tables that primarily present regression analyses or statistical modeling of pharmacokinetic parameters.**

Return the results as a Python list of table indexes in this exact format:
[table index 1, table index 2, ...]

The content including markdown table to analyze:
{table_content}
""")


class TablesSelectionResult(PKSumCommonAgentResult):
    """Tables Selection Result"""

    selected_table_indexes: List[str] = Field(
        description="""a list of selected table indexes, such as ["1", "2", "3"]"""
    )


def post_process_selected_table_ids(
    res: TablesSelectionResult,
    html_tables: list[dict[str, str | DataFrame]],
):
    ids = res.selected_table_indexes
    if ids is None:
        raise ValueError("Invalid selected tables")

    indices = []
    for id in ids:
        id = int(id)
        if id < 0 or id > len(html_tables):
            raise RetryException(
                "Please generate valid table id, wrong answer example: `{id}`"
            )
        indices.append(id)

    tables = []
    for ix in indices:
        tables.append(html_tables[ix])

    return tables


def select_pk_summary_tables(html_tables: list[dict[str, str | DataFrame]], llm):
    table_content = generate_tables_prompts(html_tables, True)
    system_prompt = SELECT_PK_TABLES_PROMPT.format(table_content=table_content)

    agent = PKSumCommonAgent(llm=llm)
    res, tables, token_usage, _ = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=INSTRUCTION_PROMPT,
        schema=TablesSelectionResult,
        post_process=post_process_selected_table_ids,
        html_tables=html_tables,
    )

    logger.info(f"Selected tables (indices): {res.selected_table_indexes}")

    return tables, res.selected_table_indexes, token_usage
