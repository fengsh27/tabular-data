import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from TabFuncFlow.utils.table_utils import (
    fix_col_name,
    markdown_to_dataframe,
    dataframe_to_markdown,
)
from TabFuncFlow.operations.f_transpose import f_transpose
from TabFuncFlow.utils.table_utils import (
    deduplicate_headers,
    fill_empty_headers,
    remove_empty_col_row,
)
from extractor.agents.common_agent.common_agent import RetryException
from extractor.agents.pk_individual.pk_ind_common_agent import PKIndCommonAgentResult

logger = logging.getLogger(__name__)

PARAMETER_TYPE_ALIGN_PROMPT = ChatPromptTemplate.from_template("""
There is now a table related to pharmacokinetics (PK). 
{md_table_individual}
The extracted table headers are as follows:
{md_table_individual_header}
Carefully examine the pharmacokinetics (PK) table and follow these steps to determine how the PK parameter type is represented:
(1) Identify how the PK parameter type (e.g., Cmax, tmax, t1/2, etc.) is structured in the table:
Please answer with a JSON object with the single key "col_name", like this: {{"col_name": "column_name"}}, where
col_name is the column name, it represents the PK parameter type serves as the row header or is listed under the specific column. If the PK parameter type is represented as column headers, use null.
(2) Ensure a thorough analysis of the table structure before selecting your answer.
""")


class ParameterTypeAlignResult(PKIndCommonAgentResult):
    col_name: str | None = Field(
        description="""The name of the column representing the PK parameter type, which serves as the row header or is listed under a specific column. 
If the PK parameter type is represented as column headers, this value will be None."""
    )

def get_parameter_type_align_prompt(md_table_individual: str):
    df = markdown_to_dataframe(md_table_individual)
    headers = [f'"{col}"' for col in df.columns]
    headers = ",".join(headers)
    return PARAMETER_TYPE_ALIGN_PROMPT.format(
        md_table_individual=md_table_individual,
        md_table_individual_header=headers,
    )

# Transposing on `col_name` turns that column's values into the new header row, so they
# must behave like parameter-type labels: (mostly) distinct, and enough of them to make a
# table. A column whose values repeat (a drug name per group of subjects, a dose, a subject
# index) is not a parameter-type column - PMID 33253437's blank-headed first column held 6
# drug names over 47 rows and the transpose wrecked the table. The same blank header
# ("Unnamed: 0") on PMID 34746508 holds 9 distinct parameter labels and MUST be transposed
# (cases are the columns there), which is why the check is on the values, not the name.
# This replaces the name-based guard from commit 6e8669a.
MIN_DISTINCT_LABEL_RATIO = 0.5
MIN_TRANSPOSED_COLUMNS = 3


def _label_stats(series) -> tuple[int, int]:
    """(non-empty values, distinct non-empty values) of the would-be header column."""
    vals = [str(v).strip() for v in series]
    vals = [v for v in vals if v and v.lower() != "nan"]
    return len(vals), len(set(vals))


def check_transpose_result(col_name: str, df_table, df_transposed) -> None:
    """Raise RetryException when transposing on `col_name` cannot have been right."""
    if col_name not in df_table.columns:
        raise RetryException(
            f"{col_name!r} is not a column of the table; answer with one of the listed column "
            "names exactly, or null if the parameter types are already the column headers."
        )
    n, distinct = _label_stats(df_table[col_name])
    if n == 0 or distinct / n < MIN_DISTINCT_LABEL_RATIO:
        raise RetryException(
            f"Column {col_name!r} cannot be the parameter-type column: its {n} values hold only "
            f"{distinct} distinct label(s), so they repeat like a group label (drug, dose, subject), "
            "not like PK parameter types. Answer with the column that really lists the parameter "
            "types, or null if the parameter types are already the column headers."
        )
    if df_transposed.shape[1] < MIN_TRANSPOSED_COLUMNS:
        raise RetryException(
            f"Transposing on column {col_name!r} leaves only {df_transposed.shape[1]} column(s); "
            "that is not a usable table. Answer null if the parameter types are already the column "
            "headers."
        )


def try_fix_parameter_type_align(res: ParameterTypeAlignResult, md_table_individual: str) -> str:
    """Last-attempt fallback once the retries are used up: keep the table as it is (the
    pre-2026-10 behaviour for a rejected answer) rather than lose the paper."""
    logger.warning(f"Parameter-type align: giving up on col_name={res.col_name!r}; table left untransposed")
    return dataframe_to_markdown(markdown_to_dataframe(md_table_individual))


def post_process_parameter_type_align(
    res: ParameterTypeAlignResult, md_table_individual: str
):
    """Please note: the condition for transposing is the opposite of that for PK summary.

    col_name None: the parameter types are already column headers - table unchanged.
    col_name given: transpose on it, then `check_transpose_result` - a bad answer raises
    RetryException so the model is asked again with the reason (and after the retries
    `try_fix_parameter_type_align` keeps the table untransposed)."""
    df_table = markdown_to_dataframe(md_table_individual)
    if res.col_name is None:
        return dataframe_to_markdown(df_table)
    col_name = fix_col_name(res.col_name, md_table_individual)
    df_transposed = f_transpose(df_table)
    check_transpose_result(col_name, df_table, df_transposed)
    return deduplicate_headers(
        fill_empty_headers(fill_empty_headers(remove_empty_col_row(dataframe_to_markdown(df_transposed))))
    )
