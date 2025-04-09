from typing import Optional
import pandas as pd
import logging
from langchain_openai.chat_models.base import BaseChatOpenAI

from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflow
from extractor.article_retriever import ArticleRetriever
from extractor.html_table_extractor import HtmlTableExtractor
from extractor.table_utils import select_pk_summary_tables

logger = logging.getLogger(__name__)


def retrieve_article(pmid: str):
    retrieve_article = ArticleRetriever()
    res, html_content, _ = retrieve_article.request_article(pmid)
    if not res:
        error_msg = f"Failed to retrieve article. \n {html_content}"
        logger.error(error_msg)
        return res, error_msg
    return res, html_content


def extract_pk_summary(
    pmid: str, llm: BaseChatOpenAI
) -> tuple[bool, str, pd.DataFrame | None, int | None]:
    total_token_usage = {**DEFAULT_TOKEN_USAGE}

    def output_step(
        step_name: Optional[str] = None,
        step_description: Optional[str] = None,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ):
        nonlocal total_token_usage
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if step_description is not None:
            logger.info(step_description)
        if token_usage is not None:
            usage_str = f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            logger.info(usage_str)
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
            usage_str = f"overall total tokens: {total_token_usage['total_tokens']}, overall prompt tokens: {total_token_usage['prompt_tokens']}, overall completion tokens: {total_token_usage['completion_tokens']}"
            logger.info(usage_str)
        if step_reasoning_process is not None:
            logger.info(f"\n\n{step_reasoning_process}\n\n")
        if step_output is not None:
            logger.info(step_output)

    # step 1, request paper from pmid
    retriever = ArticleRetriever()
    res, html_content, code = retriever.request_article(pmid)
    if not res:
        return res, html_content, None, None

    # step 2: extract tables from paper
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    if len(tables) == 0:
        return False, "No table found", None, None

    # step 3: select pk summary tables
    selected_tables, indexes, token_usage = select_pk_summary_tables(tables, llm)
    if len(selected_tables) == 0:
        return False, "No table found related to PK Summary", None, None

    # step 4: extract pk summary data
    dfs = []
    for table in selected_tables:
        df_table = table["table"]
        caption = "\n".join([table["caption"], table["footnote"]])
        workflow = PKSumWorkflow(llm=llm)
        workflow.build()
        df = workflow.go_md_table(
            md_table=dataframe_to_markdown(df_table),
            caption_and_footnote=caption,
            step_callback=output_step,
        )
        dfs.append(df)
    df_combined = (
        pd.concat(dfs, axis=0).reset_index(drop=True)
        if len(dfs) > 0
        else pd.DataFrame()
    )

    return (
        True,
        f"{pmid}: Completed extracting PK Summary data",
        df_combined,
        total_token_usage,
    )
