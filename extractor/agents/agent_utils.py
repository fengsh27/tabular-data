import re
from typing import Optional, Protocol
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)

from extractor.database.pmid_db import PMIDDB
from extractor.pmid_extractor.article_retriever import ArticleRetriever
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor
from extractor.utils import convert_html_to_text_no_table, remove_references


class StepCallback(Protocol):
    def __call__(
        self,
        step_name: Optional[str] = None,
        step_description: Optional[str] = None,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ) -> None: ...


def display_md_table(md_table):
    """
    Adds labels to each row in a Markdown table.

    The first row is prefixed with "col:", and subsequent rows are prefixed with "row 1:", "row 2:", etc.

    :param md_table: Markdown table as a string
    :return: Modified Markdown table with labeled rows
    """
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return md_table  # Not enough lines to form a valid table

    labeled_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            headers = line.split("|")
            headers = [f'"{header.strip()}"' for header in headers if header.strip()]
            labeled_lines.append(f"col: | {' | '.join(headers)} |")
        elif i == 1 and re.match(r"\|\s*-+\s*\|", line):
            labeled_lines.append(line)  # Keep separator unchanged
        else:
            labeled_lines.append(f"row {i - 2}: {line}")

    return "\n".join(labeled_lines)


PROMPT_TOKENS="prompt_tokens"
COMPLETION_TOKENS="completion_tokens"
TOTAL_TOKENS="total_tokens"

DEFAULT_TOKEN_USAGE = {
    "total_tokens": 0,
    "completion_tokens": 0,
    "prompt_tokens": 0,
}


def increase_token_usage(
    token_usage: Optional[dict] = None,
    incremental: dict = {**DEFAULT_TOKEN_USAGE},
):
    if token_usage is None:
        token_usage = {**DEFAULT_TOKEN_USAGE}
    token_usage["total_tokens"] += incremental["total_tokens"]
    token_usage["completion_tokens"] += incremental["completion_tokens"]
    token_usage["prompt_tokens"] += incremental["prompt_tokens"]

    return token_usage


def extract_integers(text):
    """
    Extract only "pure integers":
    - Skip numbers with decimal points (including forms like 3.14, 1.0, .99, etc.)
    - Skip numbers immediately followed (possibly with spaces) by '%' or '％' (e.g., 8%, 8 %)
    - Keep only integers like 42, 100, etc.
    - Remove duplicates and exclude 0
    """
    # 1) (?<![\d.]) ensures that the left side is not a digit or a dot.
    # 2) (\d+) matches a sequence of digits.
    # 3) (?![\d.]| *[%％]) ensures that the right side is not a digit, a dot,
    #    or zero or more spaces followed by '%' or '％'.
    pattern = r"(?<![\d.])(\d+)(?![\d.]| *[%％])"

    # Convert matched digit-strings to integers, use a set to remove duplicates,
    # then remove 0 if present
    return list(set(int(num_str) for num_str in re.findall(pattern, text)) - {0})


def from_system_template(template, **kwargs):
    prompt_template = PromptTemplate.from_template(template, **kwargs)
    message = SystemMessagePromptTemplate(prompt=prompt_template)
    return ChatPromptTemplate.from_messages([message])


def extract_pmid_info_to_db(
    pmid: str, 
    pmid_db: PMIDDB, 
    html_content: str | None = None
) -> tuple[str, str, str, str, list[dict], list[str]]:
    """
    Extract pmid info from database or web, and save to database.

    Returns:
        tuple[str, str, str, str, list[dict], list[str]]: pmid, title, abstract, full_text, tables, sections
    """
    info = pmid_db.select_pmid_info(pmid)
    if info is not None:
        return info
    if html_content is None:
        retriever = ArticleRetriever()
        res, html_content, code = retriever.request_article(pmid)
        if not res:
            return None, None, None, None, None, None
    extractor = HtmlTableExtractor()
    tables = extractor.extract_tables(html_content)
    sections = extractor.extract_sections(html_content)
    abstract = extractor.extract_abstract(html_content)
    title = extractor.extract_title(html_content)
    full_text = convert_html_to_text_no_table(html_content)
    full_text = remove_references(full_text)
    pmid_db.insert_pmid_info(pmid, title, abstract, full_text, tables, sections)
    return pmid, title, abstract, full_text, tables, sections

def get_reasoning_process(result: tuple) -> str:
    res = result[0]
    if hasattr(res, "reasoning_process") and res.reasoning_process is not None:
        return res.reasoning_process
    reasoning_process = result[3] if len(result) == 4 else "N/A"
    return reasoning_process
