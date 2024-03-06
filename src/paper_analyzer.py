
import logging
from typing import Optional
from src.request_openai import request_to_chatgpt
from src.request_paper import (
    request_paper,
    convert_html_to_text
)

logger = logging.getLogger(__name__)

template_prompts = """
Please act as a Medical Assistant, extract the following information from the provided biomedical paper and output as a table in markdown format:
1. Population: Describe the patient age distribution, including categories such as "pediatric," "adults," "old adults," "maternal," "fetal," "neonate," etc.
2. Data Source: Identify where the data originated, such as "Statewide surveillance in Victoria, Australia between 1 March 2002 and 31 August 2004." 
3. Inclusion Criteria: Specify the cases that were included in the paper.
4. Exclusion Criteria: Specify the cases that were excluded from the paper.
5. Study Type: Determine the type of study conducted, such as "clinical trial," "pharmacoepidemiology," etc.
6. Outcome: Describe the main outcome of the study, for example, "the disappearance of proteinuria."
7. Study Design: Explain how the study was designed, e.g., "prospective single-arm open-labeled pilot trial."
8. Sample Size: Provide the sample size used in the study.
Please note: only output markdown table without any other characters and embed the text in code chunks, so it won't convert to HTML in the assistant.,
Now, you don't need to response until I post the paper.
"""

def populate_paper_to_template(pmid: str, prompts: Optional[str] = None):
    (res, html_content, code) = request_paper(pmid)
    if not res:
        return (res, html_content, 0)
    text = convert_html_to_text(html_content)
    # remove "References"
    ix = text.rfind("References")
    if ix < 0:
        # not find Reference
        logger.warn(f"Can't find 'References' in paper {pmid}")
    else:
        text = text[:ix]
    
    try:
        the_prompts = {'role': "user", "content": prompts if prompts is not None else template_prompts}
        res = request_to_chatgpt([the_prompts], text)
        return res
    except Exception as e:
        logger.error(e)
        return (False)

    

    
