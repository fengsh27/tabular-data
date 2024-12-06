
import requests
from requests import Response
import logging
from typing import Optional, Dict
import os

from tenacity import (
    retry, stop_after_attempt, wait_exponential
)
from ratelimit import (
    limits, sleep_and_retry
)

logger = logging.getLogger(__name__)

# Rate limit configuration (e.g., 3 requests per second)
@limits(calls=3, period=1)
@sleep_and_retry
@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def make_get_request(
    url, 
    headers: Dict[str, str], 
    allow_redirects: bool, 
    cookies: Dict[str, str], 
    *args, **kwargs
    ) -> Response:
    logger.info(f"make get request to {url}")
    print(f"make get request to {url}")
    res = requests.get(
        url, 
        headers=headers, 
        cookies=cookies, 
        allow_redirects=allow_redirects,
        *args, **kwargs
    )
    
    return res

@limits(calls=3, period=1)
@sleep_and_retry
@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def make_article_request(url: str, fn: str, img_fn: Optional[str] = None) -> Response:
    baseurl = os.environ.get("BASE_URL", "http://127.0.0.1:3000")
    api_path = '/api/article'
    the_url = f"{baseurl}{api_path}"
    logger.info(f"make article({url}) request to {the_url}")
    params = {
        "url": url,
        "output": fn,
        "png_output": img_fn,
    } if img_fn is not None else {
        "url": url,
        "output": fn,
    }
    res = requests.get(
        the_url,
        params=params,
    )
    return res

