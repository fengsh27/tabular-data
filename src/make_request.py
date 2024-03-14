
import requests
import logging

from tenacity import (
    retry, stop_after_attempt, wait_exponential
)
from ratelimit import (
    limits, sleep_and_retry
)

logger = logging.getLogger(__name__)

# Rate limit configuration (e.g., 5 requests per second)
@limits(calls=3, period=1)
@sleep_and_retry
@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def make_get_request(url, *args, **kwargs):
    logger.info(f"make get request to {url}")
    print(f"make get request to {url}")
    return requests.get(url, *args, **kwargs)

