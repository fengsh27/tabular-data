
import requests
import logging

logger = logging.getLogger(__name__)

def make_get_request(url, *args, **kwargs):
    logger.info(f"make get request to {url}")
    print(f"make get request to {url}")
    return requests.get(url, *args, **kwargs)

