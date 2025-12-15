import requests
import json
import dotenv
import os
import logging

from .test_gpt_oss_2 import msg

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

def test_gpt_oss_3():
    url = os.getenv("OLLAMA_BASE_URL")
    payload = {
        "model": "qwen3:235b", # "gpt-oss:20b",     # or any model installed in your local ollama
        "prompt": msg+msg+msg+msg+msg,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 4096
        }
    }
    response = requests.post(f"{url}/api/generate", json=payload)
    res = response.json()
    logger.info(res)
