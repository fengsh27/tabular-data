import pytest
import os
import requests
from dotenv import load_dotenv
import json
import logging

from .test_gpt_oss_with_request_const import verification_prompt
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPEVerificationStepResult

load_dotenv()
logger = logging.getLogger(__name__)

def test_gpt_oss_with_requests():
    schema = PKPEVerificationStepResult.model_json_schema()
    schema.pop("title", None)
    url = os.getenv("OLLAMA_BASE_URL")
    payload = {
        "model": "gpt-oss:20b",
        "prompt": verification_prompt,
        "stream": False,
        "think": "low", # False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 8192,
        },
    }
    """
    payload = {
        "model": "gpt-oss:20b",     # or any model installed in your local ollama
        # "model": "qwen3:30b",
        "prompt": verification_prompt,
        "think": False,
        "stream": False,
        # "format": schema, 
        "options": {
            "num_ctx": 16384,
            "num_predict": 4096,
        }
    }""" 
    response = requests.post(f"{url}/api/generate", json=payload)
    response.raise_for_status()

    res = response.json()
    logger.info(res.get("response"))
    logger.info(
        "done=%s done_reason=%s response_len=%d thinking_len=%d error=%s",
        res.get("done"),
        res.get("done_reason"),
        len(res.get("response") or ""),
        len(res.get("thinking") or ""),
        res.get("error")
    )

