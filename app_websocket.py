from datetime import datetime
from typing import List, Optional
from fastapi import WebSocket
import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage

logger = logging.getLogger(__name__)

async def curation_work_websocket(
    pmid: str,
    pipelines: List[str],
):
    pass


async def curation_websocket(websocket: WebSocket):
    token_usage_accumulate: dict[str, int] = {**DEFAULT_TOKEN_USAGE}
    async def step_callback(
        step_name: Optional[str]=None,
        step_description: Optional[str]=None,
        step_output: Optional[str]=None,
        step_reasoning: Optional[str] = None,
        token_usage: Optional[dict[str, int]] = None,
        step_result: Optional[str] = None,
    ):
        nonlocal token_usage_accumulate
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            token_usage_accumulate = increase_token_usage(token_usage_accumulate, token_usage)
            logger.info(
                f"overall total tokens: {token_usage_accumulate['total_tokens']}, overall prompt tokens: {token_usage_accumulate['prompt_tokens']}, overall completion tokens: {token_usage_accumulate['completion_tokens']}"
            )
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
            message = f"##### [{datetime.now():%Y-%m-%d %H:%M:%S}] {step_name}"
            await websocket.send_json({"type": "step_name", "message": message})
        if step_description is not None:
            logger.info(step_description)
        if step_output is not None:
            logger.info(step_output)
            await websocket.send_json({"type": "step_output", "message": step_output})
        if step_reasoning is not None:
            logger.info(f"\n\n{step_reasoning}\n\n")
            await websocket.send_json({"type": "step_reasoning", "message": step_reasoning})
        if step_result is not None:
            logger.info(f"Result: {step_result}")
            await websocket.send_json({"type": "step_result", "message": step_result})


    await websocket.accept()
    try:
        data = await websocket.receive_json()
        print(data)
        await websocket.send_json(data)
    except Exception as e:
        print(e)
    finally:
        await websocket.close()
