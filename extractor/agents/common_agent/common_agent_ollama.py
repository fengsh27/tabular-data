from typing import Any, Callable, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama.chat_models import ChatOllama
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from extractor.agents.agent_utils import COMPLETION_TOKENS, DEFAULT_TOKEN_USAGE, PROMPT_TOKENS, TOTAL_TOKENS
from extractor.llm_utils import get_format_instructions, structured_output_llm
from extractor.utils import escape_braces_for_format
from .common_agent import (
    CommonAgent,
    RetryException,
)
from .common_agent_2steps import FINAL_STEP_SYSTEM_PROMPTS, CommonAgentTwoChainSteps, CommonAgentTwoSteps

logger = logging.getLogger(__name__)

class CommonAgentOllama(CommonAgent):
    def __init__(self, llm: ChatOllama):
        super().__init__(llm)

    @staticmethod
    def normalize_token_usage(token_usage):
        usage = token_usage
        if not isinstance(token_usage, dict):
            usage = vars(token_usage)
        if not PROMPT_TOKENS in usage and 'input_tokens' in usage:
            usage[PROMPT_TOKENS] = usage['input_tokens']
        if not COMPLETION_TOKENS in usage and 'output_tokens' in usage:
            usage[COMPLETION_TOKENS] = usage['output_tokens']
        
        return {
            PROMPT_TOKENS: usage[PROMPT_TOKENS],
            COMPLETION_TOKENS: usage[COMPLETION_TOKENS],
            TOTAL_TOKENS: usage[TOTAL_TOKENS],
        }

    @staticmethod
    def get_runnable_agent(
        prompt: ChatPromptTemplate,
        llm: ChatOllama,
        schema: any,
        schema_basemodel: Optional[BaseModel] = None,
    ):
        if schema_basemodel is not None:
            parser = PydanticOutputParser(pydantic_object=schema_basemodel)
        else:
            parser = PydanticOutputParser(pydantic_object=schema)
        
        def runnable_agent(input: dict) -> tuple[Any, dict | None]:
            msg = prompt.format_messages(**input)
            raw = llm.invoke(msg)
            token_usage = CommonAgentOllama.normalize_token_usage(raw.usage_metadata)
            try:
                res = parser.parse(raw.content)
                return res, token_usage
            except Exception as e:
                logger.error(e)
                return None, token_usage
            return parser.parse(raw), token_usage
        return RunnableLambda(runnable_agent)
        
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        schema_basemodel: Optional[BaseModel] = None,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ) -> tuple[Any, Any, dict | None, Any | None]:
        system_prompt = escape_braces_for_format(system_prompt)
        format_instructions = get_format_instructions(schema)
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        system_prompt = system_prompt + "\n\n" + format_instructions
        instruction_prompt = escape_braces_for_format(instruction_prompt)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        # Initialize the callback handler
        callback_handler = OpenAICallbackHandler()

        updated_prompt = self._process_retryexception_message(prompt)
        agent = CommonAgentOllama.get_runnable_agent(updated_prompt, self.llm, schema, schema_basemodel)
        # agent = updated_prompt | self.llm.with_structured_output(schema)

        try:
            # res = agent.invoke({"input": instruction_prompt})
            res, token_usage = agent.invoke(
                {"input": instruction_prompt},
            )
            self._incre_token_usage(token_usage)
        except Exception as e:
            logger.error(f"Error executing chain: {e}")
            raise e
        processed_res = res
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                if self.try_fix_error is not None and self.exceptions is not None and len(self.exceptions) == 4:
                    fixed_res = self.try_fix_error(res, **kwargs)
                    if fixed_res is not None:
                        return res, fixed_res, self.token_usage, None
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, None


class CommonAgentOllamaTwoSteps(CommonAgentTwoSteps):
    def __init__(self, llm: ChatOllama):
        super().__init__(llm)

    def _invoke_agent(
        self, 
        system_prompt: str, 
        instruction_prompt: str, 
        schema: any, 
        schema_basemodel: Optional[BaseModel] = None, 
        post_process: Optional[Callable] = None, 
        **kwargs: Optional[Any],
    ):
        format_instructions = get_format_instructions(schema)
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        system_prompt = system_prompt + "\n\n" + format_instructions
        callback_handler = OpenAICallbackHandler()
        cot_prompt = self._build_prompt_for_cot_step(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
        )
        
        cot_prompt = cot_prompt + "\n\n" + format_instructions
        cot_res = self.llm.invoke(
            cot_prompt.invoke(input={}).to_messages(),
            config={
                "callbacks": [callback_handler],
            },
        )
        reasoning_process = cot_res.content
        token_usage = cot_res.usage_metadata
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        cot_tokens = {
            "total_tokens": total_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
        self._incre_token_usage(cot_tokens)
        updated_prompt = self._build_prompt_for_final_step(
            system_prompt=system_prompt,
            cot_msg=reasoning_process,
        )
        updated_prompt = updated_prompt + "\n\n" + format_instructions
        agent = structured_output_llm(self.llm, schema, updated_prompt)
        try:
            res = agent.invoke(
                input={"input": "Now, let's provide the final answer."},
                config={
                    "callbacks": [callback_handler],
                },
            )
            self._incre_token_usage(callback_handler)
        except Exception as e:
            logger.error(str(e))
            raise e
        processed_res = res
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, reasoning_process


class CommonAgentOllamaTwoChainSteps(CommonAgentTwoChainSteps):
    def __init__(self, llm: ChatOllama):
        super().__init__(llm)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self, 
        system_prompt: str, 
        instruction_prompt: str, 
        schema: any, 
        schema_basemodel: Optional[BaseModel] = None,
        post_process: Optional[Callable] = None, 
        **kwargs: Optional[Any],
    ):
        processed_system_prompt = escape_braces_for_format(system_prompt)
        format_instructions = get_format_instructions(schema)
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        processed_system_prompt = processed_system_prompt + "\n\n" + format_instructions
        callback_handler = OpenAICallbackHandler()
        
        cot_prompt = self._build_prompt_for_cot_step(
            system_prompt=processed_system_prompt, 
            instruction_prompt=instruction_prompt
        )
        try:
            msgs = cot_prompt.invoke(input={}).to_messages()
            cot_res = self.llm.invoke(msgs, config={
                "callbacks": [callback_handler],
            })
            reasoning_process = cot_res.content
            token_usage = cot_res.usage_metadata
            input_tokens = token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("output_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            cot_tokens = {
                "total_tokens": total_tokens,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
            self._incre_token_usage(cot_tokens)
        except Exception as e:
            logger.error(str(e))
            raise e
        
        try:
            processed_reasoning_process = escape_braces_for_format(reasoning_process)
            final_msg = FINAL_STEP_SYSTEM_PROMPTS.format(
                llm_response=processed_reasoning_process,
            )
            msgs = [(
                "human",
                final_msg,
            )]
            final_prompt = ChatPromptTemplate.from_messages(msgs)
            if schema_basemodel is not None:
                agent = structured_output_llm(self.llm, schema_basemodel, final_prompt)
            else:
                agent = structured_output_llm(self.llm, schema, final_prompt)
            res = agent.invoke(
                input={"input": "Now, let's provide the final answer."},
                config={
                    "callbacks": [callback_handler],
                },
            )
            self._incre_token_usage(callback_handler)
        except Exception as e:
            logger.error(str(e))
            raise e
        processed_res = res
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, reasoning_process

