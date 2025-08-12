
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypedDict
import logging
from langchain_openai.chat_models.base import BaseChatOpenAI

from ..agent_utils import DEFAULT_TOKEN_USAGE

logger = logging.getLogger(__name__)

class CommonState(TypedDict):
    """
    CommonState is a TypedDict that defines the structure of the state
    used in the CommonStep class.
    """
    llm: Optional[BaseChatOpenAI]
    step_output_callback: Optional[Callable]

    final_answer: Optional[Any]

class CommonStep(ABC):
    """
    CommonStep is a base class for defining common steps in a workflow.
    It provides methods to execute the step and handle exceptions.
    """

    def __init__(self, llm: BaseChatOpenAI):
        super().__init__()
        self.step_name = ""
        self.llm = llm

    def enter_step(self, state):
        if state["step_output_callback"] is None:
            return
        state["step_output_callback"](
            step_name=self.step_name, 
        )
        if "step_count" in state:
            state["step_count"] += 1

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        if state["step_output_callback"] is None:
            return
        if token_usage is not None:
            state["step_output_callback"](token_usage=token_usage)

    def execute(self, state):
        """
        Execute the step. This method should be overridden by subclasses.
        """
        self.enter_step(state)
        state, token_usage = self._execute_directly(state)
        self.leave_step(state, token_usage)
        return state

    def _print_step(
        self,
        state,
        step_name: str | None = None,
        step_output: str | None = None,
        token_usage: dict | object | None = None,
    ):
        step_callback = state["step_output_callback"]
        if step_callback is None:
            return
        # convert token_usage to dict
        if token_usage is not None and not isinstance(token_usage, dict):
            token_usage = vars(token_usage)
            # In case token_usage.total_tokens is 0
            token_usage = { **DEFAULT_TOKEN_USAGE, **token_usage }
        step_callback(
            step_name=step_name,
            step_output=step_output,
            token_usage=token_usage,
        )                

    @abstractmethod
    def _execute_directly(self, state) -> tuple[dict, dict[str, int]]:
        """
        Execute the step directly. This method should be overridden by subclasses.
        Args:
            state (CommonState): The state of the workflow.
        Returns:
            tuple[dict, dict[str, int]]: The updated state and token usage.
        """
        pass



