from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import logging

from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState, get_common_agent
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_summary.pk_sum_workflow_utils import (
    pk_sum_enter_step,
    pk_sum_leave_step,
)
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
    PKSumCommonAgent,
)
from extractor.prompts_utils import generate_previous_errors_prompt

logger = logging.getLogger(__name__)


class PKSumCommonStep(ABC):
    def __init__(self):
        super().__init__()
        self.start_title = ""
        self.start_descp = ""
        self.end_title = ""

    def enter_step(self, state: PKSumWorkflowState):
        pk_sum_enter_step(state, self.start_title, self.start_descp)

    def leave_step(
        self,
        state: PKSumWorkflowState,
        res: PKSumCommonAgentResult | None = None,
        processed_res: Any | None = None,
        token_usage: dict | None = None,
    ):
        pk_sum_leave_step(
            state=state,
            step_output=self.end_title,
            step_reasoning_process=res["reasoning_process"]
            if res is not None and "reasoning_process" in res
            else None,
            token_usage=token_usage,
        )

    def _step_output(
        self,
        state: PKSumWorkflowState,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
    ):
        def default_output(
            step_output: Optional[str] = None,
            step_reasoning_process: Optional[str] = None,
        ):
            if step_reasoning_process is not None:
                logger.info(f"\n\nReasoning: \n{step_reasoning_process}\n\n")
            if step_output is not None:
                logger.info(step_output)

        step_callback = (
            state["step_callback"]
            if "step_callback" in state and state["step_callback"] is not None
            else default_output
        )
        step_callback(
            step_reasoning_process=step_reasoning_process,
            step_output=step_output,
        )

    def _get_previous_errors_prompt(self, state: PKSumWorkflowState) -> str:
        previous_errors = state["previous_errors"] if "previous_errors" in state else "N/A"
        return generate_previous_errors_prompt(previous_errors)

    def execute(self, state: PKSumWorkflowState):
        self.enter_step(state)
        res, processed_res, token_usage = self.execute_directly(state)
        self.leave_step(state, res, processed_res, token_usage)

        return state

    @abstractmethod
    def execute_directly(
        self,
        state: PKSumWorkflowState,
    ) -> tuple[PKSumCommonAgentResult, Any | None, dict | None]:
        """execute directly"""


class PKSumCommonAgentStep(PKSumCommonStep):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_system_prompt(self, state: PKSumWorkflowState):
        """get system prompt"""

    def get_instruction_prompt(self, state: PKSumWorkflowState):
        """get instruction prompt"""
        return INSTRUCTION_PROMPT

    @abstractmethod
    def get_schema(self) -> PKSumCommonAgentResult | dict:
        """get result schema (pydantic BaseModel or json schema)"""

    @abstractmethod
    def get_post_processor_and_kwargs(
        self, state: PKSumWorkflowState
    ) -> tuple[
        Callable | None,
        dict | None,
    ]:
        """get post_processor and its kwargs"""

    def execute_directly(
        self, state: PKSumWorkflowState
    ) -> tuple[
        PKSumCommonAgentResult,
        Any | None,
        dict | None,
    ]:
        """execute step directly"""
        system_prompt = self.get_system_prompt(state)
        instruction_prompt = self.get_instruction_prompt(state)
        llm = state["llm"]
        schema = self.get_schema()
        post_process, kwargs = self.get_post_processor_and_kwargs(state)
        agent = get_common_agent(llm=llm) # PKSumCommonAgent(llm=llm)
        if kwargs is not None:
            res, processed_res, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
                **kwargs,
            )
        else:
            res, processed_res, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
            )
        reasoning_process = reasoning_process if reasoning_process is not None else None
        reasoning_process = res.reasoning_process if reasoning_process is None and hasattr(res, "reasoning_process") else reasoning_process
        reasoning_process = reasoning_process if reasoning_process is not None else ""
        self._step_output(state, step_reasoning_process=reasoning_process)
        return res, processed_res, token_usage
