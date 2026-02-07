from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import logging

from extractor.agents.pk_drug_individual.pk_drug_ind_workflow_utils import PKDrugIndWorkflowState
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_drug_individual.pk_drug_ind_workflow_utils import (
    pk_drug_ind_enter_step,
    pk_drug_ind_leave_step,
)
from extractor.agents.pk_drug_individual.pk_drug_ind_common_agent import (
    PKDrugIndCommonAgentResult,
    PKDrugIndCommonAgent,
)
from extractor.prompts_utils import generate_previous_errors_prompt

logger = logging.getLogger(__name__)


class PKDrugIndCommonStep(ABC):
    def __init__(self):
        super().__init__()
        self.start_title = ""
        self.start_descp = ""
        self.end_title = ""

    def enter_step(self, state: PKDrugIndWorkflowState):
        pk_drug_ind_enter_step(state, self.start_title, self.start_descp)

    def leave_step(
        self,
        state: PKDrugIndWorkflowState,
        res: PKDrugIndCommonAgentResult | None = None,
        processed_res: Any | None = None,
        token_usage: dict | None = None,
    ):
        pk_drug_ind_leave_step(
            state=state,
            step_output=self.end_title,
            step_reasoning_process=res["reasoning_process"]
            if res is not None and "reasoning_process" in res
            else None,
            token_usage=token_usage,
        )

    def _step_output(
        self,
        state: PKDrugIndWorkflowState,
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

    def _get_previous_errors_prompt(self, state: PKDrugIndWorkflowState) -> str:
        previous_errors = state["previous_errors"] if "previous_errors" in state else "N/A"
        return generate_previous_errors_prompt(previous_errors)

    def execute(self, state: PKDrugIndWorkflowState):
        self.enter_step(state)
        res, processed_res, token_usage = self.execute_directly(state)
        self.leave_step(state, res, processed_res, token_usage)

        return state

    @abstractmethod
    def execute_directly(
        self,
        state: PKDrugIndWorkflowState,
    ) -> tuple[PKDrugIndCommonAgentResult, Any | None, dict | None]:
        """execute directly"""


class PKDrugIndCommonAgentStep(PKDrugIndCommonStep):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_system_prompt(self, state: PKDrugIndWorkflowState):
        """get system prompt"""

    def get_instruction_prompt(self, state: PKDrugIndWorkflowState):
        """get instruction prompt"""
        return INSTRUCTION_PROMPT

    @abstractmethod
    def get_schema(self) -> PKDrugIndCommonAgentResult | dict:
        """get result schema (pydantic BaseModel or json schema)"""

    @abstractmethod
    def get_post_processor_and_kwargs(
        self, state: PKDrugIndWorkflowState
    ) -> tuple[
        Callable | None,
        dict | None,
    ]:
        """get post_processor and its kwargs"""

    def execute_directly(
        self, state: PKDrugIndWorkflowState
    ) -> tuple[
        PKDrugIndCommonAgentResult,
        Any | None,
        dict | None,
    ]:
        """execute step directly"""
        system_prompt = self.get_system_prompt(state)
        instruction_prompt = self.get_instruction_prompt(state)
        llm = state["llm"]
        schema = self.get_schema()
        post_process, kwargs = self.get_post_processor_and_kwargs(state)
        agent = PKDrugIndCommonAgent(llm=llm)
        if kwargs is not None:
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
                **kwargs,
            )
        else:
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
            )
        reasoning_process = ""
        if res is not None:
            try:
                reasoning_process = (
                    res["reasoning_process"]
                    if type(res) == dict
                    else res.reasoning_process
                )
            except Exception as e:
                logger.error(
                    f"Failed to access res.reasoning_process.\nError is {str(e)}"
                )
                pass
        self._step_output(state, step_reasoning_process=reasoning_process)
        return res, processed_res, token_usage
