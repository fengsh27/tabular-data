
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
from pydantic import BaseModel

from extractor.agents.pk_summary.pk_sum_workflow_utils import PKSumWorkflowState
from extractor.agents.agent_prompt_utils import INSTRUCTION_PROMPT
from extractor.agents.pk_summary.pk_sum_workflow_utils import (
    pk_sum_enter_step,
    pk_sum_leave_step,
)
from extractor.agents.pk_summary.pk_sum_common_agent import (
    PKSumCommonAgentResult,
    PKSumCommonAgent
)

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
            step_reasoning_process=res['reasoning_process'] if res is not None and 'reasoning_process' in res else None,
            token_usage=token_usage
        )

    def execute(self, state: PKSumWorkflowState):
        self.enter_step(state)
        res, processed_res, token_usage = self.execute_directly(state)
        self.leave_step(state, res, processed_res, token_usage)

        return state

    @abstractmethod
    def execute_directly(self,  state: PKSumWorkflowState,) -> Tuple[
        PKSumCommonAgentResult, 
        Any | None, 
        dict | None
    ]:
        """ execute directly """


class PKSumCommonAgentStep(PKSumCommonStep):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_system_prompt(self, state: PKSumWorkflowState):
        """get system prompt"""

    def get_instruction_prompt(self, state: PKSumWorkflowState):
        """ get instruction prompt """
        return INSTRUCTION_PROMPT
    
    @abstractmethod
    def get_schema(self) -> PKSumCommonAgentResult | dict:
        """ get result schema (pydantic BaseModel or json schema)"""

    @abstractmethod
    def get_post_processor_and_kwargs(self, state: PKSumWorkflowState) -> Tuple[
        Callable | None,
        dict | None,
    ]:
        """ get post_processor and its kwargs """

    def execute_directly(self, state: PKSumWorkflowState) -> Tuple[
        PKSumCommonAgentResult,
        Any | None,
        dict | None,
    ]:
        """ execute step directly """
        system_prompt = self.get_system_prompt(state)
        instruction_prompt = self.get_instruction_prompt(state)
        llm = state["llm"]
        schema = self.get_schema()
        post_process, kwargs = self.get_post_processor_and_kwargs(state)
        agent = PKSumCommonAgent(llm=llm)
        if kwargs is not None:
            return agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
                **kwargs,
            )
        else:
            return agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
            )
        


