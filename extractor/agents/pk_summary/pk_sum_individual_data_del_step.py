import logging

from extractor.agents.agent_utils import DEFAULT_TOKEN_USAGE, display_md_table
from extractor.agents.pk_summary.pk_sum_common_step import PKSumCommonAgentStep
from extractor.agents.pk_summary.pk_sum_individual_data_del_agent import (
    INDIVIDUAL_DATA_DEL_PROMPT,
    IndividualDataDelResult,
    post_process_individual_del_result,
)

logger = logging.getLogger(__name__)


class IndividualDataDelStep(PKSumCommonAgentStep):
    """The step to delete individual data"""

    def __init__(self):
        super().__init__()
        self.start_title = "Deleting Individual Data"
        self.end_title = "Completed to Deleting Individual Data"

    def execute_directly(self, state):
        # This step only trims individual-level rows out of a summary table, so when the
        # model cannot answer it (retries exhausted on an unparsable reply - qwen3.6 lost 6
        # of 63 pk-summary runs this way, the whole paper each time) the right failure is to
        # keep the table as it is, not to drop it. try_fix_error cannot do this: it only
        # runs for a RetryException from post_process, not for a parse failure. The tokens
        # spent on the failed attempts are not accounted for (the agent is local to
        # the base implementation); the same was true when the paper was lost.
        try:
            return super().execute_directly(state)
        except Exception as e:  # noqa: BLE001 - tenacity.RetryError, parse or post_process errors
            logger.warning(
                "Deleting Individual Data failed (%r); keeping the table unchanged.", e, exc_info=True
            )
            return (
                IndividualDataDelResult(processed=False, row_list=None, col_list=None),
                state["md_table"],
                {**DEFAULT_TOKEN_USAGE},
            )

    def get_system_prompt(self, state):
        md_table = state["md_table"]
        system_prompt = INDIVIDUAL_DATA_DEL_PROMPT.format(
            processed_md_table=display_md_table(md_table)
        )
        previous_errors_prompt = self._get_previous_errors_prompt(state)
        return system_prompt + previous_errors_prompt

    def get_schema(self):
        return IndividualDataDelResult

    def get_post_processor_and_kwargs(self, state):
        md_table = state["md_table"]
        return post_process_individual_del_result, {"md_table": md_table}

    def leave_step(self, state, res, processed_res=None, token_usage=None):
        if processed_res is not None:
            state["md_table_summary"] = processed_res
            self._step_output(state, step_output="Result (md_table_summary):")
            self._step_output(state, step_output=processed_res)
        return super().leave_step(state, res, processed_res, token_usage)
