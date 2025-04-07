from extractor.agents.pk_summary.pk_sum_assembly_step import (
    AssemblyStep,
)
from extractor.agents.pk_summary.pk_sum_workflow import PKSumWorkflowState


def test_AssemblyStep_30825333_table_2(
    llm,
    caption_30825333_table_2,
    drug_list_30825333_table_2,
    patient_list_30825333_table_2,
    type_unit_list_30825333_table_2,
    value_list_30825333_table_2,
    step_callback,
):
    step = AssemblyStep()
    state = PKSumWorkflowState()
    state["llm"] = llm
    state["caption"] = caption_30825333_table_2
    state["drug_list"] = drug_list_30825333_table_2
    state["patient_list"] = patient_list_30825333_table_2
    state["type_unit_list"] = type_unit_list_30825333_table_2
    state["value_list"] = value_list_30825333_table_2
    state["step_callback"] = step_callback

    step.execute(state)

    assert state["df_combined"] is not None
