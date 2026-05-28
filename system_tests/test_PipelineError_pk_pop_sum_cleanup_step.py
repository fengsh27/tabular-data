import pytest

from TabFuncFlow.utils.table_utils import markdown_to_dataframe

md_table_assembled_18701886 = """
| Patient ID | Patient characteristic | Characteristic sub-category | Unit | Main value | Population | Pregnancy stage | Pediatric/Gestational age | Source text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| B | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| C | Gender | Male | N/A | M | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| D | Gender | Male | N/A | M | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| D.1 | Gender | Male | N/A | M | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| E | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| F | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| G | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| H | Gender | Male | N/A | M | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
| I | Gender | Female | N/A | F | Maternal | Delivery | N/A | Table 1 Maternal buprenorphine dosing and neonatal outcome measures |
"""

def test_pk_popu_sum_row_cleanup_step(llm, step_callback):
    from extractor.agents.pk_population_summary.pk_popu_sum_row_cleanup_step import (
        RowCleanupStep,
    )
    from extractor.agents.pk_population_summary.pk_popu_sum_workflow_utils import (
        PKPopuSumWorkflowState,
    )

    step = RowCleanupStep()
    state = PKPopuSumWorkflowState()
    state["llm"] = llm
    state["step_callback"] = step_callback
    state["df_combined"] = markdown_to_dataframe(md_table_assembled_18701886)

    state = step.execute(state)
    assert state is not None
    assert "df_combined" in state

