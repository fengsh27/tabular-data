
import pytest
from langgraph.graph import StateGraph, START, END

from TabFuncFlow.utils.table_utils import dataframe_to_markdown, markdown_to_dataframe
from extractor.agents.agent_utils import extract_pmid_info_to_db
from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_correction_step import PKPECuratedTablesCorrectionStep
from extractor.agents.pk_pe_agents.pk_pe_verification_step import PKPECuratedTablesVerificationStep
from extractor.constants import MAX_STEP_COUNT

curated_table = """
col: | "Drug name" | "Analyte" | "Specimen" | "Population" | "Pregnancy stage" | "Pediatric/Gestational age" | "Subject N" | "Parameter type" | "Parameter unit" | "Statistics type" | "Main value" | "Variation type" | "Variation value" | "Interval type" | "Lower bound" | "Upper bound" | "P value" | "Time value" | "Time unit" |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
row 0: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Maximum plasma concentration | ng/ml | Mean | 11.96 | N/A | N/A | Range | 9.42 | 16.49 | N/A | N/A | N/A |
row 1: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Time to maximum plasma concentration | h | Mean | 3.50 | N/A | N/A | Range | 2.57 | 3.63 | N/A | N/A | N/A |
row 2: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Absorption half-life | h | Mean | 3.16 | N/A | N/A | Range | 2.62 | 3.68 | N/A | N/A | N/A |
row 3: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Absorption rate constant | h−1 | Mean | 0.23 | N/A | N/A | Range | 0.19 | 0.28 | N/A | N/A | N/A |
row 4: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Elimination half-life | h | Mean | 10.35 | N/A | N/A | Range | 9.39 | 11.32 | N/A | N/A | N/A |
row 5: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Elimination rate constant | h−1 | Mean | 0.068 | N/A | N/A | Range | 0.061 | 0.075 | N/A | N/A | N/A |
row 6: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Area under the curve (0–∞) | (ng h)/ml | Mean | 175.25 | N/A | N/A | Range | 145.74 | 204.75 | N/A | N/A | N/A |
row 7: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Total clearance (normalized by bioavailability) | ml/(min kg) | Mean | 2.61 | N/A | N/A | Range | 2.34 | 2.88 | N/A | N/A | N/A |
row 8: | Lorazepam | Lorazepam | Plasma | Maternal | Trimester 3 | N/A | N/A | Volume of distribution (normalized by bioavailability) | l | Mean | 178.78 | N/A | N/A | Range | 146.46 | 211.10 | N/A | N/A | N/A |
row 9: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Maximum plasma concentration | ng/ml | Mean | 35.55 | N/A | N/A | Range | 8.27 | 62.83 | N/A | N/A | N/A |
row 10: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Time to maximum plasma concentration | h | Mean | 4.33 | N/A | N/A | Range | 2.90 | 5.77 | N/A | N/A | N/A |
row 11: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Absorption half-life | h | Mean | 1.37 | N/A | N/A | Range | 1.15 | 1.58 | N/A | N/A | N/A |
row 12: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Absorption rate constant | h−1 | Mean | 0.52 | N/A | N/A | Range | 0.44 | 0.59 | N/A | N/A | N/A |
row 13: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Elimination half-life | h | Mean | 18.17 | N/A | N/A | Range | 14.10 | 22.23 | N/A | N/A | N/A |
row 14: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Elimination rate constant | h−1 | Mean | 0.039 | N/A | N/A | Range | 0.032 | 0.047 | N/A | N/A | N/A |
row 15: | Lorazepam | Lorazepam-glucuronide | Plasma | Maternal | Trimester 3 | N/A | N/A | Area under the curve (0–∞) | (ng h)/ml | Mean | 481.19 | N/A | N/A | Range | 252.87 | 709.51 | N/A | N/A | N/A |
row 16: | Lorazepam | Lorazepam | Urine | Maternal | N/A | N/A | N/A | Total amount excreted in urine | μg | Mean | 8.18 | N/A | N/A | Range | 2.67 | 13.70 | N/A | N/A | N/A |
row 17: | Lorazepam | Lorazepam | Urine | Maternal | N/A | N/A | N/A | Fraction excreted in urine | % | Mean | 0.29 | N/A | N/A | Range | 0.12 | 0.45 | N/A | N/A | N/A |
row 18: | Lorazepam | Lorazepam | Urine | Maternal | N/A | N/A | N/A | Renal clearance | ml/(min kg) | Mean | 0.0099 | N/A | N/A | Range | 0.0049 | 0.015 | N/A | N/A | N/A |
row 19: | Lorazepam | Lorazepam | Urine | Maternal | N/A | N/A | N/A | Elimination half-life | h | Mean | 12.75 | N/A | N/A | Range | 10.71 | 14.79 | N/A | N/A | N/A |
row 20: | Lorazepam | Lorazepam | Urine | Maternal | N/A | N/A | N/A | Elimination rate constant | h−1 | Mean | 0.057 | N/A | N/A | Range | 0.048 | 0.065 | N/A | N/A | N/A |
row 21: | Lorazepam | Lorazepam-glucuronide | Urine | Maternal | N/A | N/A | N/A | Total amount excreted in urine | μg | Mean | 899.77 | N/A | N/A | Range | 534.58 | 1265.0 | N/A | N/A | N/A |
row 22: | Lorazepam | Lorazepam-glucuronide | Urine | Maternal | N/A | N/A | N/A | Fraction excreted in urine | % | Mean | 44.97 | N/A | N/A | Range | 26.65 | 63.29 | N/A | N/A | N/A |
row 23: | Lorazepam | Lorazepam-glucuronide | Urine | Maternal | N/A | N/A | N/A | Renal clearance | ml/(min kg) | Mean | 1.12 | N/A | N/A | Range | 0.69 | 1.55 | N/A | N/A | N/A |
row 24: | Lorazepam | Lorazepam-glucuronide | Urine | Maternal | N/A | N/A | N/A | Elimination half-life | h | Mean | 11.5 | N/A | N/A | Range | 6.14 | 16.86 | N/A | N/A | N/A |
row 25: | Lorazepam | Lorazepam-glucuronide | Urine | Maternal | N/A | N/A | N/A | Elimination rate constant | h−1 | Mean | 0.066 | N/A | N/A | Range | 0.040 | 0.093 | N/A | N/A | N/A |
row 26: | Lorazepam | Lorazepam | Cord blood | Maternal | Delivery | N/A | 8 | Cord blood concentration | ng/ml | Mean | 6.78 | N/A | N/A | Range | 5.39 | 8.17 | N/A | 163.2–423 | Min |
row 27: | Lorazepam | Lorazepam | Maternal blood | Maternal | Delivery | N/A | 8 | Maternal blood concentration | ng/ml | Mean | 9.91 | N/A | N/A | Range | 7.68 | 12.14 | N/A | 163.2–423 | Min |
row 28: | Lorazepam | Lorazepam | Cord blood | Maternal | Delivery | N/A | 8 | Collection time | min | Mean | 293.4 | N/A | N/A | Range | 163.2 | 423 | N/A | 163.2–423 | Min |
row 29: | Lorazepam | Lorazepam | Cord blood | Maternal | Delivery | N/A | 8 | Cord blood to maternal blood ratio | unitless | Mean | 0.73 | N/A | N/A | Range | 0.52 | 0.94 | N/A | 163.2–423 | Min |
"""

def test_verification_step_correction_step(llm, step_callback, pmid_db):
    pmid = "16143486"
    pmid, title, abstract, full_text, tables, sections = extract_pmid_info_to_db(pmid, pmid_db)
    assert pmid is not None
    print_step = step_callback
    def check_verification_step(state: PKPECurationWorkflowState):
        if state["final_answer"] is not None and state["final_answer"]:
            print_step(step_name="Final Answer")
            print_step(step_output=state["final_answer"])
            return END
        if "step_count" in state and state["step_count"] >= MAX_STEP_COUNT:
            print_step(step_name="Max Step Count Reached")
            return END
        if not "curated_table" in state or (state["curated_table"] is None or len(state["curated_table"]) == 0):
            print_step(step_name="No Curated Table")
            return END
        return "correction_step"
    
    # prepare tables
    source_tables = []
    for ix in [1, 2, 3]:
        table = tables[ix]
        caption = "\n".join([table["caption"], table["footnote"]])
        source_table = dataframe_to_markdown(table["table"])
        source_tables.append(f"caption: \n{caption}\n\n table: \n{source_table}")

    verification_step = PKPECuratedTablesVerificationStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetic summary",
    )
    correction_step = PKPECuratedTablesCorrectionStep(
        llm=llm,
        pmid=pmid,
        domain="pharmacokinetic summary",
    )
    graph = StateGraph(PKPECurationWorkflowState)
    graph.add_node("verification_step", verification_step.execute)
    graph.add_node("correction_step", correction_step.execute)
    graph.add_edge(START, "verification_step")
    graph.add_conditional_edges(
        "verification_step",
        check_verification_step,
        {"correction_step", END},
    )
    graph.add_edge("correction_step", "verification_step")
    compiled_graph = graph.compile()
    res = compiled_graph.invoke({
        "pmid": pmid,
        "paper_title": title,
        "paper_abstract": abstract,
        "full_text": full_text,
        "source_tables": source_tables,
        "step_output_callback": print_step,
        "step_count": 0,
        "curated_table": curated_table,
    })
    assert res is not None
    assert res["final_answer"] is not None
    assert res["final_answer"]
    assert res["curated_table"] is not None
    df = markdown_to_dataframe(res["curated_table"])
    assert df.shape[0] == 30
    assert len(res["curated_table"]) > 0



