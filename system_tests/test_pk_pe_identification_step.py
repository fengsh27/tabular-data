

from extractor.agents.pk_pe_agents.pk_pe_agents_types import PKPECurationWorkflowState
from extractor.agents.pk_pe_agents.pk_pe_identification_step import PKPEIdentificationStep


def test_pk_pe_identification_step_22050870(llm, step_callback):
    step = PKPEIdentificationStep(llm=llm)
    state = PKPECurationWorkflowState(
        step_output_callback=step_callback,
        paper_title="Pharmacokinetics of intravenous lorazepam in pediatric patients with and without status epilepticus",
        paper_abstract="""Objective: To evaluate the single dose pharmacokinetics of an intravenous dose of lorazepam in pediatric patients treated for status epilepticus (SE) or with a history of SE.
Study design: Ten hospitals in the Pediatric Emergency Care Applied Research Network enlisted patients 3 months to 17 years with convulsive SE (status cohort) or for a traditional pharmacokinetics study (elective cohort). Sparse sampling was used for the status cohort, and intensive sampling was used for the elective cohort. Non-compartmental analyses were performed on the elective cohort, and served to nest compartmental population pharmacokinetics analysis for both cohorts.
Results: A total of 48 patients in the status cohort and 15 patients in the elective cohort were enrolled. Median age was 7 years, 2 months. The population pharmacokinetics parameters were: clearance, 1.2 mL/min/kg; half-life, 16.8 hours; and volume of distribution, 1.5 L/kg. On the basis of the pharmacokinetics model, a 0.1 mg/kg dose is expected to achieve concentrations of approximately 100 ng/mL and maintain concentrations >30 to 50 ng/mL for 6 to 12 hours. A second dose of 0.05 mg/kg would achieve desired therapeutic serum levels for approximately 12 hours without excessive sedation. Age-dependent dosing is not necessary beyond using a maximum initial dose of 4 mg.
Conclusions: Lorazepam pharmacokinetics in convulsive SE is similar to earlier pharmacokinetics measured in pediatric patients with cancer, except for longer half-life, and similar to adult pharmacokinetics parameters except for increased clearance.
Copyright Â© 2012 Mosby, Inc. All rights reserved.""",
    )

    state = step.execute(state=state)
    assert state["paper_type"] is not None
