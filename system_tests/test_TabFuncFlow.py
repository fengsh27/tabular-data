

from dotenv import load_dotenv
import pytest
from TabFuncFlow.steps_pk_summary.s_pk_extract_drug_info import s_pk_extract_drug_info
from TabFuncFlow.utils.table_utils import single_html_table_to_markdown


load_dotenv()

html_content = """
<section id="S8">
   <h3 class="pmc_sec_title">Pharmacokinetics</h3>
   <p id="P25"><a class="usa-link" href="#T2">Table II</a> summarizes the noncompartmental PK evaluation for the Elective Cohort. Eight pharmacokinetic samples (0.3%) were excluded, seven due to suspected contamination from the infused drug and one due to collection during the lorazepam infusion. Overall, the mean area-under-the-curve (AUC<sub>0–∞</sub>) was 822.5 ng*hr/mL and the median AUC<sub>0–∞</sub> was 601.5 ng*hr/mL with an average dose of 0.04 mg/kg. The overall fit of the population PK model was good over the wide range of individuals in the population. There were no covariates meeting criteria for inclusion into the model. Thirty three subjects (23 Status Cohort, 10 Elective Cohort) had received, at baseline, at least one agent that can induce drug metabolizing enzymes. The calculated value for terminal (beta) half-life was 16 hours for a typical 24 kg child. The empiric Bayesian estimated parameters from the post-hoc analysis are summarized in <a class="usa-link" href="#T3">Table III</a>. The model demonstrated good model prediction of observed concentrations even when applied to the patients exhibiting the highest and lowest individual clearances, and in a patient who received a total of 9 doses of lorazepam during the PK sampling interval.</p>
   <section class="tw xbox font-sm" id="T2">
      <h4 class="obj_head">Table 2.</h4>
      <div class="caption p">
         <p id="P40">Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0−∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life.</p>
      </div>
      <div class="tbl-box p" tabindex="0">
         <table class="content" frame="hsides" rules="groups">
            <thead>
               <tr>
                  <th align="left" colspan="1" rowspan="1"></th>
                  <th align="center" colspan="1" rowspan="1">C<sub>max</sub><br/>(ng/mL)</th>
                  <th align="center" colspan="1" rowspan="1" valign="top">AUC<sub>0−∞</sub> </th>
                  <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/kg)</th>
                  <th align="center" colspan="1" rowspan="1">CL<br/>(mL/min/m<sup>2</sup>)</th>
                  <th align="center" colspan="1" rowspan="1">Vdz<br/>(L/kg)</th>
                  <th align="center" colspan="1" rowspan="1">T<sub>1/2</sub><br/>(hr)</th>
               </tr>
            </thead>
            <tbody>
               <tr>
                  <td align="left" colspan="1" rowspan="1">N</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
                  <td align="center" colspan="1" rowspan="1">15</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1" valign="top">Range</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">29.3–209.6</td>
                  <td align="center" colspan="1" rowspan="1">253.3–3202.5</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">3.33–131.50</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">5.5–67.5</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">0.33–4.05</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">9.5–47.0</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1">Mean ± s.d.</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">56.1 ± 44.9</td>
                  <td align="center" colspan="1" rowspan="1">822.5 ± 706.1</td>
                  <td align="center" colspan="1" rowspan="1">49.33 ± 30.83</td>
                  <td align="center" colspan="1" rowspan="1" valign="top">31.95 ± 13.99</td>
                  <td align="center" colspan="1" rowspan="1">1.92 ± 0.84</td>
                  <td align="center" colspan="1" rowspan="1">20.5 ± 10.2</td>
               </tr>
               <tr>
                  <td align="left" colspan="1" rowspan="1">Median</td>
                  <td align="center" colspan="1" rowspan="1">42.2</td>
                  <td align="center" colspan="1" rowspan="1">601.5</td>
                  <td align="center" colspan="1" rowspan="1">41.50</td>
                  <td align="center" colspan="1" rowspan="1">32.34</td>
                  <td align="center" colspan="1" rowspan="1">1.94</td>
                  <td align="center" colspan="1" rowspan="1">18.1</td>
               </tr>
            </tbody>
         </table>
      </div>
      <div class="p text-right font-secondary"><a class="usa-link" href="table/T2/" rel="noopener noreferrer" target="_blank">Open in a new tab</a></div>
   </section>
</section>
"""

caption_and_footnote = "Non-compartmental pharmacokinetics parameters from Elective Cohort patients. C<sub>max</sub> is maximum concentration. AUC<sub>0−∞</sub> is area-under-the-curve to infinity. CL is clearance. Vdz is apparent volume of distribution. T<sub>1/2</sub> is half-life."

@pytest.mark.skip()
def test_extract_drug_info():
    markdown_table = single_html_table_to_markdown(html_content)
    print(markdown_table)
    drug_info = s_pk_extract_drug_info(markdown_table, caption_and_footnote, "chatgpt_4o")
    print(drug_info)



