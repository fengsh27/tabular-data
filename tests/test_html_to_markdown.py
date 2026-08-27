
import pytest

from TabFuncFlow.utils.table_utils import single_html_table_to_markdown, markdown_to_dataframe


html_content_29100749_table_2 = """
<div class="tables frame-topbot rowsep-0 colsep-0" id="t0015"><span class="captions text-s"><span id="ca0020"><p id="sp0030"><span class="label">Table 2</span>. Concentrations (ng/ml) of BP-1, BP-3 and 4-MBP in maternal serum, maternal urine, amniotic fluid and fetal serum from four different pregnant women: samples collected simultaneously at respectively cordocentesis and delivery.</p></span></span><div class="groups"><table><thead><tr class="valign-top"><td class="rowsep-0" rowspan="2" scope="col"><span class="screen-reader-only">Empty Cell</span></td><th class="rowsep-1" rowspan="2" scope="col">ID</th><th class="rowsep-1" colspan="4" scope="col">Cordocentesis</th><th class="rowsep-1" colspan="3" scope="col">Delivery</th></tr><tr class="rowsep-1 valign-top"><th scope="col">Urine</th><th scope="col">Amnion</th><th scope="col">Serum</th><th scope="col">Fetal serum</th><th scope="col">Urine</th><th scope="col">Serum</th><th scope="col">Cord blood</th></tr></thead><tbody><tr><th rowspan="4" scope="row">BP-1</th><td>1</td><td>–</td><td>&lt; LOD<a class="anchor anchor-primary" data-sd-ui-side-panel-opener="true" data-xocs-content-id="tf0025" data-xocs-content-type="reference" href="#tf0025" name="btf0025"><span class="anchor-text-container"><span class="anchor-text"><sup>a</sup></span></span></a></td><td>&lt; LOD</td><td>&lt; LOD</td><td>–</td><td>–</td><td>–</td></tr><tr><td>2</td><td>–</td><td>&lt; LOD</td><td>2.28</td><td>0.36</td><td>–</td><td>&lt; LOD</td><td>&lt; LOD</td></tr><tr><td>3</td><td>6.47</td><td>&lt; LOD</td><td>&lt; LOD</td><td>&lt; LOD</td><td>–</td><td>–</td><td>–</td></tr><tr><td>4</td><td>3.68</td><td>–</td><td>&lt; LOD</td><td>&lt; LOD</td><td>4.13</td><td>&lt; LOD</td><td>&lt; LOD</td></tr><tr><th rowspan="4" scope="row">BP-3</th><td>1</td><td>–</td><td>&lt; LOD</td><td>0.34</td><td>&lt; LOD</td><td>–</td><td>–</td><td>–</td></tr><tr><td>2</td><td>–</td><td>0.33</td><td>37</td><td>10.1</td><td>–</td><td>1</td><td>&lt; LOD</td></tr><tr><td>3</td><td>106.3</td><td>&lt; LOD</td><td>0.77</td><td>&lt; LOD</td><td>–</td><td>–</td><td>–</td></tr><tr><td>4</td><td>17.9</td><td>–</td><td>0.55</td><td>&lt; LOD</td><td>32</td><td>0.77</td><td>&lt; LOD</td></tr><tr><th rowspan="4" scope="row">4-MBP</th><td>1</td><td>–</td><td>&lt; LOD</td><td>0.59</td><td>0.31</td><td>–</td><td>–</td><td>–</td></tr><tr><td>2</td><td>–</td><td>&lt; LOD</td><td>1.62</td><td>&lt; LOD</td><td>–</td><td>1.12</td><td>&lt; LOD</td></tr><tr><td>3</td><td>&lt; LOD</td><td>&lt; LOD</td><td>1.04</td><td>1.3</td><td>–</td><td>–</td><td>–</td></tr><tr><td>4</td><td>0.61</td><td>–</td><td>4.98</td><td>1.19</td><td>&lt; LOD</td><td>6.31</td><td>&lt; LOD</td></tr></tbody></table></div><div class="legend"><div class="u-margin-s-bottom" id="sp0035">BP-1: benzophenone-1; BP-3: benzophenone-3; 4-MBP: 4-methyl-benzophenone.</div></div><dl class="footnotes"><dt id="tf0025">a</dt><dd><div class="u-margin-s-bottom" id="np0025">LOD: limit of detection.</div></dd></dl></div>
"""

html_content_32153014_table_2 = """
<section class="tw xbox font-sm" id="cpt1827-tbl-0002" lang="en"><h5 class="obj_head">Table 2.</h5>
<div class="caption p"><p>Placental transfer and placental exposure to infliximab and etanercept in patients with autoimmune diseases</p></div>
<div class="tbl-box p" tabindex="0"><table class="content" frame="hsides" rules="groups">
<colgroup><col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
<col style="border-right:solid 1px #000000" span="1">
</colgroup><thead valign="bottom">
<tr style="border-bottom:solid 1px #000000">
<th align="left" rowspan="3" valign="bottom" colspan="1">Patient</th>
<th align="center" rowspan="3" valign="bottom" colspan="1">TNF inhibitor</th>
<th align="center" rowspan="3" valign="bottom" colspan="1">Dosing regimen</th>
<th align="center" rowspan="3" valign="bottom" colspan="1">Time from last dose to delivery (days)</th>
<th align="center" colspan="3" style="border-bottom:solid 1px #000000" valign="bottom" rowspan="1">TNF inhibitor</th>
<th align="center" rowspan="3" valign="bottom" colspan="1">Cord‐to‐maternal ratio</th>
<th align="center" rowspan="3" valign="bottom" colspan="1">Placenta‐to‐maternal ratio</th>
</tr>
<tr style="border-bottom:solid 1px #000000">
<th align="center" colspan="2" style="border-bottom:solid 1px #000000" valign="bottom" rowspan="1">(µg/mL serum)</th>
<th align="center" style="border-bottom:solid 1px #000000" valign="bottom" rowspan="1" colspan="1">Mean&nbsp;±&nbsp;SD (µg/g tissue)</th>
</tr>
<tr style="border-bottom:solid 1px #000000">
<th align="center" valign="bottom" rowspan="1" colspan="1">Maternal</th>
<th align="center" valign="bottom" rowspan="1" colspan="1">Cord</th>
<th align="center" valign="bottom" rowspan="1" colspan="1">Placenta</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left" rowspan="2" colspan="1">1</td>
<td align="center" rowspan="2" colspan="1">Infliximab</td>
<td align="center" rowspan="1" colspan="1">400&nbsp;mg per 8&nbsp;weeks</td>
<td align="center" rowspan="2" colspan="1">23</td>
<td align="center" rowspan="2" colspan="1">25.3</td>
<td align="center" rowspan="2" colspan="1">29.8</td>
<td align="center" rowspan="2" colspan="1">5.8&nbsp;±&nbsp;0.9</td>
<td align="center" rowspan="2" colspan="1">1.18</td>
<td align="center" rowspan="2" colspan="1">0.35</td>
</tr>
<tr><td align="center" rowspan="1" colspan="1">(5&nbsp;mg/kg)</td></tr>
<tr>
<td align="left" rowspan="2" colspan="1">2</td>
<td align="center" rowspan="2" colspan="1">Infliximab</td>
<td align="center" rowspan="1" colspan="1">400&nbsp;mg per 8&nbsp;weeks</td>
<td align="center" rowspan="2" colspan="1">57</td>
<td align="center" rowspan="2" colspan="1">12.0</td>
<td align="center" rowspan="2" colspan="1">24.0</td>
<td align="center" rowspan="2" colspan="1">1.8&nbsp;±&nbsp;0.0</td>
<td align="center" rowspan="2" colspan="1">2.00</td>
<td align="center" rowspan="2" colspan="1">0.23</td>
</tr>
<tr><td align="center" rowspan="1" colspan="1">(5&nbsp;mg/kg)</td></tr>
<tr>
<td align="left" rowspan="2" colspan="1">3</td>
<td align="center" rowspan="2" colspan="1">Infliximab</td>
<td align="center" rowspan="1" colspan="1">400&nbsp;mg per 8&nbsp;weeks</td>
<td align="center" rowspan="2" colspan="1">31</td>
<td align="center" rowspan="2" colspan="1">17.0</td>
<td align="center" rowspan="2" colspan="1">29.0</td>
<td align="center" rowspan="2" colspan="1">4.8&nbsp;±&nbsp;1.5</td>
<td align="center" rowspan="2" colspan="1">1.71</td>
<td align="center" rowspan="2" colspan="1">0.44</td>
</tr>
<tr><td align="center" rowspan="1" colspan="1">(5&nbsp;mg/kg)</td></tr>
<tr>
<td align="left" rowspan="1" colspan="1">4</td>
<td align="center" rowspan="1" colspan="1">Etanercept</td>
<td align="center" rowspan="1" colspan="1">50&nbsp;mg per 12&nbsp;days</td>
<td align="center" rowspan="1" colspan="1">4</td>
<td align="center" rowspan="1" colspan="1">3.0</td>
<td align="center" rowspan="1" colspan="1">0.1</td>
<td align="center" rowspan="1" colspan="1">0.1&nbsp;±&nbsp;0.1</td>
<td align="center" rowspan="1" colspan="1">0.04</td>
<td align="center" rowspan="1" colspan="1">0.03</td>
</tr>
<tr style="border-bottom:solid 1px #000000">
<td align="left" rowspan="1" colspan="1">5</td>
<td align="center" rowspan="1" colspan="1">Etanercept</td>
<td align="center" rowspan="1" colspan="1">50&nbsp;mg per week</td>
<td align="center" rowspan="1" colspan="1">29</td>
<td align="center" rowspan="1" colspan="1">&lt;0.1</td>
<td align="center" rowspan="1" colspan="1">&lt;0.1</td>
<td align="center" rowspan="1" colspan="1">&lt;0.1</td>
<td align="center" rowspan="1" colspan="1">NA</td>
<td align="center" rowspan="1" colspan="1">NA</td>
</tr>
<tr>
<td align="left" rowspan="1" colspan="1">&nbsp;6</td>
<td align="center" rowspan="1" colspan="1">Etanercept</td>
<td align="center" rowspan="1" colspan="1">50&nbsp;mg per week</td>
<td align="center" rowspan="1" colspan="1">16</td>
<td align="center" rowspan="1" colspan="1">0.2</td>
<td align="center" rowspan="1" colspan="1">NA</td>
<td align="center" rowspan="1" colspan="1">&lt;0.1</td>
<td align="center" rowspan="1" colspan="1">NA</td>
<td align="center" rowspan="1" colspan="1">NA</td>
</tr>
</tbody>
</table></div>
<div class="p text-right font-secondary"><a href="table/cpt1827-tbl-0002/" class="usa-link" target="_blank" rel="noopener noreferrer">Open in a new tab</a></div>
<div class="tw-foot p">
<div class="fn" id="cpt1827-note-0003"><p>Placental transfer is represented as cord‐to‐maternal ratios based on serum levels, and placental exposure is calculated as placenta‐to‐maternal ratios based on placental tissue concentrations corrected for serum levels and maternal calculated whole blood concentrations. Cord blood of patient 6 was not available.</p></div>
<div class="fn" id="cpt1827-note-0002"><p>NA, not assessed; TNF, tumor necrosis factor.</p></div>
</div></section>
"""

def test_html_to_markdown():
    md_table = single_html_table_to_markdown(html_content_29100749_table_2)
    assert md_table is not None
    assert len(md_table) > 0

    md_table_1 = single_html_table_to_markdown(html_content_32153014_table_2)
    assert md_table_1 is not None
    assert len(md_table_1) > 0
    df = markdown_to_dataframe(md_table_1)
    assert df.shape[0] == 9
    assert df.shape[1] == 9


    

