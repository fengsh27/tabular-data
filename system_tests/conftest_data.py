## ===============================================
## 29100749
## ===============================================

from TabFuncFlow.utils.table_utils import single_html_table_to_markdown


paper_title_29100749 = "Presence of benzophenones commonly used as UV filters and absorbers in paired maternal and fetal samples"
paper_abstract_29100749 = """
Background: Previous studies have demonstrated widespread exposure of humans to certain benzophenones commonly used as UV filters or UV absorbers; some of which have been demonstrated to have endocrine disrupting abilities.

Objectives: To examine whether benzophenones present in pregnant women pass through the placental barrier to amniotic fluid and further to the fetal blood circulation.

Methods: A prospective study of 200 pregnant women with simultaneously collected paired samples of amniotic fluid and maternal serum and urine. In addition, unique samples of human fetal blood (n=4) obtained during cordocentesis: and cord blood (n=23) obtained at delivery, both with paired maternal samples of serum and urine collected simultaneously, were used. All biological samples were analyzed by TurboFlow-liquid chromatography - tandem mass spectrometry for seven different benzophenones.

Results: Benzophenone-1 (BP-1), benzophenone-3 (BP-3), 4-methyl-benzophenone (4-MBP), and 4-hydroxy-benzophenone (4-HBP) were all detectable in amniotic fluid and cord blood samples and except 4-HBP also in fetal blood; albeit at a low frequency. BP-1 and BP-3 were measured at ~10-times lower concentrations in fetal and cord blood compared to maternal serum and 1000-times lower concentration compared to maternal urine levels. Therefore BP-1 and BP-3 were only detectable in the fetal circulation in cases of high maternal exposure indicating some protection by the placental barrier. 4-MBP seems to pass into fetal and cord blood more freely with a median 1:3 ratio between cord blood and maternal serum levels. Only for BP-3, which the women seemed to be most exposed to, did the measured concentrations in maternal urine and serum correlate to concentrations measured in amniotic fluid. Thus, for BP-3, but not for the other tested benzophenones, maternal urinary levels seem to be a valid proxy for fetal exposure.

Conclusions: Detectable levels of several of the investigated benzophenones in human amniotic fluid as well as in fetal and cord blood calls for further investigations of the toxicokinetic and potential endocrine disrupting properties of these compounds in order for better assessment of the risk to the developing fetus.

Keywords: 4-Hydroxy-benzophenone (4-HBP); 4-Methyl-benzophenone (4-MBP); Benzophenone-1 (BP-1); Benzophenone-3 (BP-3); Endocrine disruptors; Fetal exposure.
"""

data_source_table_29100749_table_2 = """
| Empty Cell | ID | Cordocentesis_0 | Cordocentesis_1 | Cordocentesis_2 | Cordocentesis_3 | Delivery_0 | Delivery_1 | Delivery_2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Empty Cell | ID | Urine | Amnion | Serum | Fetal serum | Urine | Serum | Cord blood |
| BP-1 | 1 | – | < LOD | < LOD | < LOD | – | – | – |
| BP-1 | 2 | – | < LOD | 2.28 | 0.36 | – | < LOD | < LOD |
| BP-1 | 3 | 6.47 | < LOD | < LOD | < LOD | – | – | – |
| BP-1 | 4 | 3.68 | – | < LOD | < LOD | 4.13 | < LOD | < LOD |
| BP-3 | 1 | – | < LOD | 0.34 | < LOD | – | – | – |
| BP-3 | 2 | – | 0.33 | 37 | 10.1 | – | 1 | < LOD |
| BP-3 | 3 | 106.3 | < LOD | 0.77 | < LOD | – | – | – |
| BP-3 | 4 | 17.9 | – | 0.55 | < LOD | 32 | 0.77 | < LOD |
| 4-MBP | 1 | – | < LOD | 0.59 | 0.31 | – | – | – |
| 4-MBP | 2 | – | < LOD | 1.62 | < LOD | – | 1.12 | < LOD |
| 4-MBP | 3 | < LOD | < LOD | 1.04 | 1.3 | – | – | – |
| 4-MBP | 4 | 0.61 | – | 4.98 | 1.19 | < LOD | 6.31 | < LOD |
"""

curated_data_29100749 = """
| Patient ID | Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Parameter type | Parameter unit | Parameter value | Time value | Time unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | BP-1 | BP-1 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 6.47 | N/A | N/A |
| 4 | BP-1 | BP-1 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 3.68 | N/A | N/A |
| 3 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 106.3 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 17.9 | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.61 | N/A | N/A |
| 1 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LODa | N/A | N/A |
| 2 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-1 | BP-1 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.33 | N/A | N/A |
| 3 | BP-3 | BP-3 | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Amnion | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 2.28 | N/A | N/A |
| 3 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.34 | N/A | N/A |
| 2 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 37 | N/A | N/A |
| 3 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.77 | N/A | N/A |
| 4 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.55 | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.59 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.62 | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.04 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 4.98 | N/A | N/A |
| 1 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.36 | N/A | N/A |
| 3 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-1 | BP-1 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 10.1 | N/A | N/A |
| 3 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 4 | BP-3 | BP-3 | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.31 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.3 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Fetal serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.19 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 4.13 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 32 | N/A | N/A |
| 4 | BP-3 | BP-3 | Urine | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-1 | BP-1 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 2 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1 | N/A | N/A |
| 4 | BP-3 | BP-3 | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 0.77 | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 1.12 | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Serum | Maternal | Trimester 3 | N/A | Concentration | ng/ml | 6.31 | N/A | N/A |
| 1 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | BP-1 | BP-1 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | BP-3 | BP-3 | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 1 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 2 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
| 3 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | – | N/A | N/A |
| 4 | 4-MBP | 4-MBP | Cord blood | Maternal | Trimester 3 | N/A | Concentration | ng/ml | < LOD | N/A | N/A |
"""

data_col_mapping_29100749_table_2 = {
    "('Empty Cell', 'Empty Cell')": 'Uncategorized', 
    "('ID', 'ID')": 'Patient ID', 
    "('Cordocentesis', 'Urine')": 'Parameter value', 
    "('Cordocentesis', 'Amnion')": 'Parameter value', 
    "('Cordocentesis', 'Serum')": 'Parameter value', 
    "('Cordocentesis', 'Fetal serum')": 'Parameter value', 
    "('Delivery', 'Urine')": 'Parameter value', 
    "('Delivery', 'Serum')": 'Parameter value', 
    "('Delivery', 'Cord blood')": 'Parameter value',
}

data_md_table_aligned_29100749_table_2 = """
| ('Empty Cell', 'Empty Cell') | ('ID', 'ID') | ('Cordocentesis', 'Urine') | ('Cordocentesis', 'Amnion') | ('Cordocentesis', 'Serum') | ('Cordocentesis', 'Fetal serum') | ('Delivery', 'Urine') | ('Delivery', 'Serum') | ('Delivery', 'Cord blood') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BP-1 | 1 | – | < LODa | < LOD | < LOD | – | – | – |
| BP-1 | 2 | – | < LOD | 2.28 | 0.36 | – | < LOD | < LOD |
| BP-1 | 3 | 6.47 | < LOD | < LOD | < LOD | – | – | – |
| BP-1 | 4 | 3.68 | – | < LOD | < LOD | 4.13 | < LOD | < LOD |
| BP-3 | 1 | – | < LOD | 0.34 | < LOD | – | – | – |
| BP-3 | 2 | – | 0.33 | 37 | 10.1 | – | 1 | < LOD |
| BP-3 | 3 | 106.3 | < LOD | 0.77 | < LOD | – | – | – |
| BP-3 | 4 | 17.9 | – | 0.55 | < LOD | 32 | 0.77 | < LOD |
| 4-MBP | 1 | – | < LOD | 0.59 | 0.31 | – | – | – |
| 4-MBP | 2 | – | < LOD | 1.62 | < LOD | – | 1.12 | < LOD |
| 4-MBP | 3 | < LOD | < LOD | 1.04 | 1.3 | – | – | – |
| 4-MBP | 4 | 0.61 | – | 4.98 | 1.19 | < LOD | 6.31 | < LOD |
"""

data_md_table_list_29100749_table_2 = ["""
| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Cordocentesis', 'Urine') | – |
| 2 | ('Cordocentesis', 'Urine') | – |
| 3 | ('Cordocentesis', 'Urine') | 6.47 |
| 4 | ('Cordocentesis', 'Urine') | 3.68 |
| 1 | ('Cordocentesis', 'Urine') | – |
| 2 | ('Cordocentesis', 'Urine') | – |
| 3 | ('Cordocentesis', 'Urine') | 106.3 |
| 4 | ('Cordocentesis', 'Urine') | 17.9 |
| 1 | ('Cordocentesis', 'Urine') | – |
| 2 | ('Cordocentesis', 'Urine') | – |
| 3 | ('Cordocentesis', 'Urine') | < LOD |
| 4 | ('Cordocentesis', 'Urine') | 0.61 |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Cordocentesis', 'Amnion') | < LODa |
| 2 | ('Cordocentesis', 'Amnion') | < LOD |
| 3 | ('Cordocentesis', 'Amnion') | < LOD |
| 4 | ('Cordocentesis', 'Amnion') | – |
| 1 | ('Cordocentesis', 'Amnion') | < LOD |
| 2 | ('Cordocentesis', 'Amnion') | 0.33 |
| 3 | ('Cordocentesis', 'Amnion') | < LOD |
| 4 | ('Cordocentesis', 'Amnion') | – |
| 1 | ('Cordocentesis', 'Amnion') | < LOD |
| 2 | ('Cordocentesis', 'Amnion') | < LOD |
| 3 | ('Cordocentesis', 'Amnion') | < LOD |
| 4 | ('Cordocentesis', 'Amnion') | – |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Cordocentesis', 'Serum') | < LOD |
| 2 | ('Cordocentesis', 'Serum') | 2.28 |
| 3 | ('Cordocentesis', 'Serum') | < LOD |
| 4 | ('Cordocentesis', 'Serum') | < LOD |
| 1 | ('Cordocentesis', 'Serum') | 0.34 |
| 2 | ('Cordocentesis', 'Serum') | 37 |
| 3 | ('Cordocentesis', 'Serum') | 0.77 |
| 4 | ('Cordocentesis', 'Serum') | 0.55 |
| 1 | ('Cordocentesis', 'Serum') | 0.59 |
| 2 | ('Cordocentesis', 'Serum') | 1.62 |
| 3 | ('Cordocentesis', 'Serum') | 1.04 |
| 4 | ('Cordocentesis', 'Serum') | 4.98 |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 2 | ('Cordocentesis', 'Fetal serum') | 0.36 |
| 3 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 4 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 1 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 2 | ('Cordocentesis', 'Fetal serum') | 10.1 |
| 3 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 4 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 1 | ('Cordocentesis', 'Fetal serum') | 0.31 |
| 2 | ('Cordocentesis', 'Fetal serum') | < LOD |
| 3 | ('Cordocentesis', 'Fetal serum') | 1.3 |
| 4 | ('Cordocentesis', 'Fetal serum') | 1.19 |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Delivery', 'Urine') | – |
| 2 | ('Delivery', 'Urine') | – |
| 3 | ('Delivery', 'Urine') | – |
| 4 | ('Delivery', 'Urine') | 4.13 |
| 1 | ('Delivery', 'Urine') | – |
| 2 | ('Delivery', 'Urine') | – |
| 3 | ('Delivery', 'Urine') | – |
| 4 | ('Delivery', 'Urine') | 32 |
| 1 | ('Delivery', 'Urine') | – |
| 2 | ('Delivery', 'Urine') | – |
| 3 | ('Delivery', 'Urine') | – |
| 4 | ('Delivery', 'Urine') | < LOD |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Delivery', 'Serum') | – |
| 2 | ('Delivery', 'Serum') | < LOD |
| 3 | ('Delivery', 'Serum') | – |
| 4 | ('Delivery', 'Serum') | < LOD |
| 1 | ('Delivery', 'Serum') | – |
| 2 | ('Delivery', 'Serum') | 1 |
| 3 | ('Delivery', 'Serum') | – |
| 4 | ('Delivery', 'Serum') | 0.77 |
| 1 | ('Delivery', 'Serum') | – |
| 2 | ('Delivery', 'Serum') | 1.12 |
| 3 | ('Delivery', 'Serum') | – |
| 4 | ('Delivery', 'Serum') | 6.31 |""", """| ('ID', 'ID') | Parameter type | Parameter value |
| --- | --- | --- |
| 1 | ('Delivery', 'Cord blood') | – |
| 2 | ('Delivery', 'Cord blood') | < LOD |
| 3 | ('Delivery', 'Cord blood') | – |
| 4 | ('Delivery', 'Cord blood') | < LOD |
| 1 | ('Delivery', 'Cord blood') | – |
| 2 | ('Delivery', 'Cord blood') | < LOD |
| 3 | ('Delivery', 'Cord blood') | – |
| 4 | ('Delivery', 'Cord blood') | < LOD |
| 1 | ('Delivery', 'Cord blood') | – |
| 2 | ('Delivery', 'Cord blood') | < LOD |
| 3 | ('Delivery', 'Cord blood') | – |
| 4 | ('Delivery', 'Cord blood') | < LOD |"""]

data_caption_29100749_table_2 = """Table 2. Concentrations (ng/ml) of BP-1, BP-3 and 4-MBP in maternal serum, maternal urine, amniotic fluid and fetal serum from four different pregnant women: samples collected simultaneously at respectively cordocentesis and delivery.
BP-1: benzophenone-1; BP-3: benzophenone-3; 4-MBP: 4-methyl-benzophenone."""

data_md_table_drug_29100749_table_2 = """
| Drug name | Analyte | Specimen |
| --- | --- | --- |
| BP-1 | BP-1 | Urine |
| BP-1 | BP-1 | Amnion |
| BP-1 | BP-1 | Serum |
| BP-1 | BP-1 | Fetal serum |
| BP-1 | BP-1 | Cord blood |
| BP-3 | BP-3 | Urine |
| BP-3 | BP-3 | Amnion |
| BP-3 | BP-3 | Serum |
| BP-3 | BP-3 | Fetal serum |
| BP-3 | BP-3 | Cord blood |
| 4-MBP | 4-MBP | Urine |
| 4-MBP | 4-MBP | Amnion |
| 4-MBP | 4-MBP | Serum |
| 4-MBP | 4-MBP | Fetal serum |
| 4-MBP | 4-MBP | Cord blood |
"""


## ===============================================
## 32635742
## ===============================================

html_table_32635742_table_0 = """
<figure class="table" id="tb1"><figcaption><span class="heading">Table 1</span>. Esomeprazole Concentrations in Serum and Breast Milk Samples After 10 mg of Oral Esomeprazole Administration</figcaption><div class="table-wrap" tabindex="0"><table><thead><tr><th data-xml-align="left" rowspan="2">Days postpartum</th><th colspan="2"><span>Maternal serum</span></th><th colspan="2"><span>Umbilical cord blood</span></th><th colspan="2"><span>Infant serum</span></th><th colspan="2"><span>Breast milk</span></th></tr><tr><th>Time after ESO dose (hours)</th><th>ESO concentration (ng/mL)</th><th>Time after ESO dose (hours)</th><th>ESO concentration (ng/mL)</th><th>Time after ESO dose (hours)</th><th>ESO concentration (ng/mL)</th><th>Time after ESO dose (hours)</th><th>ESO concentration (ng/mL)</th></tr></thead><tbody><tr data-xml-align="center" data-xml-valign="bottom"><td data-xml-align="left">−1</td><td>12.2</td><td>35.8</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td></tr><tr data-xml-valign="bottom"><td>0</td><td data-xml-align="center">21.5</td><td data-xml-align="center">0.6</td><td>12.5</td><td>14.8</td><td>23.2</td><td>0.0</td><td> </td><td> </td></tr><tr data-xml-valign="bottom"><td>1</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td data-xml-align="left" rowspan="2">2</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>8.2</td><td>3.0</td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>19.4</td><td>0.0</td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td data-xml-align="left" rowspan="2">3</td><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>0.7</td><td>10.5</td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td> </td><td> </td><td> </td><td> </td><td> </td><td> </td><td>8.0</td><td>0.0</td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td data-xml-align="left" rowspan="2">4</td><td rowspan="2">3.0</td><td rowspan="2">280.2</td><td> </td><td> </td><td> </td><td> </td><td>4.0</td><td>19.6</td></tr><tr data-xml-align="center" data-xml-valign="bottom"><td> </td><td> </td><td> </td><td> </td><td>10.5</td><td>0.0</td></tr></tbody></table></div><div class="notes"><div role="doc-footnote"><div id="tf1" role="paragraph">ESO, esomeprazole.</div></div></div></figure>
"""

data_caption_32635742_table_0 = """Table 1. Esomeprazole Concentrations in Serum and Breast Milk Samples After 10 mg of Oral Esomeprazole Administration
ESO, esomeprazole."""

data_md_table_32635742_table_0 = single_html_table_to_markdown(html_table_32635742_table_0)

paper_title_32635742 = "Esomeprazole During Pregnancy and Lactation: Esomeprazole Levels in Maternal Serum, Cord Blood, Breast Milk, and the Infant's Serum"

paper_abstract_32635742 = """
Background: Esomeprazole is the S-isomer of omeprazole and is used to treat stomach acid-related diseases. Most data regarding the safety of esomeprazole during pregnancy are derived from studies on omeprazole, and the data characterizing esomeprazole transfer across the placenta and excretion into breast milk are limited. In this report, we discuss the safety of esomeprazole with reference to drug concentrations in maternal and neonatal blood and breast milk. Materials and Methods: After the patient provided informed consent, esomeprazole concentrations in maternal serum, breast milk, cord blood, and infant's serum were measured after 10 mg of maternal oral esomeprazole administration. Case Report: A 34-year-old female diagnosed with rheumatoid arthritis received esomeprazole before and during pregnancy and lactation. The esomeprazole concentration in cord blood was 40% of the level in maternal serum. At 12 hours after delivery (23.2 hours after dose), omeprazole was not detected in the infant's serum. In breast milk, esomeprazole concentrations at 0.7, 4.0, and 8.2 hours after the last dose were 10.5, 19.6, and 3.0 ng/mL, respectively, and esomeprazole was not detected at 10 hours after maternal administration. The calculated daily infant dose of esomeprazole through breast milk was 0.003 mg/[kg·day]. The infant demonstrated normal developmental progress and no detectable drug-related adverse effects. Discussion and Conclusions: Exposure to esomeprazole through placenta and breast milk was not clinically relevant in the infant. Further studies are needed to evaluate any harmful effects after exposure to esomeprazole in utero or during breastfeeding after esomeprazole treatment.
"""
