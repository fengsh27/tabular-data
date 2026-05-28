"""
Fixture data for the pk_summary failure observed on PMID 19016466.

The paper "Nobori Stent Sizes, Biolimus A9 Doses and Pharmacokinetics" was
analyzed in job_3 (manifest_3) and ended with Final Answer = NoTable. The
underlying failure happened during the Time Extraction step: after five
retries the LLM repeatedly produced output that triggered
`AttributeError: 'NoneType' object has no attribute 'strip'`.

These fixtures capture the verbatim state that the Time Extraction step
received when the failure occurred. They are taken directly from
`output/tmp_3/logs/19016466.log`:

  - data_caption_19016466             — the Table 3 caption (line 660)
  - data_md_table_aligned_19016466    — the post-Align Parameter Type table
                                        (lines 180-201)
  - data_df_combined_19016466         — the post-Row-Cleanup table that was
                                        fed into Time Extraction (lines
                                        509-611). 103 rows, ~all N/A.
"""

data_caption_19016466 = (
    "Table 3 Nobori Stent Sizes, Biolimus A9 Doses and Pharmacokinetics."
)


# md_table_aligned: produced by the Align-Parameter-Type step before Row
# Cleanup. Column headers are nested multi-level (Stent / Total Exposure /
# t_max / Cmax / ...). This was already a difficult table for Row Cleanup
# because section headers like "14 mm" / "28 mm" are mixed in with the
# data rows.
data_md_table_aligned_19016466 = """
| Parameter type | ('Stent', 'Unnamed: 1_level_1') | ('Total Exposure Biolimus A9', '(μg)') | ('t_max', '[h]') | ('Cmax', '[pg/mL]') | ('t_last', '[h]') | ('C_last', '[pg/mL]') | ('AUC_obs', '[pg/mL·h]') | ('C_28d', '[pg/mL]') | ('C_6M', '[pg/mL]') | ('C_9M', '[pg/mL]') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean |  | 33655 | 1784 | 174 | 9844 | 117 | 70000 | 41 | 05 | <LLOQ |
| SD |  | 11299 | 5543 | 102 | 12461 | 70 | 101564 | 61 | 24 |  |
| CV% |  |  | 3107 | 588 | 1266 | 603 | 1451 | 1491 | 4359 |  |
| median |  |  | 20 | 177 | 4200 | 114 | 28553 | <LLOQ | <LLOQ |  |
| minimum |  | 225 | 005 | <LLOQ | 3 | <LLOQ | <LLOQ | <LLOQ | <LLOQ |  |
| maximum |  | 451 | 2160 | 32.2 | 4320 | 303 | 330965 | 17.0 | 103 |  |
| 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm | 14 mm |
| mean |  | 2256 | 16 | 166 | 10470 | 120 | 56372 | 39 | 11 | <LLOQ |
| SD |  | 24 | 11 | 80 | 14904 | 56 | 70929 | 59 | 34 |  |
| CV% |  |  | 705 | 485 | 1423 | 464 | 1258 | 1507 | 3000 |  |
| median |  | 225 | 15 | 177 | 4200 | 116 | 28654 | <LLOQ | <LLOQ |  |
| minimum |  | 225 | 005 | <LLOQ | 48 | <LLOQ | <LLOQ | <LLOQ | <LLOQ |  |
| maximum |  | 230 | 3 | 258 | 4320 | 208 | 228815 | 13.0 | 103 |  |
| 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm | 28 mm |
| mean |  | 4466 | 3553 | 182 | 9218 | 114 | 82267 | 43 | <LLOQ | <LLOQ |
| SD |  | 57 | 7660 | 123 | 10476 | 84 | 125721 | 67 |  |  |
| CV% |  |  | 2156 | 676 | 1137 | 743 | 1528 | 1555 |  |  |
| median |  | 451 | 25 | 207 | 4200 | 111 | 17152 | <LLOQ |  |  |
| minimum |  | 440 | 005 | <LLOQ | 3 | <LLOQ | <LLOQ | <LLOQ |  |  |
| maximum |  | 451 | 2160 | 322 | 2160 | 303 | 330965 | 17.0 |  |  |
"""


# df_combined: the post-Row-Cleanup table that was fed into Time Extraction.
# 103 data rows, all "Biolimus A9 / Plasma" with empty Subject N / Population,
# and Statistics-type / Main-value cells misaligned (e.g. "14 mm" landed in
# the Statistics-type column). This is the actual input that triggered the
# Time Extraction failure.
data_df_combined_19016466 = """
| Drug name | Analyte | Specimen | Population | Pregnancy stage | Pediatric/Gestational age | Subject N | Parameter type | Parameter unit | Statistics type | Main value | Variation type | Variation value | Interval type | Lower bound | Upper bound | P value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | 14 mm | N/A | 14 mm | 14 mm | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | 28 mm | N/A | 28 mm | 28 mm | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 3107 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 705 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 2156 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 588 | median | 177 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 485 | median | 177 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 676 | median | 207 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1266 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1423 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1137 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 603 | median | 114 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 464 | median | 116 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 743 | median | 111 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1451 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1258 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1528 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1491 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1507 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 1555 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 4359 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | CV% | % | CV% | 3000 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 24 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 11299 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 57 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 5543 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 11 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 7660 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 102 | CV% | 588 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 80 | CV% | 485 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 123 | CV% | 676 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 12461 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 14904 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 10476 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 70 | CV% | 603 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 56 | CV% | 464 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 84 | CV% | 743 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 101564 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 70929 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 125721 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 61 | CV% | 1491 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 59 | CV% | 1507 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 67 | CV% | 1555 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | SD | μg | SD | 34 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 451 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 2160 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 4320 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 303 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 330965 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 17.0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 103 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 230 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 3 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 32.2 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 258 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 322 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 208 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 228815 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | maximum | μg | maximum | 13.0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 33655 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 2256 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 4466 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 1784 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 16 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 3553 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 174 | SD | 102 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 166 | SD | 80 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 182 | SD | 123 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 9844 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 10470 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 9218 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 117 | SD | 70 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 120 | SD | 56 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 114 | SD | 84 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 70000 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 56372 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 82267 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 41 | SD | 61 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 39 | SD | 59 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 43 | SD | 67 | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 05 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | mean | μg | mean | 11 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 177 | minimum | <LLOQ | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 4200 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 225 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 451 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 20 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 15 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 25 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 207 | minimum | <LLOQ | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 114 | minimum | <LLOQ | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 116 | minimum | <LLOQ | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 111 | minimum | <LLOQ | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 28553 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 28654 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | median | μg | median | 17152 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | minimum | μg | minimum | 225 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | minimum | μg | minimum | 005 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | minimum | μg | minimum | 3 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | minimum | μg | minimum | 440 | N/A | N/A | N/A | N/A | N/A | N/A |
| Biolimus A9 | Biolimus A9 | Plasma | N/A | N/A | N/A | N/A | minimum | μg | minimum | 48 | N/A | N/A | N/A | N/A | N/A | N/A |
"""
