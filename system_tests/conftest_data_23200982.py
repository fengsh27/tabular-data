

data_md_table_aligned_23200982_table_3 = """
| ('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'Mother #') | ('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Last Dose (days)') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Mother') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Cord') | ('Day of Birth CZP concentrations (μg/ml)', 'Lower limit of Quantification <0.41', 'Infant') | ('Ratio Cord/Mother', 'Unnamed: 5_level_1', 'Unnamed: 5_level_2') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Mother') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Cord') | ('Day of Birth PEG concentrations (μg/ml)', 'Lower limit of Quantification <9', 'Infant') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 14 | 18.83 | 1.65 | - | 8.8% | 33.4 | * | * |
| 02 | 7 | 59.57 | 0.94 | 1.02 | 1.6% | 51.3 | * | * |
| 03 | 28 | 4.87 | 1.19 | 1.22 | 24% | * | * | * |
| 04 | 17 | 20.13 | 0.57 | 0.44 | 2.8% | 34.7 | * | * |
| 05 | 21 | 16.49 | <0.41 | <0.41 | 2.5% | 27.7 | * | No sample |
| 06 | 24 | 34.65 | 1.66 | 1.58 | 4.8% | 34.4 | * | * |
| 07 | 28 | 1.87 | <0.41 | <0.41 | 22% | * | * | * |
| 08-A | 42 | 6.32 | <0.41 | 0.58 | 6.4% | 11.1 | * | * |
| B |  |  | <0.41 | <0.41 | 6.4% |  | * | * |
| 09-A | 6 | 42.7 | 1.28 | 1.34 | 3.0% | 62.1 | * | * |
| B |  |  | 1.16 | 1.18 | 2.7% |  | * | * |
| 10 | 5 | 37.83 | 0.55 | 0.6 | 1.5% | 74.9 | * | * |
"""

data_md_table_aligned_23200982_table_2 = """
| ('Pt #', 'Pt #') | ('ADA dose', 'ADA dose') | ('ADA interval', 'ADA interval') | ('Time dose to birth (days)', 'Time dose to birth (days)') | ('ADA(μg/ml) at Birth', 'Mom:') | ('ADA(μg/ml) at Birth', 'Cord:') | ('ADA(μg/ml) at Birth', 'Infant') | ('Ratio Cord/Mother', 'Ratio Cord/Mother') | ('Follow ADA Levels (time)', 'Follow ADA Levels (time)') | ('Newborn Complications', 'Newborn Complications') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1#^ | 40 mg | EOW | 7 | 6.05 | 9.29 | 6.17 | 153% | -- | None |
| 2^ | 40 mg | EOW | 56 | 1.8399999999999999 | 5.39 | 6.01 | 293% | 1.94 (6 wks) | Pulmonary edema, brief at birth |
| 3#^ | 40 mg | EOW | 7 | 3.84 | 4.57 | -- | 119% | -- | None |
| 4# | 40 mg | EOW | 42 | 0.0 | 0.16 | -- | -- | -- | None |
| 5 | 40 mg | EOW | 35 | 2.2 | 4.18 | 4.28 | 190% | .934 (8 wks) | None |
| 6 | 40 mg | EOW | 42 | 3.21 | 4.74 | 4.87 | 148% | 1.31 (7 wks) | None |
| 7^ | 40 mg | EOW | 42 | 3.36 | 8.94 | 8.09 | 266% | 0.529 (11 wks) | None |
| 8 | 40 mg | Weekly | 1 | 16.1 | 19.7 | 17.7 | 122% | -- | None |
| 9 | 40 mg | EOW | 49 | 2.24 | 4.95 | 4.64 | 220% | -- | None |
| 10 | 40 mg | EOW | 7 | 8.48 | 8.29 | 9.17 | 98% | -- | None |
"""

data_col_mapping_23200982_table_2 = {
    "('Pt #', 'Pt #')": 'Patient ID', 
    "('ADA dose', 'ADA dose')": 'Uncategorized', 
    "('ADA interval', 'ADA interval')": 'Uncategorized', 
    "('Time dose to birth (days)', 'Time dose to birth (days)')": 'Uncategorized', 
    "('ADA(μg/ml) at Birth', 'Mom:')": 'Parameter value', 
    "('ADA(μg/ml) at Birth', 'Cord:')": 'Parameter value', 
    "('ADA(μg/ml) at Birth', 'Infant')": 'Parameter value', 
    "('Ratio Cord/Mother', 'Ratio Cord/Mother')": 'Parameter value', 
    "('Follow ADA Levels (time)', 'Follow ADA Levels (time)')": 'Uncategorized', 
    "('Newborn Complications', 'Newborn Complications')": 'Uncategorized'
}


