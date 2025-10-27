import pandas as pd

PK_INDIVIDUAL_COLUMN_NAMES = [
    "Patient ID",
    "Drug name",
    "Analyte",
    "Specimen",
    "Population",
    "Pregnancy stage",
    "Pediatric/Gestational age",
    "Parameter type",
    "Parameter unit",
    "Parameter value",
    "Time value",
    "Time unit",
]

COLUMN_NAMES_MAP = {
    "population": "Population",
    "Gestational age": "Pediatric/Gestational age",
    "drug name": "Drug name",
    "patient ID": "Patient ID",
    "Parameter": "Parameter type",
    "Time": "Time value",
    "Value": "Parameter value",
    "Unit": "Parameter unit",
}

def normalize_pk_individual_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the PK individual baseline table from wide format to long format.
    
    Converts columns like 'Parameter 1', 'drug name 1', 'Time 1', etc. into rows
    with standardized column names: 'Parameter', 'drug name', 'Time', etc.
    
    Args:
        df: DataFrame with wide format columns
        
    Returns:
        DataFrame with long format (normalized)
    """
    # Define the base columns that don't change
    base_columns = ['PMID', 'population', 'Pregnancy stage', 'Gestational age', 
                   'Specimen', 'drug name', 'patient ID']
    
    # Find all parameter groups (Parameter 1, Parameter 2, etc.)
    parameter_cols = [col for col in df.columns if col.startswith('Parameter ')]
    
    # Extract parameter numbers
    param_numbers = []
    for col in parameter_cols:
        try:
            num = int(col.split(' ')[-1])
            param_numbers.append(num)
        except ValueError:
            continue
    
    param_numbers.sort()
    
    # Create list to store normalized rows
    normalized_rows = []
    
    for ix, row in df.iterrows():
        # Extract base information
        base_info = {col: row[col] for col in base_columns if col in df.columns}
        
        # For each parameter group, create a new row
        for param_num in param_numbers:
            param_col = f'Parameter {param_num}'
            drug_col = f'drug name {param_num}'
            time_col = f'Time {param_num}'
            time_unit_col = f'Time unit {param_num}'
            value_col = f'Value {param_num}'
            unit_col = f'Unit {param_num}'
            
            # Check if this parameter group has data
            if param_col in df.columns and pd.notna(row[param_col]):
                normalized_row = base_info.copy()
                normalized_row['Parameter'] = row[param_col]
                normalized_row['Analyte'] = row[drug_col] if drug_col in df.columns else None
                normalized_row['Time'] = row[time_col] if time_col in df.columns else None
                normalized_row['Time unit'] = row[time_unit_col] if time_unit_col in df.columns else None
                normalized_row['Value'] = row[value_col] if value_col in df.columns else None
                normalized_row['Unit'] = row[unit_col] if unit_col in df.columns else None
                
                normalized_rows.append(normalized_row)
            else:
                print(f"No data found for parameter {param_num} in row {ix}")
    
    # Create the normalized DataFrame
    normalized_df = pd.DataFrame(normalized_rows)
    
    # Ensure the column order matches the expected output
    expected_columns = ['PMID', 'population', 'Pregnancy stage', 'Gestational age', 
                       'Specimen', 'drug name', 'patient ID', 'Parameter', 'Analyte', 
                       'Time', 'Time unit', 'Value', 'Unit']
    
    # Reorder columns to match expected format
    if not normalized_df.empty:
        normalized_df = normalized_df[expected_columns]
        normalized_df = normalized_df.rename(columns=COLUMN_NAMES_MAP)
    
    return normalized_df


def main():
    source_csv_files = [
        'benchmark/utils/pk_individual_baseline/10971311_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/11849190_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/18426260_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/23200982_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/24989434_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/32056930_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/32153014_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/34746508_baseline_original.csv',
        'benchmark/utils/pk_individual_baseline/33253437_baseline_original.csv',
    ]
    for source_csv_file in source_csv_files:
        df = pd.read_csv(source_csv_file)
        normalized_df = normalize_pk_individual_baseline(df)
        print(normalized_df)
        normalized_df.to_csv(source_csv_file.replace('_baseline_original.csv', '_baseline_normalized.csv'), index=False)

if __name__ == "__main__":
    main()