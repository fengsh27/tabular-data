
import pandas as pd

def common_preprocess(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    return df

