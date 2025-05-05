import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df.columns = df.columns.str.lower().str.strip()
    return df
