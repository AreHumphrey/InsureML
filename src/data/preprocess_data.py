import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna() 
    df.columns = df.columns.str.lower().str.strip()
    return df