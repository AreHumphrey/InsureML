import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df['age_squared'] = df['driver_age'] ** 2
    df['experience_ratio'] = df['driver_experience'] / (df['driver_age'] + 1)
    return df