import pandas as pd
import numpy as np


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['age_squared'] = df['driver_age'] ** 2
    df['experience_ratio'] = df['driver_experience'] / (df['driver_age'] + 1e-5)
    df['claims_per_year'] = df['num_claims'] / (df['driver_experience'] + 1e-5)
    df['violations_per_year'] = df['violation_count'] / (df['driver_experience'] + 1e-5)


    df['no_claims_long'] = ((df['num_claims'] == 0) & (df['driver_experience'] >= 5)).astype(int)
    df['experienced_clean'] = (
            (df['driver_experience'] >= 10) &
            (df['num_claims'] == 0) &
            (df['violation_count'] == 0)
    ).astype(int)
    df['young_risky'] = ((df['driver_age'] < 26) & (df['violation_count'] > 0)).astype(int)
    df['high_claims'] = (df['num_claims'] >= 2).astype(int)

    df['night_x_trips'] = df['night_driving_ratio'] * df['avg_trips_per_week']
    df['power_x_age'] = df['engine_power'] * df['vehicle_age']

    return df