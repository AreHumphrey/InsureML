import pandas as pd
import numpy as np


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['age_squared'] = df['driver_age'] ** 2
    df['experience_ratio'] = df['driver_experience'] / (df['driver_age'] + 1e-5)
    df['claims_per_year'] = df['num_claims'] / (df['driver_experience'] + 1e-5)
    df['violations_per_year'] = df['violation_count'] / (df['driver_experience'] + 1e-5)

    df['claims_recent'] = (df['days_since_last_claim'] < 365).astype(int)
    df['high_night_driving'] = (df['night_driving_ratio'] > 0.3).astype(int)
    df['young_inexperienced'] = ((df['driver_age'] < 25) & (df['driver_experience'] < 3)).astype(int)
    df['power_to_age'] = df['engine_power'] / (df['vehicle_age'] + 1e-5)
    df['claim_violation_ratio'] = df['num_claims'] / (df['violation_count'] + 1)

    return df