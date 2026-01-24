import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_features = [
        'driver_age', 'driver_experience', 'vehicle_age', 'engine_power',
        'pct_days_with_snow', 'pct_days_with_rain', 'winter_duration_months',
        'base_kbm', 'num_claims', 'violation_count', 'days_since_last_claim',
        'avg_trips_per_week', 'night_driving_ratio', 'ko_multiplier', 'num_owned_vehicles'
    ]

    for col in num_features:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    cat_features = [
        'vehicle_type', 'region', 'vehicle_purpose', 'occupation_type'
    ]

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Клипинг
    df['driver_age'] = df['driver_age'].clip(18, 80)
    df['driver_experience'] = df['driver_experience'].clip(0, 60)
    df['engine_power'] = df['engine_power'].clip(60, 300)
    df['vehicle_age'] = df['vehicle_age'].clip(0, 30)
    df['days_since_last_claim'] = df['days_since_last_claim'].clip(30, 1095)
    df['night_driving_ratio'] = df['night_driving_ratio'].clip(0, 1)
    df['pct_days_with_rain'] = df['pct_days_with_rain'].clip(0, 1)

    for col in ['num_claims', 'violation_count', 'num_owned_vehicles']:
        df[col] = df[col].astype(int)

    print("Предобработка завершена")
    return df