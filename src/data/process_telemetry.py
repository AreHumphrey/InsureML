import pandas as pd
import numpy as np
import os

def extract_features_from_trip(trip_data: pd.DataFrame) -> pd.Series:

    def to_numeric_safe(col):
        return pd.to_numeric(col, errors='coerce')

    trip_data['gps_speed'] = to_numeric_safe(trip_data['gps_speed'])
    trip_data['cTemp'] = to_numeric_safe(trip_data['cTemp'])
    trip_data['iat'] = to_numeric_safe(trip_data['iat'])

    avg_speed = trip_data['gps_speed'].mean()
    max_speed = trip_data['gps_speed'].max()
    std_speed = trip_data['gps_speed'].std()

    trip_data['timestamp'] = pd.to_datetime(trip_data['timeStamp'], errors='coerce')
    valid_times = trip_data.dropna(subset=['timestamp'])
    if len(valid_times) == 0:
        night_ratio = 0.0
    else:
        night_ratio = ((valid_times['timestamp'].dt.hour >= 22) | (valid_times['timestamp'].dt.hour < 6)).mean()

    duration_sec = (trip_data['timestamp'].iloc[-1] - trip_data['timestamp'].iloc[0]).seconds
    distance_km = avg_speed * (duration_sec / 3600)

    return pd.Series({
        'avg_speed': avg_speed if pd.notna(avg_speed) else 0.0,
        'max_speed': max_speed if pd.notna(max_speed) else 0.0,
        'std_speed': std_speed if pd.notna(std_speed) else 0.0,
        'night_driving_ratio': night_ratio,
        'trip_duration_min': duration_sec / 60 if duration_sec > 0 else 0,
        'distance_km': distance_km if pd.notna(distance_km) else 0.0,
        'has_dtc_errors': (to_numeric_safe(trip_data['dtc']) > 0).any(),
        'avg_coolant_temp': trip_data['cTemp'].mean() if pd.notna(trip_data['cTemp'].mean()) else 60.0,
        'avg_iat': trip_data['iat'].mean() if pd.notna(trip_data['iat'].mean()) else 25.0
    })

def process_raw_telemetry():

    print("Загрузка и обработка obd_data.csv...")

    df = pd.read_csv(
        "src/data/raw/obd_data.csv",
        low_memory=False,
        on_bad_lines='skip'
    )

    TRIP_ID_COLUMN = 'tripID'

    if TRIP_ID_COLUMN not in df.columns:
        raise KeyError(f"Колонка '{TRIP_ID_COLUMN}' не найдена. Доступны: {df.columns.tolist()}")

    df = df[df[TRIP_ID_COLUMN] != 'tripID']
    df = df.dropna(subset=[TRIP_ID_COLUMN])
    df[TRIP_ID_COLUMN] = df[TRIP_ID_COLUMN].astype(str).str.strip()

    print(f"Загружено {len(df)} строк телематики после очистки")

    if len(df) == 0:
        print("Нет данных для обработки")
        return

    features_list = []
    for trip_id, group in df.groupby(TRIP_ID_COLUMN):
        if len(group) <= 1:
            continue
        feats = extract_features_from_trip(group)
        feats['TripID'] = trip_id
        features_list.append(feats)

    result_df = pd.DataFrame(features_list)
    result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()

    risk_score = (
        0.4 * (result_df['std_speed'] / result_df['avg_speed']) +
        0.3 * result_df['night_driving_ratio'] +
        0.2 * (result_df['max_speed'] > 100).astype(int) +
        0.1 * (result_df['std_speed'] > result_df['std_speed'].quantile(0.8))
    )
    result_df['target'] = (risk_score > risk_score.median()).astype(int)

    output_path = "data/raw/telematics_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Обработанный датасет сохранён: {output_path}")
    print(f"Количество поездок: {len(result_df)}")

if __name__ == "__main__":
    process_raw_telemetry()