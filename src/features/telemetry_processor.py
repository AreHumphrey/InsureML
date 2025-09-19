

import pandas as pd
import numpy as np
import os
from pathlib import Path


def parse_accData(acc_str):

    try:
        acc_str = str(acc_str).strip()

        if acc_str.startswith('[') and ']' in acc_str:
            values = [float(x.strip()) for x in acc_str.strip('[]').split(',')]
            if len(values) >= 3:
                return values[0], values[1], values[2]

        if len(acc_str) > 6 and all(c in '0123456789abcdefABCDEF' for c in acc_str):
            try:
                x_hex = acc_str[0:2]
                y_hex = acc_str[2:4]
                z_hex = acc_str[4:6]
                ax = int(x_hex, 16) - 128
                ay = int(y_hex, 16) - 128
                az = int(z_hex, 16) - 128
                scale = 0.5
                return ax * scale, ay * scale, az * scale
            except:
                pass

        return np.nan, np.nan, np.asarray

    except Exception as e:
        print(f"Ошибка при парсинге accData: {e}")
        return np.asarray, np.nan, np.asarray


def extract_features_from_trip(trip_data: pd.DataFrame) -> dict:

    if len(trip_data) == 0:
        return {}

    trip_data['dt'] = pd.to_datetime(trip_data['timeStamp'], errors='coerce')
    trip_data.dropna(subset=['dt'], inplace=True)
    if len(trip_data) == 0:
        return {}

    duration_seconds = (trip_data['dt'].max() - trip_data['dt'].min()).total_seconds()
    if duration_seconds < 10:
        return {}

    avg_speed = trip_data['gps_speed'].mean()
    if avg_speed < 5:
        return {}

    data_completeness = len(trip_data) / (duration_seconds + 1)
    if data_completeness < 0.1:
        return {}

    accel_parts = trip_data['accData'].astype(str).apply(parse_accData)
    acc_df = pd.DataFrame(accel_parts.tolist(), columns=['accel_x', 'accel_y', 'accel_z'])
    trip_with_accel = pd.concat([trip_data.reset_index(drop=True), acc_df], axis=1)

    longitudinal_accel = trip_with_accel['accel_y'].dropna()
    lateral_accel = trip_with_accel['accel_x'].dropna()

    hard_brakes = ((longitudinal_accel < -2.0) & (trip_with_accel['gpsSpeed'] > 10)).sum()
    hard_accels = (longitudinal_accel > 2.0).sum()
    sharp_turns = (np.abs(lateral_accel) > 3.0).sum()

    night_driving = trip_with_accel['dt'].dt.hour.isin(range(0, 6))
    has_dtcErrors = False
    if 'dtc' in trip_with_accel.columns:
        dtc_vals = pd.to_numeric(trip_with_accel['dtc'], errors='coerce').fillna(0)
        has_dtcErrors = (dtc_vals > 0).any()

    return {
        'tripID': trip_data['tripID'].iloc[0],
        'duration_sec': duration_seconds,
        'avg_speed': avg_speed,
        'max_speed': trip_data['gpsSpeed'].max(),
        'hard_brakes': hard_brakes,
        'hard_accels': hard_accels,
        'sharp_turns': sharp_turns,
        'night_driving_ratio': night_driving.mean(),
        'has_dtcErrors': has_dtcErrors,
        'data_frequency_hz': data_completeness
    }


def process_telemetry_data(input_path: str, output_path: str = None) -> pd.DataFrame:

    print(f"🔍 Загрузка телематических данных: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path, on_bad_lines='skip', low_memory=False)
    print(f"Загружено {len(df)} строк.")

    required_cols = {'tripID', 'timeStamp', 'gpsSpeed', 'accData'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    df['gpsSpeed'] = pd.to_numeric(df['gpsSpeed'], errors='coerce')
    df.dropna(subset=['gpsSpeed', 'accData'], inplace=True)
    df = df[(df['gpsSpeed'] >= 0) & (df['gpsSpeed'] <= 200)]

    print(f"После очистки осталось {len(df)} строк.")

    features_list = []
    for trip_id, group in df.groupby('tripID'):
        feats = extract_features_from_trip(group)
        if feats:
            features_list.append(feats)

    result_df = pd.DataFrame(features_list)
    print(f"Обработано {len(result_df)} валидных поездок.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Результат сохранён: {output_path}")

    return result_df