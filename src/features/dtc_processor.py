# src/features/dtc_processor.py
"""
Простой процессор для извлечения признаков на основе DTC.
"""

import pandas as pd
from pathlib import Path


def load_and_process_dtc_data(file_path: str) -> pd.DataFrame:
    """
    Загружает CSV и извлекает признаки: наличие DTC, средняя скорость, длительность.
    """
    print(f"🔍 Загрузка данных: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ Загружено {len(df)} строк.")

    # Очистка
    df['dtc'] = pd.to_numeric(df['dtc'], errors='coerce').fillna(0)
    df['gps_speed'] = pd.to_numeric(df['gps_speed'], errors='coerce')
    df.dropna(subset=['gps_speed', 'dtc'], inplace=True)
    df = df[(df['gps_speed'] >= 0) & (df['gps_speed'] <= 200)]

    # Группировка по tripID
    features_list = []
    for trip_id, group in df.groupby('tripID'):
        duration_seconds = (pd.to_datetime(group['timeStamp']).max() - pd.to_datetime(group['timeStamp']).min()).total_seconds()
        avg_speed = group['gps_speed'].mean()

        has_dtc_errors = (group['dtc'] > 0).any()

        features_list.append({
            'tripID': trip_id,
            'duration_sec': float(duration_seconds),
            'avg_speed': float(avg_speed),
            'has_dtc_errors': bool(has_dtc_errors)
        })

    result_df = pd.DataFrame(features_list)
    print(f"📊 Обработано {len(result_df)} валидных поездок.")
    return result_df