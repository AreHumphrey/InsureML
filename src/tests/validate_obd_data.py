
import pandas as pd
import numpy as np
from pathlib import Path


def validate_columns(df):
    required_columns = {'tripID', 'timeStamp', 'gps_speed', 'accData'}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"❌ Отсутствуют обязательные столбцы: {missing}")
        return False
    else:
        print("✅ Все необходимые столбцы присутствуют.")
        return True

def parse_accData(acc_str):

    try:

        acc_str = str(acc_str).strip()

        if acc_str.startswith('[') and ']' in acc_str:
            values = [float(x.strip()) for x in acc_str.strip('[]').split(',')]
            if len(values) >= 3:
                return values[0], values[1], values[2]

        if len(acc_str) > 10 and all(c in '0123456789abcdefABCDEF' for c in acc_str):

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
                return np.nan, np.nan, np.nan

        return np.nan, np.nan, np.nan

    except Exception as e:
        print(f"⚠️ Ошибка при парсинге accData: {e}")
        return np.nan, np.nan, np.nan

def extract_acceleration_components(df):
    print("🔧 Парсинг accData...")
    parsed = df['accData'].astype(str).apply(parse_accData)
    acc_df = pd.DataFrame(parsed.tolist(), columns=['accel_x', 'accel_y', 'accel_z'])
    result = pd.concat([df.reset_index(drop=True), acc_df], axis=1)
    return result

def clean_data(df):
    initial_len = len(df)
    df['gps_speed'] = pd.to_numeric(df['gps_speed'], errors='coerce')
    df.dropna(subset=['gps_speed', 'accData'], inplace=True)
    df = df[(df['gps_speed'] >= 0) & (df['gps_speed'] <= 200)]
    cleaned_len = len(df)
    print(f"🧹 После очистки осталось {cleaned_len} строк ({initial_len - cleaned_len} удалено).")
    return df

def assess_trip_quality(df):
    if len(df) == 0:
        print("❌ Нет данных после очистки.")
        return False

    df['dt'] = pd.to_datetime(df['timeStamp'], errors='coerce')
    df.dropna(subset=['dt'], inplace=True)
    if len(df) == 0:
        print("❌ Не удалось распарсить временные метки.")
        return False

    duration_seconds = (df['dt'].max() - df['dt'].min()).total_seconds()
    if duration_seconds < 10:
        print(f"⚠️  Поездка слишком короткая: {duration_seconds:.1f} секунд (<10 сек).")
        return False

    avg_speed = df['gps_speed'].mean()
    if avg_speed < 5:
        print(f"⚠️  Средняя скорость слишком низкая: {avg_speed:.1f} км/ч.")
        return False

    data_completeness = len(df) / (duration_seconds + 1)
    if data_completeness < 0.1:
        print(f"⚠️  Низкая частота данных: {data_completeness:.2f} Гц.")
        return False

    print(f"📊 Качество поездки: хорошее. Длительность: {duration_seconds:.0f} сек, средняя скорость: {avg_speed:.1f} км/ч, частота: {data_completeness:.2f} Гц.")
    return True

def extract_behavior_features(df):
    features = {}
    features['avg_speed'] = df['gps_speed'].mean()
    features['max_speed'] = df['gps_speed'].max()

    longitudinal_accel = df['accel_y']
    valid_accel = longitudinal_accel.dropna()
    if len(valid_accel) > 0:
        features['hard_brakes'] = ((valid_accel < -2.0) & (df['gps_speed'] > 10)).sum()
        features['hard_accels'] = (valid_accel > 2.0).sum()
    else:
        features['hard_brakes'] = 0
        features['hard_accels'] = 0

    lateral_accel = df['accel_x'].dropna()
    features['sharp_turns'] = (np.abs(lateral_accel) > 3.0).sum()

    night_driving = df['dt'].dt.hour.isin(range(0, 6))
    features['night_driving_ratio'] = night_driving.mean()

    if 'dtc' in df.columns:
        dtc_vals = pd.to_numeric(df['dtc'], errors='coerce').fillna(0)
        features['has_dtc_errors'] = (dtc_vals > 0).any()
    else:
        features['has_dtc_errors'] = False

    print("\n📊 Извлечённые признаки поведения:")
    for k, v in features.items():
        print(f"   • {k}: {v:.2f}" if isinstance(v, float) else f"   • {k}: {v}")

    return features


def main():
    file_path = "src/data/raw/obd_data.csv"
    print(f"🔍 Проверка основного файла телематики: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ Файл не найден: {file_path}")
        return

    df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    print(f"✅ Загружено {len(df)} строк из obd_data.csv.")

    if not validate_columns(df):
        print("❌ Данные не подходят.")
        return

    df_clean = clean_data(df)
    if len(df_clean) == 0:
        print("❌ Нет данных после очистки.")
        return

    df_with_accel = extract_acceleration_components(df_clean)
    if df_with_accel[['accel_x', 'accel_y', 'accel_z']].isna().all().all():
        print("⚠️  Не удалось распарсить ни одного вектора ускорения.")
        return


    print("\n" + "="*70)
    print("АНАЛИЗ ПОЕЗДОК")
    print("="*70)

    trip_analysis = []
    for trip_id, group in df_with_accel.groupby('tripID'):
        print(f"\n--- Поездка ID: {trip_id} ---")
        if assess_trip_quality(group):
            feats = extract_behavior_features(group)
            trip_analysis.append({
                'tripID': trip_id,
                'duration_sec': (group['dt'].max() - group['dt'].min()).total_seconds(),
                'avg_speed': feats['avg_speed'],
                'hard_brakes': feats['hard_brakes'],
                'night_driving_ratio': feats['night_driving_ratio'],
                'has_dtc_errors': feats['has_dtc_errors']
            })

    if trip_analysis:
        summary = pd.DataFrame(trip_analysis)
        print("\n" + "="*70)
        print("СВОДКА ПО ВСЕМ ПОЕЗДКАМ")
        print("="*70)
        print(f"Всего поездок: {len(summary)}")
        print(f"Средняя длительность: {summary['duration_sec'].mean():.0f} сек")
        print(f"Средняя скорость: {summary['avg_speed'].mean():.1f} км/ч")
        print(f"Среднее число резких тормозов: {summary['hard_brakes'].mean():.1f}")
        print(f"Доля поездок ночью: {summary['night_driving_ratio'].mean():.1%}")
        print(f"Доля поездок с ошибками DTC: {summary['has_dtc_errors'].mean():.1%}")


        summary.to_csv("src/data/validation_summary.csv", index=False)
        print("✅ Сводка сохранена: data/validation_summary.csv")
    else:
        print("❌ Ни одна поездка не прошла проверку качества.")

if __name__ == "__main__":
    main()