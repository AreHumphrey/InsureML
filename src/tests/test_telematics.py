# src/tests/test_telematics.py
"""
Тестирование модели на obd_data_large.csv.
Прямо используем имеющиеся столбцы accel_x/y/z, без преобразования в accData.
"""

import unittest

import numpy as np
import pandas as pd
from pathlib import Path
from src.models.lightgbm.telematics_model import TelematicsRiskModel


class TestObdDataLarge(unittest.TestCase):

    def setUp(self):
        """Подготавливаем путь к файлу"""
        self.file_path = "src/data/raw/test_telemetry/obd_data_large.csv"
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {self.file_path}")

        self.df = pd.read_csv(self.file_path)
        print(f"✅ Загружено {len(self.df)} строк из {self.file_path}.")

    def test_columns_present(self):
        """Проверяем, что все нужные колонки есть"""
        required_cols = {
            'trip_id', 'driver_id', 'timestamp', 'gps_speed',
            'accel_x', 'accel_y', 'accel_z'
        }
        missing = required_cols - set(self.df.columns)

        self.assertEqual(len(missing), 0, f"❌ Отсутствуют колонки: {missing}")
        print("✅ Все необходимые колонки присутствуют.")

    def test_no_empty_gps_speed(self):
        """Проверяем, что нет пустых значений gps_speed"""
        invalid_rows = self.df[self.df['gps_speed'].isna()]
        self.assertEqual(len(invalid_rows), 0, f"❌ Есть строки с пустым gps_speed: {len(invalid_rows)}")
        print("✅ Нет пустых значений скорости.")

    def test_valid_speed_range(self):
        """Скорость должна быть в диапазоне 0–200 км/ч"""
        out_of_range = self.df[(self.df['gps_speed'] < 0) | (self.df['gps_speed'] > 200)]
        self.assertEqual(len(out_of_range), 0, f"❌ Скорость вне диапазона: {len(out_of_range)} строк")
        print("✅ Все значения скорости в допустимом диапазоне.")

    def test_trip_duration_and_features(self):
        """Извлекаем признаки для каждой поездки"""
        results = []

        for trip_id, group in self.df.groupby('trip_id'):
            # Преобразуем время
            group = group.copy()
            group['timestamp'] = pd.to_datetime(group['timestamp'], errors='coerce')
            group.dropna(subset=['timestamp'], inplace=True)
            group.sort_values('timestamp', inplace=True)

            duration_seconds = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()

            avg_speed = group['gps_speed'].mean()
            max_speed = group['gps_speed'].max()

            # Резкие ускорения/торможения (продольное ускорение)
            longitudinal_accel = group['accel_y']
            hard_brakes = ((longitudinal_accel < -2.0) & (group['gps_speed'] > 10)).sum()
            hard_accels = (longitudinal_accel > 2.0).sum()

            # Агрессивные повороты (боковое ускорение)
            lateral_accel = group['accel_x']
            sharp_turns = (np.abs(lateral_accel) > 3.0).sum()

            # Вождение ночью
            night_driving = group['timestamp'].dt.hour.isin(range(0, 6))
            night_driving_ratio = night_driving.mean()

            result = {
                'trip_id': trip_id,
                'duration_sec': duration_seconds,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'hard_brakes': int(hard_brakes),
                'hard_accels': int(hard_accels),
                'sharp_turns': int(sharp_turns),
                'night_driving_ratio': night_driving_ratio,
                'has_dtc_errors': False,
                'data_frequency_hz': len(group) / (duration_seconds + 1e-8)
            }
            results.append(result)

            print(f"\n📊 Поездка ID {int(trip_id)}:")
            print(f"   • Длительность:         {duration_seconds:.0f} сек")
            print(f"   • Средняя скорость:     {avg_speed:.1f} км/ч")
            print(f"   • Максимальная:         {max_speed:.1f} км/ч")
            print(f"   • Резкие тормоза:       {hard_brakes}")
            print(f"   • Резкие разгоны:       {hard_accels}")
            print(f"   • Агрессивные повороты: {sharp_turns}")
            print(f"   • Доля ночного:         {night_driving_ratio:.1%}")

        return pd.DataFrame(results)

    def test_model_prediction(self):
        """Тестируем модель на извлечённых признаках"""
        features_df = self.test_trip_duration_and_features()

        # Загружаем модель
        model = TelematicsRiskModel(model_path="src/outputs/lightgbm/telematics_model_v1.pkl")

        print("\n🎯 ПРЕДСКАЗАНИЕ МОДЕЛИ")
        print("=" * 50)

        for _, row in features_df.iterrows():
            # Подготовка данных для модели
            X = row.drop(['trip_id']).to_frame().T  # сделать DataFrame

            try:
                risk_score = model.predict_risk(X)
                print(f"Поездка ID {int(row['trip_id'])} → риск ДТП: {risk_score:.2%}")
            except Exception as e:
                print(f"❌ Ошибка при предсказании для поездки {row['trip_id']}: {e}")


if __name__ == '__main__':
    unittest.main()