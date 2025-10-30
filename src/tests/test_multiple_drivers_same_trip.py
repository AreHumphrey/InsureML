# src/tests/test_multiple_drivers_same_trip.py
"""
Тестирование одного и того же OBD-файла для разных водителей.
"""

from src.models.hybrid.kbm_calculator import HybridKBMCalculator


def main():
    print("🚀 ТЕСТИРОВАНИЕ ОДНОЙ ПОЕЗДКИ ДЛЯ РАЗНЫХ ВОДИТЕЛЕЙ")
    print("=" * 90)

    # Инициализация калькулятора
    calculator = HybridKBMCalculator(model_path="src/outputs/insurance_model_v1.cbm")

    # Разные водители — от низкого до высокого риска
    drivers = [
        {
            'driver_age': 55,
            'driver_experience': 30,
            'vehicle_age': 5,
            'vehicle_type': 'suv',
            'engine_power': 180,
            'vehicle_purpose': 'personal',
            'region': 'rural',
            'pct_days_with_snow': 0.4,
            'pct_days_with_rain': 0.3,
            'winter_duration_months': 6,
            'base_kbm': 0.85,
            'num_claims': 0,
            'violation_count': 0,
            'days_since_last_claim': 1095,
            'occupation_type': 'retired',
            'avg_trips_per_week': 2,
            'night_driving_ratio': 0.05,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Пенсионер (низкий риск)'
        },
        {
            'driver_age': 35,
            'driver_experience': 12,
            'vehicle_age': 2,
            'vehicle_type': 'van',
            'engine_power': 160,
            'vehicle_purpose': 'commercial',
            'region': 'Moscow',
            'pct_days_with_snow': 0.35,
            'pct_days_with_rain': 0.45,
            'winter_duration_months': 5,
            'base_kbm': 1.0,
            'num_claims': 1,
            'violation_count': 0,
            'days_since_last_claim': 600,
            'occupation_type': 'courier',
            'avg_trips_per_week': 20,
            'night_driving_ratio': 0.4,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Курьер (высокая нагрузка)'
        },
        {
            'driver_age': 24,
            'driver_experience': 3,
            'vehicle_age': 1,
            'vehicle_type': 'hatchback',
            'engine_power': 105,
            'vehicle_purpose': 'personal',
            'region': 'urban',
            'pct_days_with_snow': 0.3,
            'pct_days_with_rain': 0.4,
            'winter_duration_months': 4,
            'base_kbm': 1.0,
            'num_claims': 0,
            'violation_count': 2,
            'days_since_last_claim': 100,
            'occupation_type': 'student',
            'avg_trips_per_week': 6,
            'night_driving_ratio': 0.3,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Молодой водитель с нарушениями'
        }
    ]

    # Список OBD-файлов для тестирования
    obd_files = [
        None,
        "src/data/tests/v2_no_dtc.csv",
        "src/data/tests/v2_with_dtc.csv",
        "src/data/tests/v2_aggressive.csv"
    ]

    # Заголовок таблицы
    print(f"{'Водитель':<25} {'Файл':<25} {'Базовый КБМ':<12} {'Рекоменд. КБМ':<14} {'Итоговый КБМ':<13} {'DTC'}")
    print("-" * 90)

    for driver in drivers:
        for obd_file in obd_files:
            file_name = obd_file or "Нет файла"
            has_dtc_str = "—"

            # Создаём временный список из одного водителя
            result_df = calculator.calculate(
                cases=[driver],
                obd_file_path=obd_file,
                show_plot=False
            )

            row = result_df.iloc[0]
            base_kbm = row['Базовый КБМ']
            adj_kbm = row['Рекомендуемый КБМ']
            final_kbm = row['Итоговый КБМ']

            # Определяем, был ли DTC
            if 'наличие DTC' in row['Корректировки']:
                has_dtc_str = "Да"
            elif 'нет' not in row['Корректировки']:
                has_dtc_str = "Нет"

            print(f"{driver['description']:<25} "
                  f"{file_name.split('/')[-1]:<25} "
                  f"{base_kbm:<12} "
                  f"{adj_kbm:<14} "
                  f"{final_kbm:<13} "
                  f"{has_dtc_str}")

        print("─" * 90)  # Разделитель между водителями

    print("✅ Все комбинации протестированы.")


if __name__ == "__main__":
    main()