from src.models.hybrid.kbm_calculator import HybridKBMCalculator
import pandas as pd


def main():
    print(" Тестик")
    print("=" * 85)

    calculator = HybridKBMCalculator(model_path="outputs/insurance_model_v1.cbm")

    cases = [
        {
            'driver_age': 55,
            'driver_experience': 30,
            'vehicle_age': 7,
            'vehicle_type': 'suv',
            'engine_power': 219,
            'vehicle_purpose': 'personal',
            'region': 'rural',
            'pct_days_with_snow': 0.5,
            'pct_days_with_rain': 0.325,
            'winter_duration_months': 6,
            'base_kbm': 0.85,
            'num_claims': 0,
            'violation_count': 0,
            'days_since_last_claim': 1095,
            'occupation_type': 'retired',
            'avg_trips_per_week': 2.0,
            'night_driving_ratio': 0.016,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Опытный водитель (низкий риск)'
        },
        {
            'driver_age': 45,
            'driver_experience': 20,
            'vehicle_age': 5,
            'vehicle_type': 'suv',
            'engine_power': 180,
            'vehicle_purpose': 'personal',
            'region': 'rural',
            'pct_days_with_snow': 0.5,
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
            'description': 'Аккуратный пенсионер'
        },
        {
            'driver_age': 29,
            'driver_experience': 5,
            'vehicle_age': 3,
            'vehicle_type': 'sedan',
            'engine_power': 150,
            'vehicle_purpose': 'personal',
            'region': 'Moscow',
            'pct_days_with_snow': 0.35,
            'pct_days_with_rain': 0.45,
            'winter_duration_months': 5,
            'base_kbm': 1.0,
            'num_claims': 2,
            'violation_count': 1,
            'days_since_last_claim': 400,
            'occupation_type': 'office_worker',
            'avg_trips_per_week': 8,
            'night_driving_ratio': 0.1,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Средний водитель с 2 ДТП'
        },
        {
            'driver_age': 22,
            'driver_experience': 2,
            'vehicle_age': 1,
            'vehicle_type': 'hatchback',
            'engine_power': 105,
            'vehicle_purpose': 'personal',
            'region': 'urban',
            'pct_days_with_snow': 0.3,
            'pct_days_with_rain': 0.486,
            'winter_duration_months': 4,
            'base_kbm': 1.55,
            'num_claims': 3,
            'violation_count': 4,
            'days_since_last_claim': 50,
            'occupation_type': 'student',
            'avg_trips_per_week': 15.0,
            'night_driving_ratio': 0.5,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Молодой водитель с высоким риском'
        }
    ]

    result_df = calculator.calculate(
        cases=cases,
        obd_file_path=None,
        show_plot=True
    )

    print("\n АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("-" * 85)

    low_risk = result_df.iloc[0]
    high_risk = result_df.iloc[-1]

    if low_risk['Рекомендуемый КБМ'] <= low_risk['Базовый КБМ']:
        print("Низкорисковый профиль: КБМ снижен (логично)")
    else:
        print("Низкорисковый профиль: КБМ не снижен ")

    if high_risk['Рекомендуемый КБМ'] > high_risk['Базовый КБМ']:
        print("Высокорисковый профиль: КБМ повышен (логично)")
    else:
        print("Высокорисковый профиль: КБМ не повышен")

    return result_df


if __name__ == "__main__":
    main()