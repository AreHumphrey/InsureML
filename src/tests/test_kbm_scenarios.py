

from src.models.hybrid.osago_calculator import OSAGOCalculator
import pandas as pd


def test_case(name: str, obd_file_path: str = None):
    print(f"\n СЦЕНАРИЙ: {name}")
    if obd_file_path:
        print(f" Используется файл: {obd_file_path}")
    else:
        print(" OBD-файл не предоставлен")

    calc = OSAGOCalculator(model_path="outputs/insurance_model_v1.cbm")

    driver_case = pd.DataFrame([{
        'driver_age': 35,
        'driver_experience': 10,
        'vehicle_age': 3,
        'vehicle_type': 'sedan',
        'engine_power': 150,
        'vehicle_purpose': 'personal',
        'region': 'Moscow',
        'pct_days_with_snow': 0.35,
        'pct_days_with_rain': 0.45,
        'winter_duration_months': 5,
        'base_kbm': 1.0,
        'num_claims': 1,
        'violation_count': 0,
        'days_since_last_claim': 600,
        'occupation_type': 'office_worker',
        'avg_trips_per_week': 8,
        'night_driving_ratio': 0.1,
        'ko_multiplier': 1.0,
        'num_owned_vehicles': 1,
        'description': 'Тестовый водитель'
    }])

    result = calc.calculate_osago_premium(
        driver_data=driver_case,
        obd_file_path=obd_file_path,
        base_tariff=2000.0,
        region_coeff=1.8,
        engine_power_coeff=1.2,
        age_exp_coeff=1.4,
        unlimited_drivers=False,
        season_coeff=1.0
    )

    print(f"\n РЕЗУЛЬТАТЫ РАСЧЁТА ОСАГО")
    print("=" * 50)
    print(f"   • Базовый КБМ:     {result['base_kbm']:.2f}")
    print(f"   • Итоговый КБМ:    {result['final_kbm']:.2f}")
    print(f"   • Итоговый тариф:  {result['tariff']:.2f} ₽")
    print("=" * 50)

    return result


def main():
    print("Тестик")
    print("=" * 80)

    test_case("Только анкета", obd_file_path=None)

    test_case("Спокойная поездка", obd_file_path="src/data/tests/v2_no_dtc.csv")

    test_case("Поездка с DTC", obd_file_path="src/data/tests/v2_with_dtc.csv")

    test_case("Агрессивное вождение", obd_file_path="src/data/tests/v2_aggressive.csv")

    print("\nВсе сценарии протестированы.")


if __name__ == "__main__":
    main()