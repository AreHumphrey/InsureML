
from src.models.hybrid.kbm_calculator import HybridKBMCalculator


def test_case(name: str, obd_file_path: str = None):
    print(f"\nСЦЕНАРИЙ: {name}")
    if obd_file_path:
        print(f"Используется файл: {obd_file_path}")
    else:
        print("OBD-файл не предоставлен")

    calculator = HybridKBMCalculator(model_path="src/outputs/insurance_model_v1.cbm")

    case = {
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
    }

    result_df = calculator.calculate(
        cases=[case],
        obd_file_path=obd_file_path,
        show_plot=False
    )

    return result_df


def main():
    print("ЗАПУСК ТЕСТОВЫХ СЦЕНАРИЕВ ГИБРИДНОГО КБМ")
    print("=" * 80)

    test_case("Только анкета", obd_file_path=None)

    test_case("Спокойная поездка", obd_file_path="src/data/tests/v2_no_dtc.csv")

    test_case("Поездка с DTC", obd_file_path="src/data/tests/v2_with_dtc.csv")

    test_case("Агрессивное вождение", obd_file_path="src/data/tests/v2_aggressive.csv")

    print("\nВсе сценарии протестированы.")


if __name__ == "__main__":
    main()