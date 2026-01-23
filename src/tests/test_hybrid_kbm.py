from src.models.hybrid.kbm_calculator import HybridKBMCalculator


def main():
    calculator = HybridKBMCalculator(model_path="src/outputs/insurance_model_v1.cbm")

    cases = [
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
        }

    ]

    result_df = calculator.calculate(
        cases=cases,
        obd_file_path="src/data/raw/v2.csv",
        show_plot=True
    )

    return result_df


if __name__ == "__main__":
    main()