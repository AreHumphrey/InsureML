import pandas as pd
from src.models.catboost.insurance_model import InsuranceRiskModel


def main():
    new_case = pd.DataFrame([{
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
        'num_owned_vehicles': 1
    }])

    model = InsuranceRiskModel(model_path="./outputs/insurance_model_v1.pkl")

    proba = model.predict_proba(new_case)

    kbm = model.calculate_adjusted_kbm(new_case, base_kbm=1.0)

    print(f"\nВероятность ДТП: {proba:.2%}")
    print(f"Рекомендуемый КБМ: {kbm}")
    print("Этот коэффициент может быть использован в официальной формуле расчёта ОСАГО.")


if __name__ == "__main__":
    main()