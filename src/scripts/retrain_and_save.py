
from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
import os


def main():
    print("Создаём тестовые данные для переобучения...")

    X_train = pd.DataFrame([
        {
            'driver_age': 30, 'driver_experience': 8, 'vehicle_age': 3,
            'vehicle_type': 'sedan', 'engine_power': 150,
            'vehicle_purpose': 'personal', 'region': 'Moscow',
            'pct_days_with_snow': 0.35, 'pct_days_with_rain': 0.45,
            'winter_duration_months': 5, 'base_kbm': 1.0,
            'num_claims': 1, 'violation_count': 1,
            'days_since_last_claim': 365, 'occupation_type': 'office_worker',
            'avg_trips_per_week': 8, 'night_driving_ratio': 0.1,
            'ko_multiplier': 1.0, 'num_owned_vehicles': 1
        },
        {
            'driver_age': 25, 'driver_experience': 3, 'vehicle_age': 1,
            'vehicle_type': 'hatchback', 'engine_power': 105,
            'vehicle_purpose': 'personal', 'region': 'urban',
            'pct_days_with_snow': 0.3, 'pct_days_with_rain': 0.4,
            'winter_duration_months': 4, 'base_kbm': 1.0,
            'num_claims': 0, 'violation_count': 2,
            'days_since_last_claim': 100, 'occupation_type': 'student',
            'avg_trips_per_week': 6, 'night_driving_ratio': 0.3,
            'ko_multiplier': 1.0, 'num_owned_vehicles': 1
        }
    ])

    y_train = [0, 1]

    cat_features = ['vehicle_type', 'region', 'vehicle_purpose', 'occupation_type']
    for col in cat_features:
        if col in X_train.columns:
            known_categories = X_train[col].astype('category').cat.categories.tolist()
            if "unknown" not in known_categories:
                known_categories.append("unknown")
            X_train[col] = pd.Categorical(X_train[col], categories=known_categories)

    print("Данные готовы. Обучаем модель...")

    model = InsuranceRiskModel()
    model.train(X_train, pd.Series(y_train))

    output_dir = "src/outputs"
    os.makedirs(output_dir, exist_ok=True)

    model.model.save_model(f"{output_dir}/insurance_model_v1.cbm")
    print("Модель сохранена как ../outputs/insurance_model_v1.cbm")

    model.save_model(f"{output_dir}/insurance_model_v1_full.pkl")
    print("Модель сохранена как ../outputs/insurance_model_v1_full.pkl")


if __name__ == "__main__":
    main()