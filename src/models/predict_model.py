import pandas as pd
from src.models.insurance_model import InsuranceRiskModel
import sys

def main():

    new_case = pd.DataFrame([{
        'driver_age': 29,
        'driver_experience': 5,
        'vehicle_age': 3,
        'vehicle_type': 'sedan',
        'region': 'Moscow',
        'has_violations': 1,
        'num_claims': 2,
        'accident_history_score': 0.7,
        'weather_condition': 'snow',
        'road_type': 'urban',
        'traffic_density': 'high',
        'trip_purpose': 'commute'
    }])

    # Загрузка модели
    model = InsuranceRiskModel(model_path="outputs/insurance_model_v1.pkl")

    # Предсказание вероятности
    proba = model.predict_proba(new_case)
    tariff = model.calculate_tariff(proba)

    print(f"\nВероятность ДТП: {proba:.2%}")
    print(f"Рассчитанный страховой тариф: {tariff:.2f}₽")

if __name__ == "__main__":
    main()
