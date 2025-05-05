import pandas as pd
from src.models.insurance_model import InsuranceRiskModel

def test_sample_case():
    model = InsuranceRiskModel("outputs/insurance_model_v1.pkl")

    new_case = pd.DataFrame([{
        "driver_age": 42,
        "driver_experience": 20,
        "vehicle_age": 6,
        "vehicle_type": "hatchback",
        "region": "rural",
        "has_violations": 0,
        "num_claims": 0,
        "accident_history_score": 0.1,
        "weather_condition": "normal",
        "road_type": "highway",
        "traffic_density": "low",
        "trip_purpose": "personal"
    }])

    proba = model.predict_proba(new_case)
    tariff = model.calculate_tariff(proba)

    print(f"Вероятность ДТП: {proba * 100:.2f}%")
    print(f"Рассчитанный страховой тариф: {tariff:.2f}₽")

if __name__ == "__main__":
    test_sample_case()