import pandas as pd
from src.models.insurance_model import InsuranceRiskModel

def test_sample_case_2():
    model = InsuranceRiskModel("outputs/insurance_model_v1.pkl")

    realistic_case = pd.DataFrame([{
        "driver_age": 28,
        "driver_experience": 5,
        "vehicle_age": 3,
        "vehicle_type": "crossover",
        "region": "urban",
        "has_violations": 1,
        "num_claims": 1,
        "accident_history_score": 4.5,
        "weather_condition": "rainy",
        "road_type": "urban",
        "traffic_density": "high",
        "trip_purpose": "commute"
    }])

    proba = model.predict_proba(realistic_case)
    tariff = model.calculate_tariff(proba)

    print(f"Вероятность ДТП: {proba * 100:.2f}%")
    print(f"Рассчитанный страховой тариф: {tariff:.2f}₽")

if __name__ == "__main__":
    test_sample_case_2()
