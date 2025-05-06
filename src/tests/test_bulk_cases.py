import pandas as pd
import matplotlib.pyplot as plt
from src.models.insurance_model import InsuranceRiskModel

def test_bulk_cases_with_plot():
    model = InsuranceRiskModel("outputs/insurance_model_v1.pkl")

    bulk_data = pd.DataFrame([
        {
            "driver_age": 25, "driver_experience": 3, "vehicle_age": 2, "vehicle_type": "sedan",
            "region": "urban", "has_violations": 1, "num_claims": 2, "accident_history_score": 3.2,
            "weather_condition": "rainy", "road_type": "urban", "traffic_density": "high", "trip_purpose": "commute"
        },
        {
            "driver_age": 52, "driver_experience": 30, "vehicle_age": 5, "vehicle_type": "suv",
            "region": "rural", "has_violations": 0, "num_claims": 0, "accident_history_score": 0.0,
            "weather_condition": "clear", "road_type": "highway", "traffic_density": "low", "trip_purpose": "personal"
        },
        {
            "driver_age": 36, "driver_experience": 15, "vehicle_age": 10, "vehicle_type": "truck",
            "region": "suburban", "has_violations": 1, "num_claims": 1, "accident_history_score": 2.5,
            "weather_condition": "foggy", "road_type": "urban", "traffic_density": "medium",
            "trip_purpose": "commercial"
        },
        {
            "driver_age": 60, "driver_experience": 40, "vehicle_age": 7, "vehicle_type": "crossover",
            "region": "urban", "has_violations": 0, "num_claims": 0, "accident_history_score": 0.0,
            "weather_condition": "snowy", "road_type": "urban", "traffic_density": "medium", "trip_purpose": "personal"
        },
        {
            "driver_age": 19, "driver_experience": 1, "vehicle_age": 1, "vehicle_type": "hatchback",
            "region": "urban", "has_violations": 2, "num_claims": 3, "accident_history_score": 6.5,
            "weather_condition": "clear", "road_type": "urban", "traffic_density": "high", "trip_purpose": "personal"
        },
        {
            "driver_age": 45, "driver_experience": 25, "vehicle_age": 4, "vehicle_type": "van",
            "region": "suburban", "has_violations": 0, "num_claims": 0, "accident_history_score": 0.2,
            "weather_condition": "rainy", "road_type": "highway", "traffic_density": "low", "trip_purpose": "commercial"
        },
        {
            "driver_age": 31, "driver_experience": 10, "vehicle_age": 3, "vehicle_type": "sedan",
            "region": "urban", "has_violations": 1, "num_claims": 0, "accident_history_score": 1.5,
            "weather_condition": "clear", "road_type": "urban", "traffic_density": "medium", "trip_purpose": "commute"
        },
        {
            "driver_age": 40, "driver_experience": 20, "vehicle_age": 8, "vehicle_type": "suv",
            "region": "rural", "has_violations": 0, "num_claims": 0, "accident_history_score": 0.0,
            "weather_condition": "foggy", "road_type": "highway", "traffic_density": "low", "trip_purpose": "personal"
        },
        {
            "driver_age": 29, "driver_experience": 7, "vehicle_age": 2, "vehicle_type": "pickup",
            "region": "urban", "has_violations": 1, "num_claims": 1, "accident_history_score": 2.0,
            "weather_condition": "rainy", "road_type": "urban", "traffic_density": "high", "trip_purpose": "commercial"
        },
        {
            "driver_age": 33, "driver_experience": 12, "vehicle_age": 6, "vehicle_type": "crossover",
            "region": "suburban", "has_violations": 0, "num_claims": 0, "accident_history_score": 0.3,
            "weather_condition": "clear", "road_type": "urban", "traffic_density": "medium", "trip_purpose": "personal"
        }
    ])

    probas = []
    tariffs = []

    for _, row in bulk_data.iterrows():
        case_df = pd.DataFrame([row])
        proba = model.predict_proba(case_df)
        tariff = model.calculate_tariff(proba)
        probas.append(proba)
        tariffs.append(tariff)

    bulk_data["probability"] = probas
    bulk_data["tariff"] = tariffs

    print("Прогноз по случаям:")
    for i, row in bulk_data.iterrows():
        print(f"\nСлучай {i+1}:")
        print(f"Вероятность ДТП: {row['probability']*100:.2f}%")
        print(f"Страховой тариф: {row['tariff']:.2f}₽")

    for feature in ["driver_age", "driver_experience", "vehicle_age", "accident_history_score", "num_claims"]:
        plt.figure(figsize=(6, 4))
        plt.scatter(bulk_data[feature], bulk_data["tariff"], c="blue")
        plt.title(f"Зависимость тарифа от {feature}")
        plt.xlabel(feature)
        plt.ylabel("Страховой тариф (₽)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_bulk_cases_with_plot()
