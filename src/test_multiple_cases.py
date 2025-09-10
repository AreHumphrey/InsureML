
import pandas as pd
from src.models.insurance_model import InsuranceRiskModel
import matplotlib.pyplot as plt

def main():

    model = InsuranceRiskModel(model_path="outputs/insurance_model_v1.pkl")

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
        },
        {
            'driver_age': 24,
            'driver_experience': 3,
            'vehicle_age': 1,
            'vehicle_type': 'hatchback',
            'engine_power': 105,
            'vehicle_purpose': 'personal',
            'region': 'urban',
            'pct_days_with_snow': 0.3,
            'pct_days_with_rain': 0.4,
            'winter_duration_months': 4,
            'base_kbm': 1.0,
            'num_claims': 0,
            'violation_count': 2,
            'days_since_last_claim': 100,
            'occupation_type': 'student',
            'avg_trips_per_week': 6,
            'night_driving_ratio': 0.3,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Молодой водитель с нарушениями'
        },
        {
            'driver_age': 35,
            'driver_experience': 12,
            'vehicle_age': 2,
            'vehicle_type': 'van',
            'engine_power': 160,
            'vehicle_purpose': 'commercial',
            'region': 'Moscow',
            'pct_days_with_snow': 0.35,
            'pct_days_with_rain': 0.45,
            'winter_duration_months': 5,
            'base_kbm': 1.0,
            'num_claims': 1,
            'violation_count': 0,
            'days_since_last_claim': 600,
            'occupation_type': 'courier',
            'avg_trips_per_week': 20,
            'night_driving_ratio': 0.4,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'description': 'Курьер (высокая нагрузка)'
        },
        {
            'driver_age': 52,
            'driver_experience': 30,
            'vehicle_age': 7,
            'vehicle_type': 'crossover',
            'engine_power': 200,
            'vehicle_purpose': 'personal',
            'region': 'StPetersburg',
            'pct_days_with_snow': 0.4,
            'pct_days_with_rain': 0.35,
            'winter_duration_months': 5,
            'base_kbm': 0.85,
            'num_claims': 0,
            'violation_count': 0,
            'days_since_last_claim': 1095,
            'occupation_type': 'office_worker',
            'avg_trips_per_week': 4,
            'night_driving_ratio': 0.05,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 2,
            'description': 'Опытный водитель (низкий риск)'
        }
    ]

    results = []
    for case in cases:
        data = {k: [v] for k, v in case.items() if k != 'description'}
        df = pd.DataFrame(data)

        proba = model.predict_proba(df)
        kbm = model.calculate_adjusted_kbm(df, base_kbm=case['base_kbm'])

        results.append({
            'Описание': case['description'],
            'Вероятность ДТП': f"{proba:.2%}",
            'Рекомендуемый КБМ': round(kbm, 2)
        })

    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    plt.figure(figsize=(10, 5))
    plt.barh(results_df['Описание'], results_df['Рекомендуемый КБМ'], color='skyblue', edgecolor='black')
    plt.title("Скорректированный КБМ по разным типам водителей")
    plt.xlabel("Коэффициент бонус-малус (КБМ)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()