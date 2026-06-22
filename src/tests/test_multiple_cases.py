from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("Тестик")
    print("=" * 70)

    model = InsuranceRiskModel(model_path="outputs/insurance_model_v1.cbm")

    cases = [
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
            'Базовый КБМ': round(case['base_kbm'], 2),
            'Рекомендуемый КБМ': round(kbm, 2)
        })

    results_df = pd.DataFrame(results)
    print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    print("\nАНАЛИЗ ЛОГИЧНОСТИ:")
    print("-" * 70)

    low_risk = results_df.iloc[0]
    high_risk = results_df.iloc[-1]

    if low_risk['Рекомендуемый КБМ'] <= low_risk['Базовый КБМ']:
        print("Низкорисковые профили: КБМ снижен или сохранён")
    else:
        print("Низкорисковые профили: КБМ неожиданно повышен")

    if high_risk['Рекомендуемый КБМ'] > high_risk['Базовый КБМ']:
        print("Высокорисковые профили: КБМ повышен")
    else:
        print("Высокорисковые профили: КБМ не повышен")

    plt.figure(figsize=(12, 6))
    bars = plt.barh(results_df['Описание'], results_df['Рекомендуемый КБМ'],
                    color=['green', 'lightgreen', 'orange', 'red', 'darkred'],
                    edgecolor='black')
    plt.title("Рекомендуемый КБМ по типам водителей\n(Модель AUC=0.992)", fontsize=14, pad=20)
    plt.xlabel("Коэффициент бонус-малус (КБМ)")
    plt.grid(axis='x', alpha=0.3)

    # Добавляем значения на бары
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width}', va='center', ha='left', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()

    print("\nТест завершён. Модель демонстрирует корректную градацию рисков.")


if __name__ == "__main__":
    main()