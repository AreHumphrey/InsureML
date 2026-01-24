from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # Загрузка данных
    try:
        df = pd.read_csv("src/data/raw/insurance_data.csv")
    except FileNotFoundError:
        print("❌ Файл данных не найден. Используем тестовые данные.")
        df = pd.DataFrame([{
            'driver_age': 30,
            'driver_experience': 8,
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
            'violation_count': 1,
            'days_since_last_claim': 365,
            'occupation_type': 'office_worker',
            'avg_trips_per_week': 8,
            'night_driving_ratio': 0.1,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'target': 0
        }, {
            'driver_age': 25,
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
            'target': 1
        }])

    X = df.drop(columns=["target"])
    y = df["target"]

    model = InsuranceRiskModel()
    model.train(X, y)

    # Сохраняем в .cbm
    model.save_model("outputs/insurance_model_v1.cbm")
    print("✅ Модель успешно обучена и сохранена как .cbm")

    # Оценка на train (для диагностики)
    X_processed = model.preprocess(X)
    y_pred = model.model.predict(X_processed)
    y_proba = model.model.predict_proba(X_processed)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print("\n" + "="*60)
    print("МЕТРИКИ НА ОБУЧЕНИИ (для диагностики)")
    print("="*60)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC:          {auc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()