from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_robust_csv(file_path: str) -> pd.DataFrame:

    try:

        df = pd.read_csv(file_path)
        print(f"Успешно загружено {len(df)} строк стандартным способом")
        return df
    except pd.errors.ParserError as e:
        print(f" Обнаружены ошибки парсинга: {e}")
        print("Повторная попытка с пропуском проблемных строк...")

        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            print(f"Загружено {len(df)} строк после пропуска битых")
            return df
        except Exception as e2:
            print(f"Не удалось загрузить даже с пропуском: {e2}")
            raise


def main():
    try:
        df = load_robust_csv("src/data/raw/insurance_data.csv")
    except FileNotFoundError:
        print(" Файл данных не найден. Используем тестовые данные.")
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
            'target': 1
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

    if 'target' not in df.columns:
        print("Колонка 'target' отсутствует. Создаём на основе num_claims...")
        df['target'] = (df['num_claims'] > 0).astype(int)

    required_cols = [
        'driver_age', 'driver_experience', 'vehicle_age', 'vehicle_type',
        'engine_power', 'vehicle_purpose', 'region', 'pct_days_with_snow',
        'pct_days_with_rain', 'winter_duration_months', 'base_kbm',
        'num_claims', 'violation_count', 'days_since_last_claim',
        'occupation_type', 'avg_trips_per_week', 'night_driving_ratio',
        'ko_multiplier', 'num_owned_vehicles'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Отсутствуют колонки: {missing_cols}")
        raise ValueError(f"Не хватает колонок: {missing_cols}")

    initial_len = len(df)
    df = df.dropna(subset=['driver_age', 'driver_experience', 'num_claims', 'target'])
    final_len = len(df)
    if initial_len != final_len:
        print(f" Удалено {initial_len - final_len} строк с пропущенными ключевыми значениями")

    X = df.drop(columns=["target"])
    y = df["target"]

    if len(X) == 0:
        raise ValueError("После очистки не осталось данных для обучения!")

    model = InsuranceRiskModel()
    model.train(X, y)

    model.save_model("outputs/insurance_model_v1.cbm")

    X_processed = model.preprocess(X)
    y_pred = model.model.predict(X_processed)
    y_proba = model.model.predict_proba(X_processed)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba)

    print("\n" + "=" * 60)
    print("МЕТРИКИ НА ОБУЧЕНИИ")
    print("=" * 60)
    print(f"Использовано строк: {len(X)}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC:          {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()