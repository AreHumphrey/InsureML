import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from src.data.load_data import load_dataset
from src.data.preprocess_data import preprocess

def main():
    df = load_dataset("data/raw/insurance_data.csv")
    df_clean = preprocess(df)
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    categorical_features = [
        "vehicle_type", "region", "weather_condition", "road_type", "traffic_density", "trip_purpose"
    ]

    model = CatBoostClassifier(verbose=0)
    pool = Pool(X, y, cat_features=categorical_features)
    model.fit(pool)

    importances = model.get_feature_importance()
    features = X.columns

    feature_names_rus = {
        "driver_age": "Возраст водителя",
        "driver_experience": "Стаж водителя",
        "vehicle_age": "Возраст ТС",
        "vehicle_type": "Тип ТС",
        "region": "Регион",
        "has_violations": "Нарушения ПДД",
        "num_claims": "Количество заявлений",
        "accident_history_score": "Индекс аварийности",
        "weather_condition": "Погодные условия",
        "road_type": "Тип дороги",
        "traffic_density": "Плотность трафика",
        "trip_purpose": "Цель поездки"
    }

    translated_features = [feature_names_rus.get(f, f) for f in features]

    plt.figure(figsize=(10, 6))
    plt.barh(translated_features, importances)
    plt.xlabel("Важность признака")
    plt.title("Влияние признаков на вероятность ДТП")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_ru.png")
    plt.show()

if __name__ == "__main__":
    main()
