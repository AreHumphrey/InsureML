# src/train/train_dtc_model.py
from src.features.dtc_processor import load_and_process_dtc_data
from src.models.lightgbm.dtc_risk_model import DTCKBMModel
import pandas as pd
import os


def main():
    # Пути к данным
    file_paths = [
        "src/data/raw/allcars.csv",
        "src/data/raw/v2.csv"
    ]

    all_features = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ Файл не найден: {file_path}")
            continue

        features = load_and_process_dtc_data(file_path)
        all_features.append(features)

    if not all_features:
        print("❌ Нет данных для обучения.")
        return

    # Объединяем все данные
    combined_features = pd.concat(all_features, ignore_index=True)
    print(f"✅ Объединено {len(combined_features)} поездок.")

    # Создание целевой переменной
    y = combined_features['has_dtc_errors'].astype(int)
    X = combined_features.drop(columns=['tripID', 'has_dtc_errors'], errors='ignore')

    # Явное указание типов
    X['duration_sec'] = X['duration_sec'].astype(float)
    X['avg_speed'] = X['avg_speed'].astype(float)

    # Обучение модели
    model = DTCKBMModel()
    model.train(X, y)

    # Сохранение
    model_dir = "outputs/lightgbm"
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/dtc_risk_model_v1.pkl")

    print("✅ Модель DTC-Based Risk Model обучена и сохранена!")
    print(f"   → Признаки: {list(X.columns)}")
    print(f"   → Размер выборки: {len(X)}")


if __name__ == "__main__":
    main()