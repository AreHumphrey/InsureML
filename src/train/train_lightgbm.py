# src/train/train_lightgbm.py
from src.models.lightgbm.telematics_model import TelematicsRiskModel
from src.features.telemetry_processor import process_telemetry_data
import pandas as pd
import os


def main():
    raw_input_path = "src/data/raw/obd_data.csv"
    processed_output_path = "src/data/processed/telem_features.csv"

    X_raw = process_telemetry_data(
        input_path=raw_input_path,
        output_path=processed_output_path
    )

    if len(X_raw) == 0:
        print("❌ Нет данных для обучения.")
        return

    # Простой риск: число резких тормозов на минуту
    brakes_per_min = X_raw['hard_brakes'] / (X_raw['duration_sec'] / 60)
    y = (brakes_per_min > brakes_per_min.quantile(0.7)).astype(int)  # топ-30% самых агрессивных

    X = X_raw.drop(columns=['tripID'], errors='ignore')

    model = TelematicsRiskModel()
    model.train(X, y)

    model_dir = "src/outputs/lightgbm"
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/telematics_model_v1.pkl")

    print("✅ Модель LightGBM обучена и сохранена!")
    print(f"   → Признаки: {list(X.columns)}")
    print(f"   → Размер выборки: {len(X)}")


if __name__ == "__main__":
    main()