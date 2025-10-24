# src/tests/test_dtc_model.py
from src.models.lightgbm.dtc_risk_model import DTCKBMModel
import pandas as pd


def main():
    # Примеры поездок
    safe_trip = pd.DataFrame([{
        'duration_sec': 1800.0,
        'avg_speed': 45.0,
        'has_dtc_errors': False
    }])

    risky_trip = pd.DataFrame([{
        'duration_sec': 2700.0,
        'avg_speed': 65.0,
        'has_dtc_errors': True
    }])

    # Загрузка модели
    model = DTCKBMModel(model_path="outputs/lightgbm/dtc_risk_model_v1.pkl")
    print("✅ Модель DTC-риска загружена")

    # Предсказания
    risk_safe = model.predict_risk(safe_trip)
    print(f"\n🎯 Спокойная поездка (нет DTC) → риск: {risk_safe:.2%}")

    risk_risky = model.predict_risk(risky_trip)
    print(f"🎯 Агрессивная поездка (есть DTC) → риск: {risk_risky:.2%}")

    # Логика
    if risk_risky > risk_safe:
        print("\n✅ МОДЕЛЬ КОРРЕКТНО ОЦЕНИВАЕТ РИСК ПО НАЛИЧИЮ DTC")
        print(f"   Разница в риске: {risk_risky - risk_safe:.2%}")
    else:
        print("\n⚠️  Модель требует дообучения — не различает наличие DTC")


if __name__ == "__main__":
    main()