# src/tests/test_telematics.py
from src.models.lightgbm.telematics_model import TelematicsRiskModel
import pandas as pd

def print_trip_summary(trip_df: pd.DataFrame, title: str):
    """
    Выводит подробную сводку по поездке.
    """
    row = trip_df.iloc[0]
    print(f"\n{'='*60}")
    print(f"📌 {title.upper()}")
    print(f"{'='*60}")
    print(f"🚗 Общие параметры:")
    print(f"   • Средняя скорость:      {row['avg_speed']:>6.1f} км/ч")
    print(f"   • Максимальная скорость: {row['max_speed']:>6.1f} км/ч")
    print(f"   • Колебания скорости:    {row['std_speed']:>6.1f} км/ч")
    print(f"   • Пробег:                {row['distance_km']:>6.1f} км")
    print(f"   • Длительность:          {row['trip_duration_min']:>6.0f} мин")

    print(f"\n🌙 Вождение в темное время:")
    print(f"   • Доля ночных поездок:    {row['night_driving_ratio']:.0%}")

    print(f"\n🔧 Техническое состояние:")
    print(f"   • Есть ошибки DTC:        {'Да' if row['has_dtc_errors'] else 'Нет'}")
    print(f"   • Ср. температура ОЖ:     {row['avg_coolant_temp']:>6.0f} °C")
    print(f"   • Ср. температура на впуске: {row['avg_iat']:>6.0f} °C")

def main():
    # Пример "спокойной" поездки
    safe_trip = pd.DataFrame([{
        'avg_speed': 45.0,
        'max_speed': 80.0,
        'std_speed': 8.0,
        'night_driving_ratio': 0.05,
        'trip_duration_min': 30,
        'distance_km': 25,
        'has_dtc_errors': False,
        'avg_coolant_temp': 90,
        'avg_iat': 25
    }])

    # Пример "агрессивной" поездки
    risky_trip = pd.DataFrame([{
        'avg_speed': 65.0,
        'max_speed': 120.0,
        'std_speed': 22.0,
        'night_driving_ratio': 0.7,
        'trip_duration_min': 45,
        'distance_km': 60,
        'has_dtc_errors': True,
        'avg_coolant_temp': 98,
        'avg_iat': 30
    }])

    # Загрузка модели
    model = TelematicsRiskModel(model_path="outputs/lightgbm/telematics_model_v1.pkl")
    print("✅ Модель телематики загружена")

    # Сводка по спокойной поездке
    print_trip_summary(safe_trip, "Спокойная поездка")
    risk_safe = model.predict_risk(safe_trip)
    print(f"\n🎯 Прогноз модели:")
    print(f"   → Вероятность ДТП: {risk_safe:.2%}")

    # Сводка по агрессивной поездке
    print_trip_summary(risky_trip, "Агрессивная поездка")
    risk_risky = model.predict_risk(risky_trip)
    print(f"\n🎯 Прогноз модели:")
    print(f"   → Вероятность ДТП: {risk_risky:.2%}")

    # Итоговое сравнение
    print("\n" + "="*60)
    if risk_risky > risk_safe:
        print("✅ МОДЕЛЬ КОРРЕКТНО ОЦЕНИВАЕТ ПОВЕДЕНИЕ")
        print(f"   Разница в риске: {risk_risky - risk_safe:.2%}")
    else:
        print("⚠️  Модель не различает уровни риска — требуется дообучение")

if __name__ == "__main__":
    main()