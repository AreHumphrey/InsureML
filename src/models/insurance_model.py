# src/models/insurance_model.py
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

# Решение для FutureWarning: Downcasting
pd.set_option('future.no_silent_downcasting', True)


class InsuranceRiskModel:
    def __init__(self, model_path: str = None):
        """
        Инициализация модели.
        :param model_path: путь к сохранённой модели
        """
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = CatBoostClassifier(verbose=0)

        self.required_features = [
            'driver_age',
            'driver_experience',
            'vehicle_age',
            'vehicle_type',
            'engine_power',
            'vehicle_purpose',
            'region',
            'pct_days_with_snow',
            'pct_days_with_rain',
            'winter_duration_months',
            'base_kbm',
            'num_claims',
            'violation_count',
            'days_since_last_claim',
            'occupation_type',
            'avg_trips_per_week',
            'night_driving_ratio',
            'ko_multiplier',
            'num_owned_vehicles'
        ]

        self.cat_features = [
            "vehicle_type",
            "region",
            "vehicle_purpose",
            "occupation_type"
        ]

    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка входных данных.
        :param input_data: DataFrame с данными водителя
        :return: очищенный DataFrame
        """
        df = input_data.copy()

        # Добавляем пропущенные колонки как NaN
        for col in self.required_features:
            if col not in df.columns:
                df[col] = np.nan

        # Числовые признаки
        num_features = [
            'driver_age', 'driver_experience', 'vehicle_age', 'engine_power',
            'pct_days_with_snow', 'pct_days_with_rain', 'winter_duration_months',
            'base_kbm', 'num_claims', 'violation_count', 'days_since_last_claim',
            'avg_trips_per_week', 'night_driving_ratio', 'ko_multiplier', 'num_owned_vehicles'
        ]

        for col in num_features:
            if col in df.columns:
                # Заполняем пропуски медианой
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Категориальные признаки
        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Возвращаем только нужные признаки
        return df[self.required_features]

    def train(self, data: pd.DataFrame, labels: pd.Series):
        """
        Обучение модели.
        :param data: признаки
        :param labels: целевая переменная
        """
        processed = self.preprocess(data)
        self.model.fit(processed, labels, cat_features=self.cat_features)

    def predict_proba(self, case: pd.DataFrame) -> float:
        """
        Прогноз вероятности ДТП для одного случая.
        :param case: DataFrame с одной строкой данных
        :return: вероятность ДТП (число от 0 до 1)
        """
        processed = self.preprocess(case)
        proba = self.model.predict_proba(processed)[0][1]  # вероятность класса 1
        return float(proba)

    def calculate_adjusted_kbm(self, case: pd.DataFrame, base_kbm: float = 1.0,
                               avg_proba: float = 0.3, beta: float = 1.5) -> float:
        """
        Расчёт скорректированного КБМ.
        :param case: данные водителя
        :param base_kbm: текущий КБМ
        :param avg_proba: средняя вероятность ДТП по базе
        :param beta: коэффициент чувствительности
        :return: скорректированный КБМ в диапазоне [0.46, 3.92]
        """
        proba = self.predict_proba(case)
        kbm_adjusted = base_kbm * (1 + beta * (proba - avg_proba))
        kbm_final = max(0.46, min(3.92, kbm_adjusted))
        return round(kbm_final, 2)

    def save_model(self, path: str):
        """
        Сохранение модели.
        :param path: путь для сохранения
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)