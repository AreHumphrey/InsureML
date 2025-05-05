import os

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib


class InsuranceRiskModel:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = CatBoostClassifier(verbose=0)

        self.required_features = [
            'driver_age', 'driver_experience', 'vehicle_age', 'vehicle_type',
            'region', 'has_violations', 'num_claims', 'accident_history_score',
            'weather_condition', 'road_type', 'traffic_density', 'trip_purpose'
        ]

    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        df = input_data.copy()
        for col in self.required_features:
            if col not in df.columns:
                df[col] = np.nan
        df.fillna(0, inplace=True)
        return df[self.required_features]

    def train(self, data: pd.DataFrame, labels: pd.Series):
        processed = self.preprocess(data)
        cat_features = [
            "vehicle_type", "region", "weather_condition", "road_type", "traffic_density", "trip_purpose"
        ]
        self.model.fit(processed, labels, cat_features=cat_features)

    def predict_proba(self, case: pd.DataFrame) -> float:
        processed = self.preprocess(case)
        proba = self.model.predict_proba(processed)[0][1]
        return proba

    def calculate_tariff(self, proba: float, base_tariff: float = 10000, alpha: float = 1.5) -> float:
        k_ml = 1 + alpha * proba
        return round(base_tariff * k_ml, 2)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
