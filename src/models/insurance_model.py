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

        self.cat_features = [
            "vehicle_type", "region", "weather_condition", "road_type", "traffic_density", "trip_purpose"
        ]

    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        df = input_data.copy()
        for col in self.required_features:
            if col not in df.columns:
                df[col] = np.nan

        num_features = ['driver_age', 'driver_experience', 'vehicle_age', 'accident_history_score']
        for col in num_features:
            if df[col].isnull().all():
                df[col] = 0
            else:
                df[col].fillna(df[col].median(), inplace=True)

        for col in self.cat_features:
            if col in df.columns:
                df[col].fillna("unknown", inplace=True)

        return df[self.required_features]

    def train(self, data: pd.DataFrame, labels: pd.Series):
        processed = self.preprocess(data)
        self.model.fit(processed, labels, cat_features=self.cat_features)

    def predict_proba(self, case: pd.DataFrame) -> float:
        processed = self.preprocess(case)
        proba = self.model.predict_proba(processed)[0][1]
        return proba

    def calculate_adjusted_kbm(self,
                               case: pd.DataFrame,
                               base_kbm: float = 1.0,
                               avg_proba: float = 0.3,
                               beta: float = 1.5) -> float:

        proba = self.predict_proba(case)
        kbm_adjusted = base_kbm * (1 + beta * (proba - avg_proba))

        kbm_final = max(0.46, min(3.92, kbm_adjusted))
        return round(kbm_final, 2)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)