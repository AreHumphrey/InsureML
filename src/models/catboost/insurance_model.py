import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

pd.set_option('future.no_silent_downcasting', True)


class InsuranceRiskModel:
    def __init__(self, model_path: str = None, class_weights=None):
        if model_path and os.path.exists(model_path):
            if model_path.endswith('.cbm'):
                self.model = CatBoostClassifier()
                self.model.load_model(model_path)
            else:
                self.model = joblib.load(model_path)
        else:
            self.model = CatBoostClassifier(
                verbose=100,
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                eval_metric='AUC',
                early_stopping_rounds=50,
                random_seed=42,
                class_weights=class_weights
            )

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

        self.threshold = 0.5

    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        df = input_data.copy()

        for col in self.required_features:
            if col not in df.columns:
                df[col] = np.nan


        from src.features.feature_engineering import generate_features
        df = generate_features(df)

        num_features = [
            'driver_age', 'driver_experience', 'vehicle_age', 'engine_power',
            'pct_days_with_snow', 'pct_days_with_rain', 'winter_duration_months',
            'base_kbm', 'num_claims', 'violation_count', 'days_since_last_claim',
            'avg_trips_per_week', 'night_driving_ratio', 'ko_multiplier', 'num_owned_vehicles',
            'age_squared',
            'experience_ratio',
            'claims_per_year',
            'violations_per_year',
            'claims_recent',
            'high_night_driving',
            'young_inexperienced',
            'power_to_age',
            'claim_violation_ratio'
        ]

        for col in num_features:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        all_features = self.required_features + [
            'age_squared',
            'experience_ratio',
            'claims_per_year',
            'violations_per_year',
            'claims_recent',
            'high_night_driving',
            'young_inexperienced',
            'power_to_age',
            'claim_violation_ratio'
        ]

        for col in all_features:
            if col not in df.columns:
                df[col] = 0

        return df[all_features]

    def train(self, data: pd.DataFrame, labels: pd.Series):
        processed = self.preprocess(data)

        X_train, X_val, y_train, y_val = train_test_split(
            processed, labels, test_size=0.2, stratify=labels, random_state=42
        )

        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))

        self.model = CatBoostClassifier(
            verbose=100,
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            early_stopping_rounds=50,
            random_seed=42,
            class_weights=weight_dict
        )

        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=self.cat_features,
            verbose=100
        )

        from sklearn.metrics import f1_score
        y_proba = self.model.predict_proba(X_val)[:, 1]
        best_threshold = 0.5
        best_f1 = 0
        for th in np.arange(0.05, 0.5, 0.01):
            y_pred = (y_proba >= th).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th

        self.threshold = best_threshold
        print(f"Оптимальный порог: {self.threshold:.3f}, F1: {best_f1:.4f}")

    def predict_proba(self, case: pd.DataFrame) -> float:
        processed = self.preprocess(case)
        proba = self.model.predict_proba(processed)[0][1]
        return float(proba)

    def predict(self, case: pd.DataFrame) -> int:
        proba = self.predict_proba(case)
        return 1 if proba >= self.threshold else 0

    def calculate_adjusted_kbm(self, case: pd.DataFrame, base_kbm: float = 1.0,
                               avg_proba: float = 0.3, beta: float = 1.5) -> float:
        proba = self.predict_proba(case)
        kbm_adjusted = base_kbm * (1 + beta * (proba - avg_proba))
        kbm_final = max(0.46, min(3.92, kbm_adjusted))
        return round(kbm_final, 2)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith('.cbm'):
            self.model.save_model(path)
        else:
            joblib.dump(self.model, path)