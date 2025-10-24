# src/models/lightgbm/dtc_risk_model.py
"""
Модель риска на основе DTC.
"""

import lightgbm as lgb
import joblib
import os


class DTCKBMModel:
    def __init__(self, model_path: str = None):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=5,
            max_depth=2,
            learning_rate=0.1,
            n_estimators=10,
            min_child_samples=5,
            force_col_wise=True,
            random_state=42
        )

        if model_path and os.path.exists(model_path):
            print(f"📥 Загружаю модель из {model_path}")
            self.model = joblib.load(model_path)

    def train(self, X, y):
        """Обучает модель."""
        self.model.fit(X, y)

    def predict_risk(self, X):
        """Предсказывает вероятность высокого риска (наличие DTC)."""
        return float(self.model.predict_proba(X)[0, 1])

    def save_model(self, path: str):
        """Сохраняет модель."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Модель сохранена: {path}")