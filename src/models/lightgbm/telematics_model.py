# src/models/lightgbm/telematics_model.py
import lightgbm as lgb
import joblib
import os


class TelematicsRiskModel:
    def __init__(self, model_path: str = None):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=10,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=20,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            force_col_wise=True,
            random_state=42
        )

        if model_path and os.path.exists(model_path):
            print(f"📥 Загружаю модель из {model_path}")
            self.model = joblib.load(model_path)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_risk(self, X):
        return float(self.model.predict_proba(X)[:, 1])

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Модель сохранена: {path}")