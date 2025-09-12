import lightgbm as lgb
import joblib
import pandas as pd
import os

class TelematicsRiskModel:
    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100
            )

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict_risk(self, X: pd.DataFrame) -> float:
        return float(self.model.predict(X)[0])

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)