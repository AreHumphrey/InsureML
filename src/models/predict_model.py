import pandas as pd
from catboost import CatBoostClassifier

def predict(model: CatBoostClassifier, input_data: pd.DataFrame) -> pd.Series:

    return model.predict_proba(input_data)[:, 1]  # вероятность класса "1"