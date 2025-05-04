import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


def train_model(df: pd.DataFrame, target_column: str):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)

    return model, X_test, y_test
