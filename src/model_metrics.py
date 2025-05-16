# src/generate_and_test.py
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

# Создание синтетического датасета
def generate_dummy_data():
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        "driver_age": np.random.randint(18, 70, n),
        "driver_experience": np.random.randint(1, 50, n),
        "vehicle_age": np.random.randint(0, 20, n),
        "vehicle_type": np.random.choice(["sedan", "suv", "truck", "van"], n),
        "region": np.random.choice(["urban", "rural", "suburban"], n),
        "has_violations": np.random.choice([0, 1], n),
        "num_claims": np.random.randint(0, 4, n),
        "accident_history_score": np.random.uniform(0, 10, n),
        "weather_condition": np.random.choice(["clear", "rainy", "foggy", "snow"], n),
        "road_type": np.random.choice(["urban", "highway"], n),
        "traffic_density": np.random.choice(["low", "medium", "high"], n),
        "trip_purpose": np.random.choice(["commute", "personal", "commercial"], n),
        "target": np.random.choice([0, 1], n, p=[0.6, 0.4])
    })
    os.makedirs("data/raw", exist_ok=True)
    data.to_csv("data/raw/insurance_data.csv", index=False)
    print("✅ Синтетический датасет создан и сохранён в data/raw/insurance_data.csv")

# Запуск теста
def evaluate_model():
    df = pd.read_csv("data/raw/insurance_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    cat_features = ["vehicle_type", "region", "weather_condition", "road_type", "traffic_density", "trip_purpose"]

    model = CatBoostClassifier(verbose=0)
    model.fit(Pool(X_train, y_train, cat_features=cat_features))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📌 Классификационный отчёт:\n")
    print(classification_report(y_test, y_pred, digits=3))

    print("\n📌 Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n📈 ROC-AUC: {roc_auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'CatBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая на синтетических данных")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_dummy.png")
    plt.show()

if __name__ == "__main__":
    generate_dummy_data()
    evaluate_model()
