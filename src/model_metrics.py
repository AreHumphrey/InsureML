import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from catboost import CatBoostClassifier, Pool
from src.data.load_data import load_dataset
from src.data.preprocess_data import preprocess

def evaluate_model():
    df = load_dataset("data/raw/insurance_data.csv")
    df_clean = preprocess(df)
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    categorical_features = [
        "vehicle_type", "region", "weather_condition", "road_type", "traffic_density", "trip_purpose"
    ]

    model = CatBoostClassifier(verbose=0)
    model.fit(Pool(X, y, cat_features=categorical_features))

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("📌 Классификационный отчёт:")
    print(classification_report(y, y_pred, digits=3))

    print("📌 Матрица ошибок:")
    print(confusion_matrix(y, y_pred))

    roc_auc = roc_auc_score(y, y_proba)
    print(f"📈 ROC-AUC: {roc_auc:.3f}")

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'CatBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая модели")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
