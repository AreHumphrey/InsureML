

from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
import numpy as np


def load_and_analyze(model_path="outputs/insurance_model_v1.cbm"):

    model = InsuranceRiskModel(model_path=model_path)


    # Загрузка данных
    try:
        df = pd.read_csv("src/data/raw/insurance_data.csv")
    except FileNotFoundError:
        print("❌ Файл данных не найден. Используем тестовые данные.")
        df = pd.DataFrame([{
            'driver_age': 30,
            'driver_experience': 8,
            'vehicle_age': 3,
            'vehicle_type': 'sedan',
            'engine_power': 150,
            'vehicle_purpose': 'personal',
            'region': 'Moscow',
            'pct_days_with_snow': 0.35,
            'pct_days_with_rain': 0.45,
            'winter_duration_months': 5,
            'base_kbm': 1.0,
            'num_claims': 1,
            'violation_count': 1,
            'days_since_last_claim': 365,
            'occupation_type': 'office_worker',
            'avg_trips_per_week': 8,
            'night_driving_ratio': 0.1,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'target': 0
        }, {
            'driver_age': 25,
            'driver_experience': 3,
            'vehicle_age': 1,
            'vehicle_type': 'hatchback',
            'engine_power': 105,
            'vehicle_purpose': 'personal',
            'region': 'urban',
            'pct_days_with_snow': 0.3,
            'pct_days_with_rain': 0.4,
            'winter_duration_months': 4,
            'base_kbm': 1.0,
            'num_claims': 0,
            'violation_count': 2,
            'days_since_last_claim': 100,
            'occupation_type': 'student',
            'avg_trips_per_week': 6,
            'night_driving_ratio': 0.3,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1,
            'target': 1
        }])

    # Предобработка через единый pipeline
    X = df.drop(columns=["target"])
    y = df["target"]

    # Используем preprocess из модели — он уже включает генерацию признаков
    X_processed = model.preprocess(X)

    # Предсказания
    y_pred_proba = model.model.predict_proba(X_processed)[:, 1]

    # Используем порог из модели, если он установлен, иначе 0.5
    threshold = getattr(model, 'threshold', 0.5)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Визуализация
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_pred_proba)
    print_metrics(y, y_pred_proba, threshold=threshold)
    plot_feature_importance(model)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Нет ДТП', 'ДТП'],
                yticklabels=['Нет ДТП', 'ДТП'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xlabel('Предсказано')
    plt.ylabel('Факт')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Случайная модель')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print("\n" + "=" * 60)
    print("МЕТРИКИ МОДЕЛИ")
    print("=" * 60)
    print(f"Порог классификации: {threshold:.3f}")
    print("-" * 60)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC:          {auc:.4f}")
    print("=" * 60)


def plot_feature_importance(model):
    try:
        feat_imp = model.model.get_feature_importance()
        feature_names = model.model.feature_names_
    except Exception as e:
        print(f"⚠️ Не удалось получить важность признаков: {e}")
        return

    if len(feat_imp) != len(feature_names):
        print("⚠️ Несоответствие длины признаков и важности")
        return

    # Сортировка по убыванию
    indices = np.argsort(feat_imp)[::-1]
    top_n = min(15, len(feature_names))  # Показываем топ-15
    indices = indices[:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feat_imp[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()  # Самый важный — сверху
    plt.title('Важность признаков (Top 15)')
    plt.xlabel('Важность (Feature Importance)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_analyze()