from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
import numpy as np
import os

FEATURE_GROUPS = {
    'Персональные данные водителя': [
        'driver_age', 'driver_experience', 'occupation_type', 'num_claims',
        'violation_count', 'days_since_last_claim', 'avg_trips_per_week', 'night_driving_ratio'
    ],
    'Характеристики ТС': [
        'vehicle_age', 'vehicle_type', 'engine_power', 'vehicle_purpose', 'num_owned_vehicles'
    ],
    'Внешние и региональные факторы': [
        'region', 'pct_days_with_snow', 'pct_days_with_rain', 'winter_duration_months'
    ],
    'Базовые страховые коэффициенты': [
        'base_kbm', 'ko_multiplier'
    ],
    'Производные признаки': [
        'age_squared', 'experience_ratio', 'claims_per_year', 'violations_per_year',
        'no_claims_long', 'experienced_clean', 'young_risky', 'high_claims',
        'night_x_trips', 'power_x_age'
    ]
}

FEATURE_NAMES_RU = {

    'driver_age': 'Возраст водителя',
    'driver_experience': 'Водительский стаж',
    'occupation_type': 'Тип занятости',
    'num_claims': 'Количество страховых обращений',
    'violation_count': 'Количество нарушений ПДД',
    'days_since_last_claim': 'Дней с последнего страхового случая',
    'avg_trips_per_week': 'Среднее число поездок в неделю',
    'night_driving_ratio': 'Доля ночных поездок',
    
    'vehicle_age': 'Возраст транспортного средства',
    'vehicle_type': 'Тип транспортного средства',
    'engine_power': 'Мощность двигателя',
    'vehicle_purpose': 'Целевое назначение ТС',
    'num_owned_vehicles': 'Количество ТС в собственности',
    
    'region': 'Регион эксплуатации',
    'pct_days_with_snow': 'Доля дней со снегом',
    'pct_days_with_rain': 'Доля дней с осадками',
    'winter_duration_months': 'Продолжительность зимы (мес.)',

    'base_kbm': 'Базовый КБМ',
    'ko_multiplier': 'Коэффициент ограниченного использования',

    'age_squared': 'Возраст²',
    'experience_ratio': 'Отношение стажа к возрасту',
    'claims_per_year': 'Обращений в год',
    'violations_per_year': 'Нарушений в год',
    'no_claims_long': 'Длительная безаварийная езда',
    'experienced_clean': 'Опытный аккуратный водитель',
    'young_risky': 'Молодой рискованный водитель',
    'high_claims': 'Частые страховые обращения',
    'night_x_trips': 'Ночные поездки × частота',
    'power_x_age': 'Мощность × возраст ТС'
}


def load_robust_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        return df


def load_and_analyze(model_path="outputs/insurance_model_v1.cbm"):
    model = InsuranceRiskModel(model_path=model_path)

    os.makedirs("pictures", exist_ok=True)

    try:
        df = load_robust_csv("src/data/raw/insurance_data.csv")
        df = df.dropna(subset=['driver_age', 'driver_experience', 'num_claims'])
        if 'target' not in df.columns:
            df['target'] = (df['num_claims'] > 0).astype(int)
        else:
            df = df.dropna(subset=['target'])
        df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    except FileNotFoundError:
        print("Файла нет, используем тестовые данные")
        df = pd.DataFrame([{
            'driver_age': 30, 'driver_experience': 8, 'vehicle_age': 3,
            'vehicle_type': 'sedan', 'engine_power': 150, 'vehicle_purpose': 'personal',
            'region': 'Moscow', 'pct_days_with_snow': 0.35, 'pct_days_with_rain': 0.45,
            'winter_duration_months': 5, 'base_kbm': 1.0, 'num_claims': 1,
            'violation_count': 1, 'days_since_last_claim': 365,
            'occupation_type': 'office_worker', 'avg_trips_per_week': 8,
            'night_driving_ratio': 0.1, 'ko_multiplier': 1.0, 'num_owned_vehicles': 1, 'target': 1
        }])

    if len(df) == 0:
        raise ValueError("Нет данных для анализа")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_processed = model.preprocess(X)
    y_pred_proba = model.model.predict_proba(X_processed)[:, 1]
    threshold = getattr(model, 'threshold', 0.5)
    y_pred = (y_pred_proba >= threshold).astype(int)

    plot_confusion_matrix(y, y_pred, save_path="pictures/confusion_matrix.png")
    plot_roc_curve(y, y_pred_proba, save_path="pictures/roc_curve.png")
    print_metrics(y, y_pred_proba, threshold=threshold)
    plot_feature_importance(model, save_path="pictures/feature_importance.png")
    plot_feature_importance_russian(model, save_path="pictures/feature_importance_ru.png")
    plot_detailed_feature_importance(model, save_path="pictures/importance_detailed.png")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Нет ДТП', 'ДТП'],
                yticklabels=['Нет ДТП', 'ДТП'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xlabel('Предсказано')
    plt.ylabel('Факт')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Матрица ошибок сохранена: {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC-кривая сохранена: {save_path}")
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


def plot_feature_importance(model, save_path=None):

    try:
        feat_imp = model.model.get_feature_importance()
        feature_names = model.model.feature_names_
    except Exception as e:
        print(f"Не удалось получить важность признаков: {e}")
        return

    if len(feat_imp) != len(feature_names):
        print("Несоответствие длины признаков и важности")
        return

    indices = np.argsort(feat_imp)[::-1]
    top_n = min(15, len(feature_names))
    indices = indices[:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feat_imp[indices], color='#00D46A', edgecolor='#00664E')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.title('Важность признаков (Top-15)', fontsize=14, fontweight='bold')
    plt.xlabel('Важность (Feature Importance)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Важность признаков сохранена: {save_path}")
    plt.show()


def plot_feature_importance_russian(model, save_path=None):

    try:
        feat_imp = model.model.get_feature_importance()
        feature_names = model.model.feature_names_
    except Exception as e:
        print(f"Не удалось получить важность признаков: {e}")
        return

    if len(feat_imp) != len(feature_names):
        print("Несоответствие длины признаков и важности")
        return

    indices = np.argsort(feat_imp)[::-1]
    top_n = min(15, len(feature_names))
    indices = indices[:top_n]

    ru_names = [FEATURE_NAMES_RU.get(feature_names[i], feature_names[i]) 
                for i in indices]

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(indices)), feat_imp[indices], 
             color='#00D46A', edgecolor='#00664E', linewidth=1.5)
    plt.yticks(range(len(indices)), ru_names, fontsize=11)
    plt.gca().invert_yaxis()
    plt.title('Важность признаков (Топ-15)', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Важность (Feature Importance)', fontsize=13)
    plt.grid(axis='x', alpha=0.3, linestyle='--')

    for i, v in enumerate(feat_imp[indices]):
        plt.text(v + 0.5, i, f'{v:.2f}', 
                 va='center', fontsize=10, fontweight='bold',
                 color='#00664E')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График с русскими названиями сохранён: {save_path}")
    plt.show()


def plot_detailed_feature_importance(model, save_path=None):

    try:
        feat_imp = model.model.get_feature_importance()
        feature_names = list(model.model.feature_names_)
    except Exception as e:
        print(f"Не удалось получить важность признаков: {e}")
        return

    if len(feat_imp) != len(feature_names):
        print("Несоответствие длины признаков и важности")
        return

    importance_dict = dict(zip(feature_names, feat_imp))

    group_importances = {}
    group_colors = {
        'Персональные данные водителя': '#2196F3',      
        'Характеристики ТС': '#FF9800',                
        'Внешние и региональные факторы': '#4CAF50',  
        'Базовые страховые коэффициенты': '#9C27B0',     
        'Производные признаки': '#F44336'                
    }

    for group_name, features in FEATURE_GROUPS.items():
        total = 0
        found = 0
        for feat in features:
            if feat in importance_dict:
                total += importance_dict[feat]
                found += 1
        if found > 0:
            group_importances[group_name] = total

    total_importance = sum(group_importances.values())
    if total_importance == 0:
        print("Суммарная важность равна нулю")
        return

    group_percentages = {k: (v / total_importance) * 100 for k, v in group_importances.items()}

    sorted_groups = sorted(group_percentages.items(), key=lambda x: x[1], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [2, 1]})

    group_names = [g[0] for g in sorted_groups]
    group_values = [g[1] for g in sorted_groups]
    colors = [group_colors.get(g, '#888888') for g in group_names]

    bars = ax1.barh(range(len(group_names)), group_values, color=colors, edgecolor='black', linewidth=0.5)

    for i, (bar, name) in enumerate(zip(bars, group_names)):
        width = bar.get_width()
        abs_val = group_importances.get(name, 0)
        ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}% ({abs_val:.2f})',
                 va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_yticks(range(len(group_names)))
    ax1.set_yticklabels(group_names, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlabel('Доля в общей важности (%)', fontsize=12)
    ax1.set_title('Вклад групп признаков в прогноз модели', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, max(group_values) * 1.4)

    labels = [g[0] for g in sorted_groups]
    sizes = [g[1] for g in sorted_groups]
    pie_colors = [group_colors.get(l, '#888888') for l in labels]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=None,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax2.set_title('Распределение важности\nпо категориям', fontsize=13, fontweight='bold')

    ax2.legend(wedges, [f'{l} ({s:.1f}%)' for l, s in zip(labels, sizes)],
               loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=1, fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Детальная важность признаков сохранена: {save_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("ВАЖНОСТЬ ПРИЗНАКОВ ПО ГРУППАМ")
    print("=" * 70)
    print(f"{'Группа':<40} {'Важность':>10} {'Доля (%)':>10}")
    print("-" * 70)
    for name, pct in sorted_groups:
        abs_val = group_importances.get(name, 0)
        print(f"{name:<40} {abs_val:>10.4f} {pct:>9.1f}%")
    print("-" * 70)
    print(f"{'ИТОГО':<40} {total_importance:>10.4f} {100.0:>9.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    load_and_analyze()