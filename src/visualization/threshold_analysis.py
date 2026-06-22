from src.models.catboost.insurance_model import InsuranceRiskModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_robust_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        return df


def plot_threshold_analysis(model_path="outputs/insurance_model_v1.cbm", 
                            data_path="src/data/raw/insurance_data.csv"):

    print("Загрузка модели...")
    model = InsuranceRiskModel(model_path=model_path)
    
    print("Загрузка данных...")
    try:
        df = load_robust_csv(data_path)
        df = df.dropna(subset=['driver_age', 'driver_experience', 'num_claims'])
        
        if 'target' not in df.columns:
            df['target'] = (df['num_claims'] > 0).astype(int)
        else:
            df = df.dropna(subset=['target'])
        
        df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    except FileNotFoundError:
        print(f"Файл {data_path} не найден. Используется тестовый датасет.")

        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'driver_age': np.random.randint(18, 80, n_samples),
            'driver_experience': np.random.randint(0, 60, n_samples),
            'num_claims': np.random.poisson(0.3, n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)
        })
    
    print(f"Размер выборки: {len(df)}")

    X = df.drop(columns=["target"])
    y = df["target"].values

    X_processed = model.preprocess(X)
    y_pred_proba = model.model.predict_proba(X_processed)[:, 1]
    


    thresholds = np.arange(0.05, 0.51, 0.01)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 0: 
                prec = 0.0
                rec = 0.0
                f1 = 0.0
            else: 
                prec = np.sum(y == 1) / len(y)
                rec = 1.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
        
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

    plt.figure(figsize=(12, 7), facecolor='#0A1A15')
    ax = plt.gca()
    ax.set_facecolor('#0F2319')
    
    line_precision, = plt.plot(thresholds, precision_scores, 
                               color='#00D46A', linewidth=2.5, 
                               label='Precision (Точность)', marker='o', 
                               markersize=4, alpha=0.8)
    line_recall, = plt.plot(thresholds, recall_scores, 
                           color='#00A55E', linewidth=2.5, 
                           label='Recall (Полнота)', marker='s', 
                           markersize=4, alpha=0.8)
    line_f1, = plt.plot(thresholds, f1_scores, 
                       color='#05DC7F', linewidth=2.5, 
                       label='F1-Score', marker='^', 
                       markersize=4, alpha=0.8)
    

    optimal_threshold = 0.30
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    
    plt.axvline(x=optimal_threshold, color='#FFD700', 
                linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Оптимальный порог θ* = {optimal_threshold:.2f}')
    

    plt.scatter([optimal_threshold], [f1_scores[optimal_idx]], 
               color='#FFD700', s=150, zorder=5, 
               edgecolors='white', linewidth=2)
    

    plt.annotate(f'F1 = {f1_scores[optimal_idx]:.3f}', 
                xy=(optimal_threshold, f1_scores[optimal_idx]),
                xytext=(optimal_threshold + 0.05, f1_scores[optimal_idx] + 0.05),
                fontsize=10, color='#FFD700', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5))
    

    plt.axhline(y=0.6, color='#FF6B6B', linestyle=':', 
                linewidth=1.5, alpha=0.6, 
                label='Минимальный Recall = 0.6')

    plt.xlabel('Порог классификации (θ)', 
              color='#E0E0E0', fontsize=12, fontweight='bold')
    plt.ylabel('Значение метрики', 
              color='#E0E0E0', fontsize=12, fontweight='bold')
    plt.title('Зависимость метрик качества от порога классификации', 
             color='#E0E0E0', fontsize=14, fontweight='bold', pad=20)
    
    plt.xlim([0.05, 0.50])
    plt.ylim([0.0, 1.05])
    

    plt.grid(True, alpha=0.3, color='#00D46A', linestyle='--')
    ax.set_axisbelow(True)

    plt.tick_params(colors='#A0A0A0', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#00D46A')

    legend = plt.legend(loc='center right', fontsize=10, 
                       framealpha=0.9, facecolor='#14281E', 
                       edgecolor='#00D46A')
    for text in legend.get_texts():
        text.set_color('#E0E0E0')
    
    plt.tight_layout()

    output_path = "pictures/threshold_analysis.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, 
               facecolor='#0A1A15', bbox_inches='tight')
    print(f"График сохранен: {output_path}")

    output_png = "pictures/threshold_analysis.png"
    plt.savefig(output_png, format='png', dpi=150, 
               facecolor='#0A1A15', bbox_inches='tight')
    print(f"График сохранен: {output_png}")

    print("\n" + "="*60)
    print("СТАТИСТИКА ПО ПОРОГАМ")
    print("="*60)
    print(f"{'Порог':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    

    key_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    for thr in key_thresholds:
        idx = np.argmin(np.abs(thresholds - thr))
        marker = "оптимальный" if thr == 0.30 else ""
        print(f"{thresholds[idx]:<10.2f} {precision_scores[idx]:<12.4f} "
              f"{recall_scores[idx]:<12.4f} {f1_scores[idx]:<12.4f}{marker}")
    
    print("="*60)
    
    best_f1_idx = np.argmax(f1_scores)
    print(f"\nЛучший порог по F1: {thresholds[best_f1_idx]:.2f}")
    print(f"F1-Score: {f1_scores[best_f1_idx]:.4f}")
    print(f"Precision: {precision_scores[best_f1_idx]:.4f}")
    print(f"Recall: {recall_scores[best_f1_idx]:.4f}")
    
    plt.show()
    
    return thresholds, precision_scores, recall_scores, f1_scores


if __name__ == "__main__":
    plot_threshold_analysis()