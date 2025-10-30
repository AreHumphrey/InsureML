from src.models.catboost.insurance_model import InsuranceRiskModel
from src.utils.dtc_checker import check_dtc_in_file
import pandas as pd
import matplotlib.pyplot as plt


class HybridKBMCalculator:
    def __init__(self, model_path: str = "outputs/insurance_model_v1.cbm"):

        self.model = InsuranceRiskModel()

        if not model_path.startswith(""):
            model_path = f"outputs/{model_path}"

        if model_path.endswith(".cbm"):
            try:
                self.model.model.load_model(model_path)
                print(f"Модель загружена из {model_path}")
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки .cbm модели: {e}")
        else:
            try:
                loaded_obj = InsuranceRiskModel(model_path=model_path)
                self.model.model = loaded_obj.model
                print(f"Модель загружена из {model_path} (joblib)")
            except Exception as e:
                raise FileNotFoundError(f"Не удалось загрузить модель: {e}")

    def calculate(self, cases: list, obd_file_path: str = None, show_plot: bool = True) -> pd.DataFrame:

        has_dtc = False
        if obd_file_path:
            has_dtc = check_dtc_in_file(obd_file_path)

        results = []
        for case in cases:
            data = {k: [v] for k, v in case.items() if k != 'description'}
            df = pd.DataFrame(data)

            proba = self.model.predict_proba(df)
            base_kbm = case.get('base_kbm', 1.0)
            adjusted_kbm = self.model.calculate_adjusted_kbm(df, base_kbm=base_kbm)

            final_kbm = adjusted_kbm
            adjustments = []

            if has_dtc:
                final_kbm *= 1.5
                final_kbm = min(final_kbm, 3.92)
                adjustments.append("наличие DTC")

            results.append({
                'Описание': case['description'],
                'Вероятность ДТП': f"{proba:.2%}",
                'Базовый КБМ': round(base_kbm, 2),
                'Рекомендуемый КБМ': round(adjusted_kbm, 2),
                'Итоговый КБМ': round(final_kbm, 2),
                'Корректировки': ', '.join(adjustments) if adjustments else 'нет'
            })

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ГИБРИДНОГО РАСЧЁТА КБМ")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)

        if show_plot:
            self._plot_results(results_df)

        return results_df

    @staticmethod
    def _plot_results(results_df: pd.DataFrame):

        plt.figure(figsize=(12, 6))
        bars = plt.barh(results_df['Описание'], results_df['Итоговый КБМ'], color='skyblue', edgecolor='black')
        plt.title("Итоговый КБМ после учёта DTC", fontsize=16)
        plt.xlabel("Коэффициент бонус-малус (КБМ)")
        plt.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{width}', va='center', ha='left', fontweight='bold')

        plt.tight_layout()
        plt.show()
