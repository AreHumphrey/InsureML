from src.models.hybrid.kbm_calculator import HybridKBMCalculator  # ← НОВЫЙ ИМПОРТ
import pandas as pd


class OSAGOCalculator:
    def __init__(self, model_path: str = "src/outputs/insurance_model_v1.cbm"):

        self.kbm_model = HybridKBMCalculator(model_path=model_path)
        print("Гибридный калькулятор ОСАГО загружен")

    def calculate_osago_premium(
        self,
        driver_data: pd.DataFrame,
        obd_file_path: str = None,
        base_tariff: float = 2000.0,
        region_coeff: float = 1.0,
        engine_power_coeff: float = 1.0,
        age_exp_coeff: float = 1.0,
        unlimited_drivers: bool = False,
        season_coeff: float = 1.0
    ) -> dict:

        kbm_result_df = self.kbm_model.calculate(
            cases=[driver_data.iloc[0].to_dict()],
            obd_file_path=obd_file_path,
            show_plot=False
        )

        final_kbm = kbm_result_df['Итоговый КБМ'].iloc[0]
        print(f" Итоговый КБМ (гибридный): {final_kbm}")

        ko_coeff = 1.0
        if unlimited_drivers:
            ko_coeff = 1.8

        tariff = base_tariff * region_coeff * engine_power_coeff * age_exp_coeff * final_kbm * ko_coeff * season_coeff

        return {
            'base_kbm': kbm_result_df['Базовый КБМ'].iloc[0],
            'final_kbm': final_kbm,
            'tariff': round(tariff, 2),
            'base_tariff': base_tariff,
            'region_coeff': region_coeff,
            'engine_power_coeff': engine_power_coeff,
            'age_exp_coeff': age_exp_coeff,
            'ko_coeff': ko_coeff,
            'season_coeff': season_coeff
        }