from typing import Dict, Optional, Union
import pandas as pd

from src.models.hybrid.kbm_calculator import HybridKBMCalculator


class OSAGOCalculator:
    """
    Упрощённый калькулятор ОСАГО, использующий гибридную модель для расчёта КБМ.
    """

    def __init__(self, model_path: str = "outputs/insurance_model_v1.cbm"):
        self.kbm_model: HybridKBMCalculator = HybridKBMCalculator(model_path=model_path)
        print("Гибридный калькулятор ОСАГО загружен")

    def calculate_osago_premium(
        self,
        driver_data: pd.DataFrame,
        obd_file_path: Optional[str] = None,
        base_tariff: float = 2000.0,
        region_coeff: float = 1.0,
        engine_power_coeff: float = 1.0,
        age_exp_coeff: float = 1.0,
        unlimited_drivers: bool = False,
        season_coeff: float = 1.0,
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Рассчитывает стоимость ОСАГО.

        Параметры:
            driver_data (pd.DataFrame): DataFrame с одной строкой — данными водителя.
            obd_file_path (str, optional): путь к OBD-файлу (если используется телематика).
            base_tariff (float): базовый тариф (по умолчанию 2000 ₽).
            region_coeff (float): коэффициент региона.
            engine_power_coeff (float): коэффициент мощности двигателя.
            age_exp_coeff (float): коэффициент возраста и стажа.
            unlimited_drivers (bool): признак «неограниченное число водителей».
            season_coeff (float): сезонный коэффициент (например, 0.7 для полугодия).

        Возвращает:
            dict: словарь с деталями расчёта.
        """
        if driver_data.empty:
            raise ValueError("driver_data должен содержать хотя бы одну строку.")
        if len(driver_data) > 1:
            raise ValueError("Для расчёта поддерживается только одна запись (один водитель).")

        # Вызов гибридной модели для получения КБМ
        kbm_result_df = self.kbm_model.calculate(
            cases=[driver_data.iloc[0].to_dict()],
            obd_file_path=obd_file_path,
            show_plot=False
        )

        if kbm_result_df.empty:
            raise RuntimeError("Гибридная модель не вернула результат.")

        base_kbm = float(kbm_result_df['Базовый КБМ'].iloc[0])
        final_kbm = float(kbm_result_df['Итоговый КБМ'].iloc[0])

        print(f"Итоговый КБМ (гибридный): {final_kbm:.3f}")

        # Коэффициент «неограниченное число водителей»
        ko_coeff = 1.8 if unlimited_drivers else 1.0

        # Расчёт премии
        tariff = (
            base_tariff
            * region_coeff
            * engine_power_coeff
            * age_exp_coeff
            * final_kbm
            * ko_coeff
            * season_coeff
        )

        return {
            "base_kbm": round(base_kbm, 4),
            "final_kbm": round(final_kbm, 4),
            "tariff": round(tariff, 2),
            "base_tariff": base_tariff,
            "region_coeff": region_coeff,
            "engine_power_coeff": engine_power_coeff,
            "age_exp_coeff": age_exp_coeff,
            "ko_coeff": ko_coeff,
            "season_coeff": season_coeff,
            "unlimited_drivers": unlimited_drivers,
        }