# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from datetime import datetime
from src.models.hybrid.osago_calculator import OSAGOCalculator

app = Flask(__name__)

MODEL_PATH = "outputs/insurance_model_v1.cbm"
REGIONS_JSON = "static/data/regions.json"

# Инициализация калькулятора
try:
    calculator = OSAGOCalculator(model_path=MODEL_PATH)
    print("✅ Калькулятор ОСАГО загружен")
except Exception as e:
    print(f"⚠️ Ошибка загрузки модели: {e}")
    calculator = None

def load_regions_data():
    try:
        with open(REGIONS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def calculate_age(dob_str):
    if not dob_str: return 25 # Default fallback
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    return datetime.now().year - dob.year

def calculate_experience(license_date_str):
    if not license_date_str: return 0
    l_date = datetime.strptime(license_date_str, "%Y-%m-%d")
    years = (datetime.now() - l_date).days / 365.25
    return max(0, int(years))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/calculate", methods=["GET", "POST"])
def calculate():
    regions_data = load_regions_data()
    
    if request.method == "POST":
        if not calculator:
            return "Model not loaded", 500

        # 1. Сбор данных формы
        region_name = request.form.get("region", "Other")
        weather_data = regions_data.get(region_name, regions_data.get("Other", {}))
        
        # 2. Расчет производных полей
        driver_age = calculate_age(request.form.get("driver_dob"))
        driver_experience = calculate_experience(request.form.get("license_date"))
        vehicle_year = int(request.form.get("vehicle_year", 2020))
        vehicle_age = datetime.now().year - vehicle_year
        
        # Конвертация булевых значений
        night_driving = 0.5 if request.form.get("night_driving") == "yes" else 0.1
        is_owner = 1 if request.form.get("is_owner") == "yes" else 0
        
        # Формирование словаря для модели
        form_data = {
            'driver_age': driver_age,
            'driver_experience': driver_experience,
            'vehicle_age': vehicle_age,
            'vehicle_type': request.form.get("vehicle_type", "sedan"), # Упрощенно
            'engine_power': int(request.form.get("engine_power", 100)),
            'vehicle_purpose': 'personal',
            'region': region_name,
            'pct_days_with_snow': weather_data.get('pct_days_with_snow', 0.3),
            'pct_days_with_rain': weather_data.get('pct_days_with_rain', 0.3),
            'winter_duration_months': weather_data.get('winter_duration_months', 4),
            'base_kbm': float(request.form.get("base_kbm", 1.0)),
            'num_claims': int(request.form.get("num_claims", 0)),
            'violation_count': int(request.form.get("violation_count", 0)),
            'days_since_last_claim': 365, # Пока хардкод, если нет даты последнего ДТП
            'occupation_type': request.form.get("occupation", "office_worker"),
            'avg_trips_per_week': float(request.form.get("trips_per_week", 5)),
            'night_driving_ratio': night_driving,
            'ko_multiplier': 1.0,
            'num_owned_vehicles': 1 if is_owner else 0
        }

        driver_df = pd.DataFrame([form_data])

        # 3. Запуск расчета
        try:
            result = calculator.calculate_osago_premium(
                driver_data=driver_df,
                obd_file_path=None,
                base_tariff=2000.0,
                region_coeff=1.0, # Можно добавить логику коэффициентов регионов
                engine_power_coeff=1.0,
                age_exp_coeff=1.0,
                unlimited_drivers=False,
                season_coeff=1.0
            )
            return render_template("result.html", result=result, input_data=form_data)
        except Exception as e:
            return f"Ошибка расчета: {str(e)}", 500

    # GET запрос - просто показываем форму
    return render_template("calculate.html", regions=list(load_regions_data().keys()))

if __name__ == "__main__":
    app.run(debug=True)