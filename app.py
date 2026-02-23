# app.py
from flask import Flask, render_template, request
import pandas as pd
from src.models.hybrid.osago_calculator import OSAGOCalculator

app = Flask(__name__)


MODEL_PATH = "outputs/insurance_model_v1.cbm"


calculator = OSAGOCalculator(model_path=MODEL_PATH)
print("✅ Калькулятор ОСАГО загружен")


@app.route("/")
def index():
    return render_template("index.html")

from datetime import datetime
@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}

@app.route("/calculate", methods=["GET", "POST"])
def calculate():
    if request.method == "POST":

        form_data = {
            'driver_age': int(request.form["driver_age"]),
            'driver_experience': int(request.form["driver_experience"]),
            'vehicle_age': int(request.form["vehicle_age"]),
            'vehicle_type': request.form["vehicle_type"],
            'engine_power': int(request.form["engine_power"]),
            'vehicle_purpose': request.form["vehicle_purpose"],
            'region': request.form["region"],
            'pct_days_with_snow': float(request.form["pct_days_with_snow"]),
            'pct_days_with_rain': float(request.form["pct_days_with_rain"]),
            'winter_duration_months': int(request.form["winter_duration_months"]),
            'base_kbm': float(request.form["base_kbm"]),
            'num_claims': int(request.form["num_claims"]),
            'violation_count': int(request.form["violation_count"]),
            'days_since_last_claim': int(request.form["days_since_last_claim"]),
            'occupation_type': request.form["occupation_type"],
            'avg_trips_per_week': float(request.form["avg_trips_per_week"]),
            'night_driving_ratio': float(request.form["night_driving_ratio"]),
            'ko_multiplier': float(request.form["ko_multiplier"]),
            'num_owned_vehicles': int(request.form["num_owned_vehicles"]),
            # 'target' не передаём
        }


        driver_df = pd.DataFrame([form_data])


        region_coeff = float(request.form.get("region_coeff", 1.8))
        engine_power_coeff = float(request.form.get("engine_power_coeff", 1.0))
        age_exp_coeff = float(request.form.get("age_exp_coeff", 1.0))
        unlimited_drivers = request.form.get("unlimited_drivers", default=False, type=bool)
        season_coeff = float(request.form.get("season_coeff", 1.0))


        result = calculator.calculate_osago_premium(
            driver_data=driver_df,
            obd_file_path=None,
            base_tariff=2000.0,
            region_coeff=region_coeff,
            engine_power_coeff=engine_power_coeff,
            age_exp_coeff=age_exp_coeff,
            unlimited_drivers=unlimited_drivers,
            season_coeff=season_coeff
        )


        return render_template("result.html", result=result, input_data=form_data)

    return render_template("calculate.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


if __name__ == "__main__":
    app.run(debug=True)