# app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from src.models.hybrid.kbm_calculator import HybridKBMCalculator
from src.utils.dtc_checker import check_dtc_in_file

app = Flask(__name__)

MODEL_PATH = "outputs/insurance_model_v1.cbm"
REGIONS_JSON = "static/data/regions.json"
UPLOAD_FOLDER = "static/uploads"

# Создаём папку для загрузок
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Инициализация калькулятора
try:
    calculator = HybridKBMCalculator(model_path=MODEL_PATH)
    print("✅ HybridKBMCalculator загружен")
except Exception as e:
    print(f"⚠️ Ошибка загрузки калькулятора: {e}")
    calculator = None


# ── Вспомогательные функции ──

def load_regions_data():
    """Загружает данные о регионах из JSON"""
    try:
        with open(REGIONS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def calculate_age(dob_str):
    """Считает возраст водителя по дате рождения"""
    if not dob_str:
        return 25
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        age = datetime.now().year - dob.year
        # Корректировка: если день рождения ещё не был в этом году
        if (datetime.now().month, datetime.now().day) < (dob.month, dob.day):
            age -= 1
        return max(18, min(80, age))
    except:
        return 25


def calculate_experience(license_date_str):
    """Считает стаж вождения по дате получения прав"""
    if not license_date_str:
        return 0
    try:
        l_date = datetime.strptime(license_date_str, "%Y-%m-%d")
        years = (datetime.now() - l_date).days / 365.25
        return max(0, min(60, int(years)))
    except:
        return 0


def calculate_base_kbm(num_claims: int, driver_experience: int) -> float:
    """
    Автоматически рассчитывает базовый КБМ на основе истории вождения.
    
    Логика (упрощённая модель ОСАГО):
    - Базовое значение: 1.0
    - За каждый год без ДТП: скидка 5% (макс. до 0.46)
    - За каждое ДТП: повышение на 50%
    - Диапазон: [0.46, 3.92]
    
    Параметры:
        num_claims: количество страховых случаев
        driver_experience: стаж вождения в годах
    
    Возвращает:
        float: рассчитанный коэффициент КБМ
    """
    kbm = 1.0
    
    # Оценка безаварийных лет (упрощённо: стаж минус годы с ДТП)
    claim_free_years = max(0, driver_experience - num_claims * 2)
    
    # Скидка за безаварийную езду (макс. 54% → КБМ 0.46)
    discount = min(0.05 * claim_free_years, 0.54)
    kbm -= discount
    
    # Штраф за страховые случаи
    penalty = num_claims * 0.5
    kbm += penalty
    
    # Ограничиваем диапазон [0.46, 3.92]
    return round(max(0.46, min(3.92, kbm)), 2)


def safe_int(value, default=0, min_val=None, max_val=None):
    """Безопасное преобразование в int с ограничениями"""
    try:
        val = int(float(value)) if value else default
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0, min_val=None, max_val=None):
    """Безопасное преобразование в float с ограничениями"""
    try:
        val = float(value) if value else default
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    except (ValueError, TypeError):
        return default


# ── Маршруты ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/calculate", methods=["GET", "POST"])
def calculate():
    """
    Обработчик страницы калькулятора ОСАГО.
    
    GET: отображает форму ввода данных.
    POST: обрабатывает форму, запускает расчёт через ML-модель, возвращает результат.
    """
    
    # === Логирование запроса (для отладки) ===
    if request.method == "POST":
        print(f"🔍 POST-запрос на /calculate")
        print(f"📦 Form keys: {list(request.form.keys())}")
        print(f"📁 Files: {list(request.files.keys())}")
    
    # === Загрузка справочника регионов ===
    regions_data = load_regions_data()
    
    # === GET: просто показываем форму ===
    if request.method == "GET":
        return render_template("calculate.html", regions=list(regions_data.keys()))
    
    # === POST: обработка формы ===
    
    # 0. Проверка загрузки модели
    if not calculator:
        print("❌ Model not initialized")
        return "Model not loaded. Please restart the server.", 503
    
    try:
        # ── 1. Обработка загруженного DTC-файла ──
        dtc_file = request.files.get('dtc_file')
        has_dtc = False
        obd_file_path = None
        
        if dtc_file and dtc_file.filename and dtc_file.filename.lower().endswith('.csv'):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"dtc_{timestamp}.csv"
                obd_file_path = os.path.join(UPLOAD_FOLDER, filename)
                dtc_file.save(obd_file_path)
                print(f"💾 Файл сохранён: {obd_file_path}")
                
                has_dtc = check_dtc_in_file(obd_file_path)
                print(f"🔍 DTC-анализ: ошибки = {has_dtc}")
                
            except Exception as e:
                print(f"⚠️ Ошибка обработки DTC-файла: {e}")
                has_dtc = False
        
        # ── 2. Сбор и валидация данных формы ──
        
        # Регион + погода из JSON
        region_name = request.form.get("region", "Other")
        weather = regions_data.get(region_name, regions_data.get("Other", {}))
        
        # Расчёт возраста и стажа
        driver_age = calculate_age(request.form.get("driver_dob"))
        driver_experience = calculate_experience(request.form.get("license_date"))
        
        # Возраст автомобиля
        try:
            vehicle_year = int(request.form.get("vehicle_year", 2020))
            vehicle_age = max(0, min(30, datetime.now().year - vehicle_year))
        except (ValueError, TypeError):
            vehicle_year = 2020
            vehicle_age = 5
        
        # История страхования (для автоматического расчёта КБМ)
        num_claims = safe_int(request.form.get("num_claims"), 0, 0, 20)
        violation_count = safe_int(request.form.get("violation_count"), 0, 0, 50)
        
        # ✅ АВТОМАТИЧЕСКИЙ РАСЧЁТ Base KBM
        base_kbm = calculate_base_kbm(num_claims, driver_experience)
        print(f"🧮 Auto-calculated base_kbm: {base_kbm} (claims={num_claims}, exp={driver_experience})")
        
        # Булевы флаги → числовые коэффициенты
        night_driving = 0.5 if request.form.get("night_driving") == "yes" else 0.1
        is_owner = request.form.get("is_owner") == "true"
        unlimited_drivers = request.form.get("unlimited_drivers") == "true"
        
        # ── 3. Формирование кейса для модели ──
        case_data = {
            # Основные признаки
            'driver_age': driver_age,
            'driver_experience': driver_experience,
            'vehicle_age': vehicle_age,
            'vehicle_type': request.form.get("body_type", "sedan"),
            'engine_power': safe_int(request.form.get("engine_power"), 150, 50, 500),
            'vehicle_purpose': request.form.get("vehicle_purpose", "personal"),
            'region': region_name,
            
            # Климат из JSON по региону
            'pct_days_with_snow': weather.get('pct_days_with_snow', 0.3),
            'pct_days_with_rain': weather.get('pct_days_with_rain', 0.3),
            'winter_duration_months': weather.get('winter_duration_months', 4),
            
            # История страхования (base_kbm рассчитан автоматически!)
            'base_kbm': base_kbm,
            'num_claims': num_claims,
            'violation_count': violation_count,
            'days_since_last_claim': safe_int(request.form.get("days_since_last_claim"), 365, 0, 3650),
            
            # Поведенческие данные
            'occupation_type': request.form.get("occupation_type", "office_worker"),
            'avg_trips_per_week': safe_float(request.form.get("trips_per_week"), 5.0, 0, 100),
            'night_driving_ratio': night_driving,
            
            # Коэффициенты полиса
            'ko_multiplier': 1.8 if unlimited_drivers else 1.0,
            'num_owned_vehicles': 1 if is_owner else 0,
            
            # Мета-информация
            'description': f"Driver {driver_age}y, {driver_experience}y exp, {region_name}"
        }
        
        # ── 4. Запуск расчёта через HybridKBMCalculator ──
        result_df = calculator.calculate(
            cases=[case_data],
            obd_file_path=obd_file_path if has_dtc else None,
            show_plot=False
        )
        
        # ── 5. Подготовка данных для шаблона ──
        result_row = result_df.iloc[0].to_dict()
        
        result = {
            # Из модели (русские ключи из DataFrame)
            'tariff': result_row.get('Итоговый КБМ', 0) * 2000,
            'final_kbm': result_row.get('Итоговый КБМ'),
            'base_kbm': base_kbm,  # ← используем рассчитанное значение
            'recommended_kbm': result_row.get('Рекомендуемый КБМ'),
            'accident_proba': result_row.get('Вероятность ДТП'),
            'adjustments': result_row.get('Корректировки', 'нет'),
            
            # Контекст
            'region': region_name,
            'has_dtc': has_dtc,
            'driver_desc': case_data['description'],
            'num_claims': num_claims,
            'violation_count': violation_count
        }
        
        # ── 6. Рендеринг страницы результата ──
        print(f"✅ Расчёт завершён: base_kbm={base_kbm}, final_kbm={result['final_kbm']}, tariff={result['tariff']:.2f}₽")
        
        return render_template(
            "result.html",
            result=result,
            input_data=case_data,
            has_dtc=has_dtc
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Критическая ошибка в /calculate: {e}")
        print(error_trace)
        
        return render_template(
            "result.html",
            result={'error': str(e)},
            input_data={},
            has_dtc=False,
            error_trace=error_trace if app.debug else None
        ), 500


@app.context_processor
def inject_globals():
    """Добавляет переменные во все шаблоны"""
    return {
        'current_year': datetime.now().year,
        'regions': list(load_regions_data().keys())
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)