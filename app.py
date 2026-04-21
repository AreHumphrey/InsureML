from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import json
import os
from datetime import datetime
from src.models.hybrid.kbm_calculator import HybridKBMCalculator
from src.utils.dtc_checker import check_dtc_in_file
from flask import send_file
import io
import csv
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from flask_session import Session  

app = Flask(__name__)

# === Настройка сессий ===
app.config['SECRET_KEY'] = 'osago-secret-key-change-in-prod'  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = './flask_session_data'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)


os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

MODEL_PATH = "outputs/insurance_model_v1.cbm"
REGIONS_JSON = "static/data/regions.json"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    calculator = HybridKBMCalculator(model_path=MODEL_PATH)
    print("HybridKBMCalculator загружен")
except Exception as e:
    print(f"Ошибка загрузки калькулятора: {e}")
    calculator = None


def load_regions_data():
    try:
        with open(REGIONS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def calculate_age(dob_str):
    if not dob_str:
        return 25
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        age = datetime.now().year - dob.year
        if (datetime.now().month, datetime.now().day) < (dob.month, dob.day):
            age -= 1
        return max(18, min(80, age))
    except:
        return 25


def calculate_experience(license_date_str):
    if not license_date_str:
        return 0
    try:
        l_date = datetime.strptime(license_date_str, "%Y-%m-%d")
        years = (datetime.now() - l_date).days / 365.25
        return max(0, min(60, int(years)))
    except:
        return 0


def calculate_base_kbm(num_claims: int, driver_experience: int) -> float:
    kbm = 1.0
    claim_free_years = max(0, driver_experience - num_claims * 2)
    discount = min(0.05 * claim_free_years, 0.54)
    kbm -= discount
    penalty = num_claims * 0.5
    kbm += penalty
    return round(max(0.46, min(3.92, kbm)), 2)


def safe_int(value, default=0, min_val=None, max_val=None):
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
    try:
        val = float(value) if value else default
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    except (ValueError, TypeError):
        return default

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/calculate", methods=["GET", "POST"])
def calculate():
    if request.method == "POST":
        print(f" POST-запрос на /calculate")
    
    regions_data = load_regions_data()

    if request.method == "GET":
        return render_template("calculate.html", regions=list(regions_data.keys()))

    if not calculator:
        print("Model not initialized")
        return "Model not loaded. Please restart the server.", 503
    
    try:
        dtc_file = request.files.get('dtc_file')
        has_dtc = False
        obd_file_path = None
        
        if dtc_file and dtc_file.filename and dtc_file.filename.lower().endswith('.csv'):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"dtc_{timestamp}.csv"
                obd_file_path = os.path.join(UPLOAD_FOLDER, filename)
                dtc_file.save(obd_file_path)
                print(f"Файл сохранён: {obd_file_path}")
                has_dtc = check_dtc_in_file(obd_file_path)
                print(f"DTC-анализ: ошибки = {has_dtc}")
            except Exception as e:
                print(f" Ошибка обработки DTC-файла: {e}")
                has_dtc = False

        region_name = request.form.get("region", "Other")
        weather = regions_data.get(region_name, regions_data.get("Other", {}))

        driver_age = calculate_age(request.form.get("driver_dob"))
        driver_experience = calculate_experience(request.form.get("license_date"))

        try:
            vehicle_year = int(request.form.get("vehicle_year", 2020))
            vehicle_age = max(0, min(30, datetime.now().year - vehicle_year))
        except (ValueError, TypeError):
            vehicle_year = 2020
            vehicle_age = 5

        num_claims = safe_int(request.form.get("num_claims"), 0, 0, 20)
        violation_count = safe_int(request.form.get("violation_count"), 0, 0, 50)
        
        base_kbm = calculate_base_kbm(num_claims, driver_experience)
        print(f"Auto-calculated base_kbm: {base_kbm} (claims={num_claims}, exp={driver_experience})")

        night_driving = 0.5 if request.form.get("night_driving") == "yes" else 0.1
        is_owner = request.form.get("is_owner") == "true"
        unlimited_drivers = request.form.get("unlimited_drivers") == "true"

        case_data = {
            'driver_age': driver_age,
            'driver_experience': driver_experience,
            'vehicle_age': vehicle_age,
            'vehicle_type': request.form.get("body_type", "sedan"),
            'engine_power': safe_int(request.form.get("engine_power"), 150, 50, 500),
            'vehicle_purpose': request.form.get("vehicle_purpose", "personal"),
            'region': region_name,
            'pct_days_with_snow': weather.get('pct_days_with_snow', 0.3),
            'pct_days_with_rain': weather.get('pct_days_with_rain', 0.3),
            'winter_duration_months': weather.get('winter_duration_months', 4),
            'base_kbm': base_kbm,
            'num_claims': num_claims,
            'violation_count': violation_count,
            'days_since_last_claim': safe_int(request.form.get("days_since_last_claim"), 365, 0, 3650),
            'occupation_type': request.form.get("occupation_type", "office_worker"),
            'avg_trips_per_week': safe_float(request.form.get("trips_per_week"), 5.0, 0, 100),
            'night_driving_ratio': night_driving,
            'ko_multiplier': 1.8 if unlimited_drivers else 1.0,
            'num_owned_vehicles': 1 if is_owner else 0,
            'description': f"Driver {driver_age}y, {driver_experience}y exp, {region_name}"
        }

        result_df = calculator.calculate(
            cases=[case_data],
            obd_file_path=obd_file_path if has_dtc else None,
            show_plot=False
        )
        
        result_row = result_df.iloc[0].to_dict()
        
        result = {
            'tariff': result_row.get('Итоговый КБМ', 0) * 2000,
            'final_kbm': result_row.get('Итоговый КБМ'),
            'base_kbm': base_kbm,
            'recommended_kbm': result_row.get('Рекомендуемый КБМ'),
            'accident_proba': result_row.get('Вероятность ДТП'),
            'adjustments': result_row.get('Корректировки', 'нет'),
            'region': region_name,
            'has_dtc': has_dtc,
            'driver_desc': case_data['description'],
            'num_claims': num_claims,
            'violation_count': violation_count,
            # Доп. поля для экспорта
            'driver_age': driver_age,
            'driver_experience': driver_experience,
            'vehicle_age': vehicle_age,
            'engine_power': case_data['engine_power'],
        }

        session['last_calculation'] = result
        print(f"Результат сохранён в сессию")
    
        print(f"Расчёт завершён: base_kbm={base_kbm}, final_kbm={result['final_kbm']}, tariff={result['tariff']:.2f}₽")
        
        return render_template(
            "result.html",
            result=result,
            input_data=case_data,
            has_dtc=has_dtc
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Критическая ошибка в /calculate: {e}")
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
    return {
        'current_year': datetime.now().year,
        'regions': list(load_regions_data().keys())
    }


@app.route("/download/pdf")
def download_pdf():
    """Генерирует PDF с ПОСЛЕДНИМ рассчитанным результатом"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm

    result = session.get('last_calculation')
    
    if not result:
        return "No calculation result found. Please calculate first.", 400
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#00D46A'), spaceAfter=30)
    
    elements.append(Paragraph("OSAGOCalculator — Result", title_style))
    elements.append(Spacer(1, 0.5*cm))
    
    data = [
        ['Parameter', 'Value'],
        ['Final CTP Tariff', f"{result['tariff']:.0f} ₽"],
        ['Final KBM', f"{result['final_kbm']:.2f}"],
        ['Base KBM', f"{result['base_kbm']:.2f}"],
        ['Region', result['region']],
        ['Driver', result['driver_desc']],
        ['Accidents', str(result['num_claims'])],
        ['DTC Errors', 'Yes' if result['has_dtc'] else 'No'],
    ]
    
    table = Table(data, colWidths=[6*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00664E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#14281E')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#00D46A')),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 1*cm))
    
    elements.append(Paragraph(f"© {datetime.now().year} OSAGOCalculator, Inc. | Far Eastern Federal University", 
                             ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.grey)))
    
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"osago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )


@app.route("/download/csv")
def download_csv():

    result = session.get('last_calculation')
    
    if not result:
        return "No calculation result found. Please calculate first.", 400
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['OSAGOCalculator Results'])
    writer.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])
    
    writer.writerow(['Parameter', 'Value'])
    writer.writerow(['Final CTP Tariff (₽)', f"{result['tariff']:.0f}"])
    writer.writerow(['Final KBM', f"{result['final_kbm']:.2f}"])
    writer.writerow(['Base KBM', f"{result['base_kbm']:.2f}"])
    writer.writerow(['Region', result['region']])
    writer.writerow(['Driver', result['driver_desc']])
    writer.writerow(['Accidents', result['num_claims']])
    writer.writerow(['DTC Errors', 'Yes' if result['has_dtc'] else 'No'])
    writer.writerow([])
    
    writer.writerow(['Driver Details'])
    writer.writerow(['Age (years)', result['driver_age']])
    writer.writerow(['Experience (years)', result['driver_experience']])
    writer.writerow(['Vehicle Age (years)', result['vehicle_age']])
    writer.writerow(['Engine Power (hp)', result['engine_power']])
    
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"osago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route("/download/graph")
def download_graph():

    result = session.get('last_calculation')
    
    if not result:
        return " No calculation result found. Please calculate first.", 400
    
    base_kbm = result['base_kbm']
    final_kbm = result['final_kbm']
    average_kbm = 1.0
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0A1A15')
    ax.set_facecolor('#0F2319')
    
    categories = ['Base KBM', 'Final KBM', 'Average KBM']
    values = [base_kbm, final_kbm, average_kbm]
    colors = ['#00D46A', '#00A55E', '#8B8B8B']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='#00D46A', linewidth=2)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', 
                color='#E0E0E0', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('KBM Coefficient', color='#A0A0A0', fontsize=12)
    ax.set_title('Your KBM vs Average Market KBM', color='#E0E0E0', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.3)
    
    ax.grid(axis='y', alpha=0.3, color='#00D46A', linestyle='--')
    ax.set_axisbelow(True)
    
    ax.tick_params(colors='#A0A0A0')
    for spine in ax.spines.values():
        spine.set_color('#00D46A')
    
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buffer.seek(0)
    plt.close(fig)
    
    return send_file(
        buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=f"kbm_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)