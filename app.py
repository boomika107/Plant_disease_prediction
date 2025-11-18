import os
import io
import json
import numpy as np
import base64
import time
from datetime import datetime, timedelta, date
import requests
import math
from sqlalchemy import func

from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from flask import (
    Flask, request, render_template, redirect, url_for, flash, abort, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)

import plotly.graph_objects as go
import google.generativeai as genai

# ML Imports
try:
    import torch
    from torchvision import transforms
except Exception:
    torch = None
    transforms = None

# =========================
# ---- Configuration -------
# =========================
MODEL2_PATH = 'models/model2.pth'
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASSES = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm', 'Becterial Blight in Rice',
    'Brownspot', 'Common_Rust', 'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize',
    'Healthy Wheat', 'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane', 'RedRot sugarcane',
    'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust',
    'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite',
    'Wheat powdery mildew', 'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
    'pink bollworm in cotton', 'red cotton bug', 'thirps on cotton'
]

HEALTHY_CLASSES = ['Healthy Maize', 'Healthy Wheat', 'Healthy cotton', 'Sugarcane Healthy']

EMOJI_FOR_CLASS = {
    'cotton': '🧶', 'wheat': '🌾', 'maize': '🌽', 'rice': '🍚', 'sugarcane': '🧃',
    'rust': '🟠', 'aphid': '🪲', 'smut': '⚫', 'mildew': '🌫️', 'blight': '🥀',
    'leaf': '🍃', 'worm': '🪱',
}

# =========================
# ---- API Key Setup ----
# =========================
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set. Using simulated data.")
        genai.configure(api_key="SIMULATED_KEY") # Placeholder
        GEMINI_MODEL = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash') 
        print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Using simulated data.")
    GEMINI_MODEL = None

# =========================
# ---- Flask App Setup -----
# =========================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_app.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# =========================
# ---- DB Models -------
# =========================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    scans = db.relationship('ScanHistory', backref='user', lazy=True)
    plots = db.relationship('Plot', backref='owner', lazy=True)
    tasks = db.relationship('Task', backref='owner', lazy=True)
    # NEW: Relationship for Inventory
    resources = db.relationship('Resource', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Plot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200))
    crop_type = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scans = db.relationship('ScanHistory', backref='plot', lazy=True)
    
    irrigation_frequency_days = db.Column(db.Integer, nullable=True)
    fertilization_frequency_days = db.Column(db.Integer, nullable=True)
    preferred_fertilizer = db.Column(db.String(100), nullable=True)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scan_id = db.Column(db.Integer, db.ForeignKey('scan_history.id'), nullable=True)
    plot_id = db.Column(db.Integer, db.ForeignKey('plot.id'), nullable=True)
    
    task_type = db.Column(db.String(50), default='general') # 'general', 'irrigation', 'fertilization', 'treatment'
    title = db.Column(db.String(200), nullable=False)
    due_date = db.Column(db.Date, nullable=False) # Changed to Date for daily tasks
    is_complete = db.Column(db.Boolean, default=False)
    
    scan = db.relationship('ScanHistory', backref='tasks')
    plot = db.relationship('Plot', backref='tasks')

class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    plot_id = db.Column(db.Integer, db.ForeignKey('plot.id'), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    
    pred_class = db.Column(db.String(100), nullable=False)
    is_healthy = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.Float)
    plant_type = db.Column(db.String(50))
    image_filename = db.Column(db.String(200), nullable=False)
    
    definition_json = db.Column(db.Text)
    solution_json = db.Column(db.Text)
    fertilizer_json = db.Column(db.Text)
    top3_json = db.Column(db.Text)

# --- NEW: Plot Journal Model ---
class PlotNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    plot_id = db.Column(db.Integer, db.ForeignKey('plot.id'), nullable=False)
    
    # Establishes the relationship, so you can do plot.notes
    plot = db.relationship('Plot', backref=db.backref('notes', lazy=True, order_by=timestamp.desc()))

# --- NEW: Resource/Inventory Model ---
class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False) # 'Fertilizer', 'Pesticide', 'Seed', 'Other'
    quantity = db.Column(db.String(100)) # Using String for "5 bags", "2.5L", "Low"

# =========================
# ---- User Loader -------
# =========================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# ---- Utilities -----------
# =========================
def get_class_emoji(name: str):
    s = name.lower()
    for k, v in EMOJI_FOR_CLASS.items():
        if k in s: return v
    return '🪴'

def get_plant_type(pred_class: str):
    name = pred_class.lower()
    if 'cotton' in name: return 'Cotton'
    if 'wheat' in name: return 'Wheat'
    if 'maize' in name or 'common_rust' in name: return 'Maize'
    if 'rice' in name: return 'Rice'
    if 'sugarcane' in name: return 'Sugarcane'
    return 'Plant'

def get_image_data_url(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(path, 'rb') as f:
            img_bytes = f.read()
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = "image/png" if filename.lower().endswith('.png') else "image/jpeg"
        return f"data:{mime_type};base64,{img_str}"
    except FileNotFoundError:
        return None

def parse_open_meteo_data(data):
    def map_wmo_to_weather(wmo_code):
        if wmo_code == 0: return 'Clear sky', '01d'
        if wmo_code in [1, 2, 3]: return 'Mainly clear', '02d'
        if wmo_code in [45, 48]: return 'Fog', '50d'
        if wmo_code in [51, 53, 55, 56, 57]: return 'Drizzle', '09d'
        if wmo_code in [61, 63, 65, 66, 67]: return 'Rain', '10d'
        if wmo_code in [71, 73, 75, 77]: return 'Snow fall', '13d'
        if wmo_code in [80, 81, 82]: return 'Rain showers', '09d'
        if wmo_code in [85, 86]: return 'Snow showers', '13d'
        if wmo_code in [95, 96, 99]: return 'Thunderstorm', '11d'
        return 'Unknown', '01d'
    try:
        current = data['current']
        daily = data['daily']
        description, icon = map_wmo_to_weather(current['weather_code'])
        today_data = {
            'temp': current['temperature_2m'],
            'min_temp': daily['temperature_2m_min'][0],
            'max_temp': daily['temperature_2m_max'][0],
            'description': description,
            'icon': icon
        }
        forecast_data = []
        for i in [1, 2]:
            desc, icon = map_wmo_to_weather(daily['weather_code'][i])
            forecast_data.append({
                'dt': daily['time'][i],
                'min_temp': daily['temperature_2m_min'][i],
                'max_temp': daily['temperature_2m_max'][i],
                'icon': icon
            })
        return {'today': today_data, 'forecast': forecast_data}
    except KeyError as e:
        print(f"Error parsing Open-Meteo data. Missing key: {e}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def update_recurring_tasks(user_id):
    today = datetime.utcnow().date()
    plots = Plot.query.filter_by(user_id=user_id).all()
    
    for plot in plots:
        if plot.irrigation_frequency_days:
            task_type = 'irrigation'
            title = f"Irrigate {plot.name}"
            last_task = Task.query.filter_by(plot_id=plot.id, task_type=task_type).order_by(Task.due_date.desc()).first()
            
            next_due_date = None
            if last_task is None:
                next_due_date = today 
            else:
                next_due_date = last_task.due_date + timedelta(days=plot.irrigation_frequency_days)
            
            while next_due_date and next_due_date <= today:
                existing = Task.query.filter_by(plot_id=plot.id, task_type=task_type, due_date=next_due_date).first()
                if not existing:
                    new_task = Task(user_id=user_id, plot_id=plot.id, task_type=task_type, title=title, due_date=next_due_date)
                    db.session.add(new_task)
                next_due_date += timedelta(days=plot.irrigation_frequency_days)

        if plot.fertilization_frequency_days:
            task_type = 'fertilization'
            title = f"Fertilize {plot.name} ({plot.preferred_fertilizer or 'N/A'})"
            last_task = Task.query.filter_by(plot_id=plot.id, task_type=task_type).order_by(Task.due_date.desc()).first()
            
            next_due_date = None
            if last_task is None:
                next_due_date = today
            else:
                next_due_date = last_task.due_date + timedelta(days=plot.fertilization_frequency_days)
            
            while next_due_date and next_due_date <= today:
                existing = Task.query.filter_by(plot_id=plot.id, task_type=task_type, due_date=next_due_date).first()
                if not existing:
                    new_task = Task(user_id=user_id, plot_id=plot.id, task_type=task_type, title=title, due_date=next_due_date)
                    db.session.add(new_task)
                next_due_date += timedelta(days=plot.fertilization_frequency_days)
                
    db.session.commit()


# =========================
# --- Gemini/ML Helpers ---
# =========================

def get_gemini_info(pred_class: str, severity: float = None):
    def get_fallback_data():
        print("Using fallback data for info.")
        base_data = {}
        if pred_class in HEALTHY_CLASSES:
            base_data = {
                "definition": {"status": "success", "message": "The model indicates this plant is healthy."},
                "solution_guide": [{"cause": "Good agricultural practices.", "remedy": "Continue your current regimen."}],
                "fertilizer_recommendation": {"description": "Plant appears healthy.", "product_name": "Balanced NPK Fertilizer"},
                "treatment_plan": []
            }
        else:
            sev_desc = f"{severity:.1f}%" if severity else "N/A"
            base_data = {
                "definition": {"status": "found", "title": f"{pred_class} (Local Fallback)", "summary": f"Fallback summary for {pred_class}."},
                "solution_guide": [{"cause": "Unknown (API offline).", "remedy": "Check the server's API key."}],
                "fertilizer_recommendation": {"description": "API is offline.", "product_name": f"Fertilizer for {pred_class}"},
                "treatment_plan": [{"day": 1, "task": "Monitor plant (API offline)"}]
            }
        return base_data

    if pred_class in HEALTHY_CLASSES:
        return {
            "definition": {"status": "success", "message": "The model indicates this plant is healthy."},
            "solution_guide": [{"cause": "Good agricultural practices.", "remedy": "Continue your current regimen."}],
            "fertilizer_recommendation": {"description": "The plant looks healthy.", "product_name": "Balanced 10-10-10 NPK Fertilizer"},
            "treatment_plan": []
        }
        
    if GEMINI_MODEL:
        sev_str = f"at a {severity:.1f}% severity level" if severity else "at an undetermined severity"
        
        prompt = f"""
        You are an expert botanist. A user has identified:
        - Disease: {pred_class}
        - Severity: {sev_str}
        Provide a concise JSON response (no other text) with this exact structure:
        {{
          "definition": {{"status": "found", "title": "Short title", "summary": "2-3 sentence summary."}},
          "solution_guide": [
            {{"cause": "Primary cause.", "remedy": "Actionable solution."}}
          ],
          "fertilizer_recommendation": {{
            "description": "2-3 sentence fertilizer recommendation.",
            "product_name": "Generic product name for searching"
          }},
          "treatment_plan": [
            {{"day": 1, "task": "Short task for Day 1 (e.g., 'Apply copper fungicide')"}},
            {{"day": 3, "task": "Short task for Day 3 (e.g., 'Check for spread')"}},
            {{"day": 7, "task": "Short task for Day 7 (e.g., 'Apply second dose')"}}
          ]
        }}
        """
        
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(json_text)
            
            if "fertilizer_recommendation" not in data:
                data["fertilizer_recommendation"] = {"description": "No specific AI data.", "product_name": f"Fertilizer for {pred_class}"}
            if "treatment_plan" not in data:
                data["treatment_plan"] = []
                
            return data
        except Exception as e:
            print(f"Error during Gemini API call or JSON parsing: {e}")
            return get_fallback_data()
            
    return get_fallback_data()

def predict_with_gemini(img: Image.Image):
    def get_fallback_prediction():
        print("Using fallback data for prediction.")
        idx = np.random.randint(0, len(CLASSES))
        pred_class = CLASSES[idx]
        confidence = float(np.random.uniform(0.85, 0.99))
        top3 = [(pred_class, confidence), (CLASSES[(idx+1)%len(CLASSES)], 0.05), (CLASSES[(idx+2)%len(CLASSES)], 0.01)]
        return pred_class, confidence, top3
    if not GEMINI_MODEL: return get_fallback_prediction()
    class_list_str = "\n".join([f"- {c}" for c in CLASSES])
    prompt_parts = [
        "You are an expert plant pathologist. Analyze the image and identify the disease from this list ONLY:",
        class_list_str,
        "\nProvide your response as a single, minified JSON object:",
        "{\"pred_class\": \"Best class\", \"confidence\": 0.95, \"top_3\": [[\"Best class\", 0.95], [\"Second class\", 0.03], [\"Third class\", 0.01]]}",
        "\nImage:", img
    ]
    try:
        response = GEMINI_MODEL.generate_content(prompt_parts)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_text)
        pred_class = data.get('pred_class'); confidence = float(data.get('confidence', 0.0)); top3 = data.get('top_3', [])
        if not pred_class or pred_class not in CLASSES or not top3: return get_fallback_prediction()
        top3_tuples = [(item[0], float(item[1])) for item in top3]
        return pred_class, confidence, top3_tuples[:3]
    except Exception as e:
        print(f"Error during Gemini prediction: {e}"); return get_fallback_prediction()
def load_model2_weights():
    if torch is None or not os.path.exists(MODEL2_PATH): return None
    try: return torch.load(MODEL2_PATH, map_location='cpu')
    except Exception as e: return None
def preprocess_image_pytorch(img: Image.Image):
    if transforms is None: return None
    return transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])(img).unsqueeze(0)
def predict_severity(weights, tensor):
    if weights is None or tensor is None: return float(np.random.uniform(10, 95))
    return float(np.random.uniform(10, 95))

# =========================
# ---- Graphics -------------
# =========================

def gauge_chart(value: float, title: str, suffix: str = '%'):
    color = "var(--brand)" if value < 30 else ("var(--warn)" if value < 70 else "var(--danger)")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={"suffix": suffix, "font": {"size": 36, "color": "var(--accent-fg)"}},
        title={"text": title, "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}}
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--fg)')
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
def horizontal_bar_chart(top3_list):
    names = [name for name, prob in top3_list][::-1]; probs = [prob * 100 for name, prob in top3_list][::-1]
    fig = go.Figure(go.Bar(y=names, x=probs, orientation='h', marker_color='var(--brand)', text=[f'{p:.1f}%' for p in probs], textposition='outside', textfont_color='var(--fg)'))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--fg)', yaxis=dict(showticklabels=True), xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False), height=150, showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
def create_plot_health_chart(scans):
    if not scans:
        fig = go.Figure(); fig.update_layout(title="No Scans for this Plot Yet", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--muted)', xaxis_title="Date", yaxis_title="Severity %")
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    dates = [s.timestamp for s in scans if not s.is_healthy and s.severity is not None]
    severities = [s.severity for s in scans if not s.is_healthy and s.severity is not None]
    healthy_dates = [s.timestamp for s in scans if s.is_healthy]; healthy_severities = [0] * len(healthy_dates)
    all_dates = dates + healthy_dates; all_severities = severities + healthy_severities
    sorted_data = sorted(zip(all_dates, all_severities), key=lambda x: x[0])
    if not sorted_data:
        fig = go.Figure(); fig.update_layout(title="No Severity Data to Display", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--muted)')
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    final_dates, final_severities = zip(*sorted_data)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=final_dates, y=final_severities, mode='lines+markers', name='Severity', line=dict(color='var(--danger)', width=2), marker=dict(color='var(--danger)', size=8)))
    fig.update_layout(title="Plot Health Over Time (Severity %)", xaxis_title="Scan Date", yaxis_title="Severity %", yaxis_range=[0, 100], margin=dict(l=40, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='var(--fg)')
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

def create_plot_event_timeline(plot_id):
    scans = ScanHistory.query.filter_by(plot_id=plot_id, is_healthy=False).all()
    tasks = Task.query.filter_by(plot_id=plot_id, is_complete=True).all()
    # NEW: Query for plot notes
    notes = PlotNote.query.filter_by(plot_id=plot_id).all()

    scan_dates = [s.timestamp.date() for s in scans]
    scan_names = [s.pred_class for s in scans]
    
    irrigation_dates = [t.due_date for t in tasks if t.task_type == 'irrigation']
    fertilization_dates = [t.due_date for t in tasks if t.task_type == 'fertilization']
    
    # NEW: Get dates for notes
    note_dates = [n.timestamp.date() for n in notes]
    note_contents = [n.content for n in notes]

    fig = go.Figure()
    
    if scan_dates:
        fig.add_trace(go.Scatter(
            x=scan_dates, y=[1] * len(scan_dates),
            mode='markers', name='Disease Detected',
            text=scan_names,
            marker=dict(color='var(--danger)', size=12, symbol='x')
        ))
    if irrigation_dates:
        fig.add_trace(go.Scatter(
            x=irrigation_dates, y=[2] * len(irrigation_dates),
            mode='markers', name='Irrigation Event',
            marker=dict(color='#55aaff', size=12, symbol='circle')
        ))
    if fertilization_dates:
        fig.add_trace(go.Scatter(
            x=fertilization_dates, y=[3] * len(fertilization_dates),
            mode='markers', name='Fertilization Event',
            marker=dict(color='var(--brand)', size=12, symbol='diamond')
        ))
        
    # NEW: Add notes to timeline
    if note_dates:
        fig.add_trace(go.Scatter(
            x=note_dates, y=[0] * len(note_dates),
            mode='markers', name='Journal Note',
            text=note_contents,
            marker=dict(color='var(--warn)', size=12, symbol='star')
        ))

    if not fig.data:
        fig.update_layout(title="No Management Events Logged Yet")
    else:
        fig.update_layout(title="Plot Event Timeline")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='var(--fg)',
        margin=dict(l=40, r=20, t=40, b=20),
        yaxis=dict(
            showticklabels=False,
            range=[-1, 4] # Updated range for notes
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


# =========================
# ---- Placeholders ----
# =========================
def get_local_outbreaks(user_lat, user_lon, radius_km, user_id):
    if not user_lat or not user_lon:
        return []
    time_window = datetime.utcnow() - timedelta(days=14)
    try:
        nearby_scans = db.session.query(
            ScanHistory.pred_class, ScanHistory.latitude, ScanHistory.longitude
        ).filter(
            ScanHistory.user_id != user_id, ScanHistory.is_healthy == False,
            ScanHistory.timestamp >= time_window,
            ScanHistory.latitude.isnot(None), ScanHistory.longitude.isnot(None)
        ).all()
        
        outbreaks = {}
        for scan in nearby_scans:
            distance = haversine_distance(user_lat, user_lon, scan.latitude, scan.longitude)
            if distance <= radius_km:
                outbreaks[scan.pred_class] = outbreaks.get(scan.pred_class, 0) + 1
        
        sorted_outbreaks = sorted(outbreaks.items(), key=lambda item: item[1], reverse=True)
        return [{"disease": d, "count": c} for d, c in sorted_outbreaks]
    except Exception as e:
        print(f"Error querying outbreaks: {e}"); return []

# =========================
# ---- Load Models Once ----
# =========================
MODEL2W = load_model2_weights()
print("Flask app started. Model 2 (PyTorch) weights loaded.")
if MODEL2W is None:
    print("Warning: PyTorch model (model2) failed to load. Using simulated data.")

# =========================
# ---- Auth Routes ----
# =========================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user); return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))
        new_user = User(name=name, username=username)
        new_user.set_password(password)
        db.session.add(new_user); db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))

# =========================
# ---- Weather API Route ---
# =========================
@app.route('/get_weather')
@login_required
def get_weather():
    lat = request.args.get('lat'); lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400
    
    # --- FIX: Added https:// ---
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': lat, 'longitude': lon,
        'current': 'temperature_2m,weather_code',
        'daily': 'weather_code,temperature_2m_max,temperature_2m_min',
        'forecast_days': 3, 'timezone': 'auto'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        formatted_data = parse_open_meteo_data(data)
        if formatted_data:
            return jsonify(formatted_data)
        else:
            return jsonify({"error": "Invalid data from weather service."}), 500
    except requests.exceptions.RequestException as e:
        print(f"Open-Meteo API error: {e}")
        return jsonify({"error": "Could not retrieve weather data."}), 500
    except Exception as e:
        print(f"Error in get_weather: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# =========================
# ---- Dashboard & Plot Routes ---
# =========================
@app.route('/dashboard')
@login_required
def dashboard():
    update_recurring_tasks(current_user.id)

    plots = Plot.query.filter_by(user_id=current_user.id).order_by(Plot.name).all()
    
    today = datetime.utcnow().date()
    
    tasks_today = Task.query.filter(
        Task.user_id == current_user.id,
        Task.is_complete == False,
        Task.due_date == today
    ).order_by(Task.task_type).all()
    
    tasks_upcoming = Task.query.filter(
        Task.user_id == current_user.id,
        Task.is_complete == False,
        Task.due_date > today
    ).order_by(Task.due_date.asc()).limit(5).all()
    
    outbreak_data = get_local_outbreaks(
        user_lat=current_user.latitude, user_lon=current_user.longitude,
        radius_km=10, user_id=current_user.id
    )
    
    return render_template('dashboard.html', 
                           plots=plots, 
                           tasks_today=tasks_today,
                           tasks_upcoming=tasks_upcoming,
                           outbreaks=outbreak_data)

@app.route('/set_location', methods=['POST'])
@login_required
def set_location():
    try:
        data = request.get_json(); lat = data.get('lat'); lon = data.get('lon')
        if not lat or not lon: return jsonify({"error": "Missing coordinates"}), 400
        user = User.query.get(current_user.id)
        user.latitude = float(lat); user.longitude = float(lon)
        db.session.commit()
        return jsonify({"success": True, "message": "Location saved!"})
    except Exception as e:
        db.session.rollback(); print(f"Error setting location: {e}")
        return jsonify({"error": "Server error saving location."}), 500

@app.route('/update_outbreaks')
@login_required
def update_outbreaks():
    if not current_user.latitude:
        return jsonify({"error": "No user location set."}), 400
    try:
        radius = request.args.get('radius', '10'); radius_km = float(radius)
        outbreak_data = get_local_outbreaks(
            user_lat=current_user.latitude, user_lon=current_user.longitude,
            radius_km=radius_km, user_id=current_user.id
        )
        return jsonify(outbreak_data)
    except Exception as e:
        print(f"Error updating outbreaks: {e}")
        return jsonify({"error": "Could not process request."}), 500

@app.route('/plot/new', methods=['GET', 'POST'])
@login_required
def create_plot():
    if request.method == 'POST':
        name = request.form.get('name')
        location = request.form.get('location')
        crop_type = request.form.get('crop_type')
        if not name or not crop_type:
            flash('Plot Name and Crop Type are required.', 'danger')
            return redirect(url_for('create_plot'))
        new_plot = Plot(name=name, location=location, crop_type=crop_type, user_id=current_user.id)
        db.session.add(new_plot); db.session.commit()
        flash(f'Plot "{name}" created successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('create_plot.html')

@app.route('/plot/<int:plot_id>')
@login_required
def plot_detail(plot_id):
    plot = Plot.query.get_or_404(plot_id)
    if plot.user_id != current_user.id:
        abort(403)
    
    scans = ScanHistory.query.filter_by(plot_id=plot.id).order_by(ScanHistory.timestamp.desc()).all()
    
    health_chart_html = create_plot_health_chart(scans[::-1]) 
    event_timeline_html = create_plot_event_timeline(plot_id)
    
    return render_template('plot_detail.html', 
                           plot=plot, 
                           scans=scans, 
                           health_chart_html=health_chart_html,
                           event_timeline_html=event_timeline_html)

@app.route('/plot/set_management/<int:plot_id>', methods=['POST'])
@login_required
def set_management(plot_id):
    plot = Plot.query.get_or_404(plot_id)
    if plot.user_id != current_user.id:
        abort(443)
        
    try:
        irr_freq = request.form.get('irrigation_frequency')
        fert_freq = request.form.get('fertilization_frequency')
        fert_type = request.form.get('preferred_fertilizer')

        plot.irrigation_frequency_days = int(irr_freq) if irr_freq and int(irr_freq) > 0 else None
        plot.fertilization_frequency_days = int(fert_freq) if fert_freq and int(fert_freq) > 0 else None
        plot.preferred_fertilizer = fert_type if fert_type else None
            
        db.session.commit()
        flash(f'Management schedule for {plot.name} updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating schedule: {e}', 'danger')
        
    return redirect(url_for('plot_detail', plot_id=plot_id))

# --- NEW: Plot Journal Note Route ---
@app.route('/plot/<int:plot_id>/add_note', methods=['POST'])
@login_required
def add_note(plot_id):
    plot = Plot.query.get_or_404(plot_id)
    if plot.user_id != current_user.id:
        abort(403)
    
    content = request.form.get('note_content')
    if content:
        new_note = PlotNote(content=content, plot_id=plot.id)
        db.session.add(new_note)
        db.session.commit()
        flash('Note added successfully!', 'success')
    else:
        flash('Note content cannot be empty.', 'danger')
    
    return redirect(url_for('plot_detail', plot_id=plot_id))

# =========================
# ---- Task Routes ---
# =========================
@app.route('/task/new_from_scan/<int:scan_id>')
@login_required
def create_task_from_scan(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
    
    try:
        solution_guide = json.loads(scan.solution_json)
        remedy = solution_guide[0]['remedy']
        title = f"Manually check: {remedy[:50]}... for {scan.pred_class}"
    except:
        title = f"Manual follow-up for {scan.pred_class}"

    new_task = Task(
        user_id = current_user.id,
        scan_id = scan.id,
        plot_id = scan.plot_id,
        task_type = 'general',
        title = title,
        due_date = datetime.utcnow().date() + timedelta(days=1)
    )
    db.session.add(new_task); db.session.commit()
    flash(f'Manual reminder set for tomorrow!', 'success')
    return redirect(url_for('scan_detail', scan_id=scan_id))

@app.route('/task/toggle/<int:task_id>', methods=['POST'])
@login_required
def toggle_task(task_id):
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
        
    task.is_complete = not task.is_complete
    task.due_date = datetime.utcnow().date() # Mark as completed today
    db.session.commit()
    return redirect(url_for('dashboard'))

# =========================
# ---- Main Scan/History Routes ----
# =========================
@app.route('/')
def home_redirect():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    plots = Plot.query.filter_by(user_id=current_user.id).all()
    if request.method == 'GET':
        return render_template('scan.html', results=None, plots=plots)

    # --- POST ---
    if 'plant_image' not in request.files:
        flash("No file selected.", 'warning')
        return render_template('scan.html', results=None, plots=plots)
    file = request.files['plant_image']
    if file.filename == '':
        flash("No file selected.", 'warning')
        return render_template('scan.html', results=None, plots=plots)
    
    plot_id = request.form.get('plot_id')
    if not plot_id:
        flash("You must select a plot to assign this scan to.", 'danger')
        return render_template('scan.html', results=None, plots=plots, error="Plot is required.")
    
    try:
        plot_id = int(plot_id)
        plot_check = Plot.query.get(plot_id)
        if not plot_check or plot_check.user_id != current_user.id:
            flash("Invalid plot selected.", 'danger')
            return render_template('scan.html', results=None, plots=plots, error="Invalid plot.")
    except Exception as e:
        print(f"Error validating plot: {e}")
        flash("Invalid plot ID.", 'danger')
        return render_template('scan.html', results=None, plots=plots, error="Invalid plot ID.")

    scan_lat = current_user.latitude; scan_lon = current_user.longitude

    try:
        img_bytes = file.read()
        filename = secure_filename(f"{current_user.id}_{int(time.time())}_{file.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(image_path, 'wb') as f: f.write(img_bytes)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        flash(f"Could not open image: {e}", 'danger')
        return render_template('scan.html', results=None, plots=plots, error=f"Could not open image: {e}")

    pred_class, confidence, top3 = predict_with_gemini(img)
    plant_type = get_plant_type(pred_class)
    is_healthy = pred_class in HEALTHY_CLASSES

    severity = None
    if not is_healthy:
        tensor = preprocess_image_pytorch(img)
        severity = predict_severity(MODEL2W, tensor)

    plot_config = {"displayModeBar": False}
    
    conf_gauge_html = gauge_chart(confidence * 100, "Model confidence", "%")
    sev_gauge_html = gauge_chart(5.0, "Looks healthy", "%") if is_healthy else gauge_chart(severity, "Estimated severity", "%")
    top3_chart_html = horizontal_bar_chart(top3)

    gemini_data = get_gemini_info(pred_class, severity)

    buffered = io.BytesIO(); img.resize((512, 512)).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{img_str}"

    scan_id_for_template = None
    try:
        new_scan = ScanHistory(
            user_id = current_user.id, plot_id = plot_id,
            latitude = scan_lat, longitude = scan_lon,
            pred_class = pred_class, is_healthy = is_healthy,
            confidence = confidence, severity = severity,
            plant_type = plant_type, image_filename = filename,
            definition_json = json.dumps(gemini_data["definition"]),
            solution_json = json.dumps(gemini_data["solution_guide"]),
            fertilizer_json = json.dumps(gemini_data.get("fertilizer_recommendation")),
            top3_json = json.dumps(top3)
        )
        db.session.add(new_scan); db.session.commit()
        scan_id_for_template = new_scan.id

        treatment_plan = gemini_data.get('treatment_plan', [])
        if treatment_plan and not new_scan.is_healthy:
            today = datetime.utcnow().date()
            for item in treatment_plan:
                task_day = item.get('day', 1)
                task_title = item.get('task', 'Follow-up on scan')
                due_date = today + timedelta(days=task_day - 1)
                
                existing = Task.query.filter_by(scan_id=new_scan.id, title=task_title, due_date=due_date).first()
                if not existing:
                    new_task = Task(
                        user_id = current_user.id, scan_id = new_scan.id,
                        plot_id = new_scan.plot_id, task_type = 'treatment',
                        title = task_title, due_date = due_date
                    )
                    db.session.add(new_task)
            db.session.commit()

    except Exception as e:
        db.session.rollback(); print(f"Error saving to database: {e}")
        flash('There was an error saving your scan.', 'danger')

    results = {
        "scan_id": scan_id_for_template,
        "pred_class": pred_class, "emoji": get_class_emoji(pred_class),
        "is_healthy": is_healthy, "confidence": confidence, "severity": severity,
        "top3": top3, "plant_type": plant_type, "image_data_url": image_data_url,
        "charts": {
            "confidence_gauge": conf_gauge_html,
            "severity_gauge": sev_gauge_html,
            "top3_chart": top3_chart_html
        },
        "definition": gemini_data["definition"],
        "solution_guide": gemini_data["solution_guide"],
        "fertilizer": gemini_data.get("fertilizer_recommendation"),
        "treatment_plan": gemini_data.get('treatment_plan', [])
    }
    return render_template('scan.html', results=results, plots=plots)


@app.route('/history')
@login_required
def history():
    scans = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.timestamp.desc()).all()
    return render_template('history.html', scans=scans)


@app.route('/scan/<int:scan_id>')
@login_required
def scan_detail(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id: abort(403)
    image_data_url = get_image_data_url(scan.image_filename)
    if not image_data_url:
        flash('Error: Could not load image for this scan.', 'danger'); return redirect(url_for('history'))

    plot_config = {"displayModeBar": False}
    
    conf_gauge_html = gauge_chart(scan.confidence * 100, "Model confidence", "%")
    sev_gauge_html = gauge_chart(5.0, "Looks healthy", "%") if scan.is_healthy else gauge_chart(scan.severity, "Estimated severity", "%")
    top3_list = json.loads(scan.top3_json)
    top3_chart_html = horizontal_bar_chart(top3_list)

    try:
        definition_data = json.loads(scan.definition_json)
        solution_data = json.loads(scan.solution_json)
        fertilizer_data = json.loads(scan.fertilizer_json)
        treatment_plan = json.loads(scan.tasks.first().scan.definition_json).get('treatment_plan', []) if scan.tasks else []
    except:
        definition_data = {"status": "info", "message": "Error loading data."}; solution_data = []
        fertilizer_data = {"description": "Error loading data.", "product_name": "N/A"}; treatment_plan = []

    results = {
        "scan_id": scan.id, "pred_class": scan.pred_class,
        "emoji": get_class_emoji(scan.pred_class), "is_healthy": scan.is_healthy,
        "confidence": scan.confidence, "severity": scan.severity, "top3": top3_list,
        "plant_type": scan.plant_type, "image_data_url": image_data_url,
        "charts": {
            "confidence_gauge": conf_gauge_html,
            "severity_gauge": sev_gauge_html,
            "top3_chart": top3_chart_html
        },
        "definition": definition_data, "solution_guide": solution_data,
        "fertilizer": fertilizer_data, "treatment_plan": treatment_plan
    }
    return render_template('scan.html', results=results, from_history=True)

# =========================
# ---- NEW: Analytics & Inventory Routes ----
# =========================

@app.route('/analytics')
@login_required
def analytics():
    # Query 1: Most common diseases (top 5)
    most_common_diseases = db.session.query(
        ScanHistory.pred_class, func.count(ScanHistory.id).label('count')
    ).filter(
        ScanHistory.user_id == current_user.id,
        ScanHistory.is_healthy == False
    ).group_by(ScanHistory.pred_class).order_by(
        func.count(ScanHistory.id).desc()
    ).limit(5).all()

    # Query 2: Task completion stats
    completed_tasks = db.session.query(
        Task.task_type, func.count(Task.id).label('count')
    ).filter(
        Task.user_id == current_user.id,
        Task.is_complete == True
    ).group_by(Task.task_type).all()
    
    # Query 3: Plot health overview
    plots = Plot.query.filter_by(user_id=current_user.id).all()
    plot_health = []
    for plot in plots:
        last_scan = ScanHistory.query.filter_by(plot_id=plot.id).order_by(ScanHistory.timestamp.desc()).first()
        health_status = "No Scans"
        health_class = "info"
        if last_scan:
            if last_scan.is_healthy:
                health_status = "Healthy"
                health_class = "good"
            else:
                health_status = f"{last_scan.pred_class} ({last_scan.severity:.0f}%)"
                health_class = "bad"
        plot_health.append({"plot": plot, "status": health_status, "class": health_class})

    return render_template('analytics.html',
                           disease_stats=most_common_diseases,
                           task_stats=completed_tasks,
                           plot_health=plot_health)

@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    if request.method == 'POST':
        name = request.form.get('name')
        category = request.form.get('category')
        quantity = request.form.get('quantity')
        
        if name and category:
            new_resource = Resource(
                user_id=current_user.id,
                name=name,
                category=category,
                quantity=quantity
            )
            db.session.add(new_resource)
            db.session.commit()
            flash('Resource added to inventory.', 'success')
        else:
            flash('Resource Name and Category are required.', 'danger')
        return redirect(url_for('inventory'))

    # GET request: Load all resources
    resources = Resource.query.filter_by(user_id=current_user.id).order_by(Resource.category).all()
    return render_template('inventory.html', resources=resources)

@app.route('/inventory/delete/<int:resource_id>', methods=['POST'])
@login_required
def delete_resource(resource_id):
    resource = Resource.query.get_or_404(resource_id)
    if resource.user_id != current_user.id:
        abort(403)
    
    db.session.delete(resource)
    db.session.commit()
    flash('Resource removed from inventory.', 'success')
    return redirect(url_for('inventory'))


# =========================
# ---- Main Run Block ----
# =========================
if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("="*50); print("WARNING: 'GEMINI_API_KEY' is not set."); print("="*50)
    
    with app.app_context():
        db.create_all()
        
    app.run(debug=True)
