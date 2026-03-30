"""
PyWeather AI — Streamlit Frontend
Intel Unnati Generative AI — VIUP122 Major Project
End-to-end AI system: Real-time Weather Intelligence + ML Severity Prediction
Architecture: Frontend (Streamlit) → Backend API (FastAPI) → ML Model (Random Forest)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PyWeather AI | Intel Unnati",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: linear-gradient(135deg, #060b18 0%, #0d1b2a 50%, #0a0e1a 100%); }

.hero-banner {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(116,185,255,0.2); border-radius: 20px;
    padding: 28px 36px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.hero-title {
    font-size: 2.6em; font-weight: 900; margin: 0 0 4px 0;
    background: linear-gradient(135deg, #ffffff, #74b9ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px; line-height: 1.1;
}
.hero-sub { color: #74b9ff; font-size: 0.82em; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 10px 0; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.badge { background: rgba(116,185,255,0.1); border: 1px solid rgba(116,185,255,0.25); color: #74b9ff; padding: 4px 14px; border-radius: 20px; font-size: 0.72em; font-weight: 700; }
.badge-green { background: rgba(0,184,148,0.1); border-color: rgba(0,184,148,0.3); color: #55efc4; }
.badge-orange { background: rgba(253,203,110,0.1); border-color: rgba(253,203,110,0.3); color: #fdcb6e; }

.metric-card {
    background: linear-gradient(145deg, rgba(28,28,58,0.9), rgba(22,33,62,0.9));
    border: 1px solid rgba(116,185,255,0.12); border-radius: 18px; padding: 22px 26px; margin: 8px 0;
    backdrop-filter: blur(10px); transition: border-color 0.3s, transform 0.2s;
}
.metric-card:hover { border-color: rgba(116,185,255,0.28); transform: translateY(-1px); }

.temp-display {
    font-size: 6em; font-weight: 900;
    background: linear-gradient(135deg, #a8d8ff, #0984e3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 0; letter-spacing: -4px;
}

.section-header {
    font-size: 0.7em; font-weight: 800; color: #74b9ff;
    text-transform: uppercase; letter-spacing: 3px;
    margin: 24px 0 14px 0; padding-bottom: 10px;
    border-bottom: 1px solid rgba(116,185,255,0.15);
}

.ai-box {
    background: linear-gradient(135deg, rgba(116,185,255,0.07), rgba(9,132,227,0.03));
    border: 1px solid rgba(116,185,255,0.22); border-left: 4px solid #74b9ff;
    border-radius: 0 16px 16px 0; padding: 20px 26px; margin: 14px 0;
}
.ai-label { font-size: 0.65em; font-weight: 800; color: #74b9ff; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 12px; }
.ml-box {
    background: linear-gradient(135deg, rgba(85,239,196,0.07), rgba(0,184,148,0.03));
    border: 1px solid rgba(85,239,196,0.22); border-left: 4px solid #55efc4;
    border-radius: 0 16px 16px 0; padding: 20px 26px; margin: 14px 0;
}

.insight-card {
    background: rgba(116,185,255,0.04); border: 1px solid rgba(116,185,255,0.1);
    border-left: 3px solid #74b9ff; border-radius: 0 14px 14px 0;
    padding: 14px 22px; margin: 8px 0; font-size: 0.9em; color: #dfe6e9; line-height: 1.7;
    transition: background 0.2s;
}
.insight-card:hover { background: rgba(116,185,255,0.07); }

.tip-wear { background: linear-gradient(135deg, rgba(41,128,185,0.14), rgba(41,128,185,0.04)); border: 1px solid rgba(41,128,185,0.28); border-radius: 18px; padding: 24px 28px; margin: 10px 0; }
.tip-activity { background: linear-gradient(135deg, rgba(39,174,96,0.14), rgba(39,174,96,0.04)); border: 1px solid rgba(39,174,96,0.28); border-radius: 18px; padding: 24px 28px; margin: 10px 0; }
.tip-health { background: linear-gradient(135deg, rgba(243,156,18,0.14), rgba(243,156,18,0.04)); border: 1px solid rgba(243,156,18,0.28); border-radius: 18px; padding: 24px 28px; margin: 10px 0; }
.tip-title { font-size: 0.75em; font-weight: 800; text-transform: uppercase; letter-spacing: 2.5px; margin: 0 0 12px 0; }
.tip-body { color: #dfe6e9; line-height: 1.75; font-size: 0.95em; margin: 0; }

.glance-card {
    background: linear-gradient(145deg, rgba(28,28,58,0.9), rgba(22,33,62,0.9));
    border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
    padding: 20px 16px; text-align: center; margin: 6px 0; transition: border-color 0.2s;
}
.glance-card:hover { border-color: rgba(116,185,255,0.2); }

.forecast-row {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 14px 22px; margin: 5px 0; transition: background 0.2s, border-color 0.2s;
}
.forecast-row:hover { background: rgba(116,185,255,0.05); border-color: rgba(116,185,255,0.15); }

.backend-on { background: rgba(0,184,148,0.12); border: 1px solid rgba(0,184,148,0.3); color: #55efc4; padding: 6px 16px; border-radius: 20px; font-size: 0.74em; font-weight: 700; display: inline-block; }
.backend-off { background: rgba(255,118,117,0.12); border: 1px solid rgba(255,118,117,0.3); color: #ff7675; padding: 6px 16px; border-radius: 20px; font-size: 0.74em; font-weight: 700; display: inline-block; }

.project-info { background: linear-gradient(135deg, rgba(116,185,255,0.06), rgba(9,132,227,0.03)); border: 1px solid rgba(116,185,255,0.15); border-radius: 14px; padding: 16px 20px; margin: 12px 0; }
.project-info p { margin: 4px 0; font-size: 0.8em; color: #b2bec3; line-height: 1.8; }
.project-info strong { color: #74b9ff; }

.stButton > button { border-radius: 12px !important; font-weight: 700 !important; font-size: 0.84em !important; padding: 9px 22px !important; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #060b18 0%, #0d1424 100%) !important; border-right: 1px solid rgba(116,185,255,0.08) !important; }
section[data-testid="stSidebar"] .stButton > button { background: rgba(116,185,255,0.06) !important; border: 1px solid rgba(116,185,255,0.15) !important; color: #b2bec3 !important; width: 100%; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(116,185,255,0.14) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(255,255,255,0.025); border-radius: 14px; padding: 5px; border: 1px solid rgba(255,255,255,0.05); }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; padding: 9px 20px !important; font-weight: 600 !important; font-size: 0.86em !important; color: #636e72 !important; }
.stTabs [aria-selected="true"] { background: rgba(116,185,255,0.15) !important; color: #74b9ff !important; }
div[data-testid="stMetric"] { background: rgba(116,185,255,0.05); border: 1px solid rgba(116,185,255,0.1); border-radius: 12px; padding: 16px !important; }
div[data-testid="stMetric"] label { color: #636e72 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #74b9ff !important; font-weight: 700 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════

WMO_CODES = {
    0:("Clear Sky","☀️"), 1:("Mainly Clear","🌤️"), 2:("Partly Cloudy","⛅"),
    3:("Overcast","☁️"), 45:("Foggy","🌫️"), 48:("Rime Fog","🌫️"),
    51:("Light Drizzle","💧"), 53:("Drizzle","💧"), 55:("Dense Drizzle","💧"),
    61:("Light Rain","🌧️"), 63:("Rain","🌧️"), 65:("Heavy Rain","🌧️"),
    71:("Light Snow","❄️"), 73:("Snow","❄️"), 75:("Heavy Snow","❄️"),
    80:("Showers","🌦️"), 81:("Showers","🌦️"), 82:("Violent Showers","🌦️"),
    95:("Thunderstorm","⛈️"), 96:("Thunderstorm+Hail","⛈️"), 99:("Severe Storm","⛈️"),
}

# FIX: CHART_BASE no longer includes 'margin' — each chart passes its own margin
# to avoid Python's "duplicate keyword argument" TypeError.
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b2bec3", family="Inter"),
)


def get_aqi_info(aqi):
    if aqi <= 50:    return "Good",                           "#00b894", "🟢"
    elif aqi <= 100: return "Moderate",                       "#fdcb6e", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#e17055", "🟠"
    elif aqi <= 200: return "Unhealthy",                      "#d63031", "🔴"
    elif aqi <= 300: return "Very Unhealthy",                 "#6c5ce7", "🟣"
    else:            return "Hazardous",                      "#2d3436", "⚫"


def rule_based_severity(weather_code, temp, aqi, rain_chance, wind_speed):
    labels = ["safe and pleasant","mild caution advised","severe weather warning","extreme danger"]
    s = [0.0, 0.0, 0.0, 0.0]
    if weather_code in (99,) or temp > 42 or aqi > 300 or wind_speed > 90:
        s[3]=0.80; s[2]=0.12; s[1]=0.05; s[0]=0.03
    elif weather_code in (95,96) or temp > 38 or aqi > 200 or rain_chance > 85 or wind_speed > 60:
        s[2]=0.75; s[1]=0.15; s[3]=0.06; s[0]=0.04
    elif (weather_code in (61,63,65,71,73,75,80,81,82)
          or temp > 32 or aqi > 100 or rain_chance > 50 or wind_speed > 35):
        s[1]=0.72; s[0]=0.18; s[2]=0.07; s[3]=0.03
    else:
        s[0]=0.82; s[1]=0.12; s[2]=0.04; s[3]=0.02
    best = s.index(max(s))
    return {"label": labels[best], "score": s[best], "all": list(zip(labels, s))}


def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def fetch_weather_via_backend(city, units):
    try:
        r = requests.post(f"{BACKEND_URL}/weather-data",
                          json={"city": city, "units": units}, timeout=15)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return None, False


def fetch_weather_direct(city, units):
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1}, timeout=10).json()
        if "results" not in geo or not geo["results"]:
            st.error(f"❌ City not found: **{city}**")
            return None, None, None
        loc = geo["results"][0]
        lat, lon, tz = loc["latitude"], loc["longitude"], loc["timezone"]
        unit_temp = "celsius" if units == "metric" else "fahrenheit"
        unit_wind = "kmh"     if units == "metric" else "mph"
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": tz,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily":   "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "hourly":  "temperature_2m,precipitation_probability,relative_humidity_2m",
            "temperature_unit": unit_temp, "wind_speed_unit": unit_wind, "forecast_days": 10
        }, timeout=10).json()
        a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                         params={"latitude": lat, "longitude": lon, "current": "us_aqi"},
                         timeout=10).json()
        return w, a, loc
    except Exception as e:
        st.error(f"API Error: {e}")
        return None, None, None


def process_forecast(daily):
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df["day"]  = df["date"].dt.strftime("%a, %b %d")
    return df


def process_hourly(hourly):
    df = pd.DataFrame(hourly)
    df["Timestamp"] = pd.to_datetime(df["time"])
    df.rename(columns={"temperature_2m": "Temperature",
                        "relative_humidity_2m": "Humidity",
                        "precipitation_probability": "Rain Chance (%)"}, inplace=True)
    return df.set_index("Timestamp")


def smart_summary(condition, temp, humidity, rain_chance, wind, wind_unit, aqi, city, unit_symbol):
    feel = ("hot and humid"        if temp > 32 and humidity > 70 else
            "warm and pleasant"    if temp > 25 else
            "cool and comfortable" if temp > 15 else
            "cold"                 if temp > 5  else "freezing")
    rain_desc = ("high chance of rain — carry an umbrella" if rain_chance > 60 else
                 "possible showers later" if rain_chance > 30 else "mostly dry and clear")
    aqi_label, _, _ = get_aqi_info(aqi)
    aqi_note = (f" Air quality is {aqi_label.lower()} — limit prolonged outdoor exposure." if aqi > 150
                else f" Air quality is {aqi_label.lower()}.")
    return (
        f"{city} is currently experiencing <strong>{condition.lower()}</strong> conditions "
        f"with a temperature of <strong>{temp:.0f}{unit_symbol}</strong>, feeling {feel}. "
        f"Humidity at {humidity}% with winds of {wind} {wind_unit}. "
        f"Rain probability: <strong>{rain_chance}%</strong> — {rain_desc}.{aqi_note}"
    )


def smart_tips(condition, temp, aqi, rain_chance):
    cond = condition.lower()
    if "storm" in cond or "thunder" in cond:
        wear     = "Stay indoors if possible. Heavy waterproof gear essential if going out. Avoid tall trees and metal objects."
        activity = "Avoid all outdoor activities. Great day for movies, reading, or productive indoor work."
    elif "rain" in cond or "drizzle" in cond or "shower" in cond:
        wear     = "Waterproof jacket and umbrella are a must. Water-resistant footwear strongly recommended."
        activity = "Outdoor plans may be disrupted. Consider a café, museum, or indoor workout instead."
    elif "snow" in cond:
        wear     = "Heavy thermal coat, waterproof boots, gloves, and a warm hat are all essential."
        activity = "Snow sports are great if you enjoy them — otherwise cozy up indoors with a warm drink."
    elif temp > 35:
        wear     = "Light, breathable clothing in pale colors. Wide-brimmed hat and sunglasses strongly recommended."
        activity = "Avoid strenuous outdoor activity between 11am–4pm. Morning/evening walks or swimming are ideal."
    elif temp > 25:
        wear     = "Light cotton or linen. A T-shirt and shorts or light trousers will be very comfortable."
        activity = "Perfect for picnics, cycling, jogging, or any outdoor sport. Enjoy the weather!"
    elif temp > 15:
        wear     = "Comfortable casuals with a light jacket ready for the cooler evening."
        activity = "Good conditions for outdoor walks, sports, or alfresco dining."
    elif temp > 5:
        wear     = "Warm coat, scarf, and thermal underlayers for extended outdoor time."
        activity = "Brisk walks are still fine. Limit extended outdoor exposure."
    else:
        wear     = "Full winter gear: heavy coat, thermals, gloves, warm hat, and layered socks."
        activity = "Minimise time outdoors. Stay warm and safe."

    health = []
    if aqi > 200:    health.append("⚠️ Very unhealthy air — wear an N95 mask outdoors and keep windows closed.")
    elif aqi > 150:  health.append("⚠️ Unhealthy air — sensitive groups should stay inside.")
    elif aqi > 100:  health.append("ℹ️ Moderate air — children and elderly should limit prolonged outdoor exposure.")
    else:            health.append("✅ Air quality is acceptable for most people today.")
    if temp > 35:    health.append("💧 Drink at least 3–4 litres of water to prevent heat exhaustion.")
    if rain_chance > 50: health.append("🥾 Wet conditions — wear non-slip footwear.")
    return wear, activity, " ".join(health)


def analyze_patterns(daily_df, unit_symbol):
    hi   = daily_df["temperature_2m_max"].tolist()
    lo   = daily_df["temperature_2m_min"].tolist()
    rain = daily_df["precipitation_probability_max"].tolist()
    avg_max    = sum(hi)   / len(hi)
    avg_min    = sum(lo)   / len(lo)
    avg_rain   = sum(rain) / len(rain)
    max_temp   = max(hi);   min_temp = min(lo)
    rainy_days = sum(1 for r in rain if r > 50)
    temp_swing = max_temp - min_temp
    insights   = []
    if avg_rain > 60:   insights.append(("🌧️","High Rainfall Week","Persistent rain expected — keep an umbrella all week."))
    elif avg_rain > 30: insights.append(("🌦️","Occasional Showers","Intermittent rain possible — check forecasts before heading out."))
    else:               insights.append(("☀️","Mostly Dry Week","Great week for outdoor plans with minimal rain expected."))
    if temp_swing > 12: insights.append(("🌡️","Large Temperature Swing",f"A {temp_swing:.0f}{unit_symbol} variation this week — dress in layers."))
    elif temp_swing > 6: insights.append(("🌡️","Moderate Temp Change",f"Expect a {temp_swing:.0f}{unit_symbol} variation — light layers recommended."))
    else:               insights.append(("🌡️","Stable Temperatures","Consistent temperatures throughout the week."))
    if max_temp > 38:   insights.append(("🔥","Extreme Heat Warning","Dangerous heat expected — stay indoors and hydrate frequently."))
    elif max_temp > 32: insights.append(("☀️","Hot Days Ahead","Stay hydrated and wear sunscreen on warm days."))
    if min_temp < 2:    insights.append(("🧊","Near-Freezing Nights","Protect exposed pipes and wear heavy winter clothing in evenings."))
    elif min_temp < 10: insights.append(("🥶","Cool Nights","Cold nights ahead — keep warm clothing ready after sunset."))
    if (10 - rainy_days) >= 8: insights.append(("🌈","Excellent Outdoor Week",f"{10-rainy_days} dry days forecast — perfect for travel and outdoor activities."))
    elif rainy_days >= 7: insights.append(("☔","Very Rainy Week",f"Rain expected on {rainy_days}/10 days — plan indoor alternatives."))
    return {"avg_max": avg_max, "avg_min": avg_min, "avg_rain": avg_rain,
            "max_temp": max_temp, "min_temp": min_temp,
            "rainy_days": rainy_days, "temp_swing": temp_swing, "insights": insights}


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 12px 0;'>
        <div style='font-size:3.2em; margin-bottom:8px;'>🌤️</div>
        <h2 style='margin:0 0 4px 0; color:#ffffff; font-size:1.35em; font-weight:900; letter-spacing:-0.5px;'>PyWeather AI</h2>
        <p style='color:#74b9ff; font-size:0.7em; margin:0 0 12px 0; letter-spacing:2px; text-transform:uppercase; font-weight:700;'>Intelligent Weather Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    backend_ok, _ = check_backend()
    if backend_ok:
        st.markdown('<div class="backend-on">⚡ Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="backend-off">📡 Direct API Mode</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#74b9ff; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin:0 0 8px 0;'>🔍 Search City</p>", unsafe_allow_html=True)
    city = st.text_input("", "Ghaziabad", placeholder="Enter any city name...", label_visibility="collapsed")

    def reset_cache():
        for k in ["ai_summary","ai_severity","ai_tips","ai_analysis",
                  "weather_data","air_data","location","ml_prediction"]:
            if k in st.session_state: del st.session_state[k]

    if st.button("🔍  Search", use_container_width=True, type="primary"):
        st.session_state.city = city
        reset_cache()

    st.markdown("<p style='color:#636e72; font-size:0.72em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:16px 0 8px 0;'>📌 Quick Cities</p>", unsafe_allow_html=True)
    if "saved_cities" not in st.session_state:
        st.session_state.saved_cities = ["Delhi","Mumbai","London","New York","Tokyo"]

    cols_c = st.columns(2)
    for idx, sc in enumerate(st.session_state.saved_cities):
        with cols_c[idx % 2]:
            if st.button(sc, use_container_width=True, key=f"qc_{sc}"):
                st.session_state.city = sc
                reset_cache()

    if "city" not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.markdown("---")
    st.markdown("<p style='color:#74b9ff; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin:0 0 10px 0;'>⚙️ Units</p>", unsafe_allow_html=True)
    unit_options  = {"🌡️ Celsius (°C)": "metric", "🌡️ Fahrenheit (°F)": "imperial"}
    selected_unit = unit_options[st.radio("", list(unit_options.keys()), label_visibility="collapsed")]
    unit_symbol   = "°C" if selected_unit == "metric" else "°F"

    st.markdown("---")
    st.markdown("""
    <div class='project-info'>
        <p><strong>📚 Project:</strong> Intel Unnati Gen AI</p>
        <p><strong>📋 Course:</strong> VIUP122</p>
        <p><strong>🧠 Model:</strong> Random Forest (97.3%)</p>
        <p><strong>🌐 API:</strong> Open-Meteo (Free, No Key)</p>
        <p><strong>⚡ Backend:</strong> FastAPI</p>
        <p><strong>🎨 Frontend:</strong> Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════

st.markdown(f"""
<div class='hero-banner'>
    <p class='hero-sub'>Intel Unnati Generative AI · VIUP122 Major Project</p>
    <h1 class='hero-title'>🌤️ PyWeather AI</h1>
    <p style='color:#b2bec3; margin:6px 0 0 0; font-size:0.95em; line-height:1.6;'>
        Real-time weather intelligence powered by a
        <strong style='color:#74b9ff;'>Random Forest ML model</strong>
        &nbsp;·&nbsp; Live data via Open-Meteo API &nbsp;·&nbsp; Full end-to-end AI system
    </p>
    <div class='hero-badges'>
        <span class='badge'>🤖 Random Forest Classifier</span>
        <span class='badge badge-green'>✅ 97.3% Accuracy</span>
        <span class='badge badge-orange'>📡 Live Weather API</span>
        <span class='badge'>⚡ FastAPI Backend</span>
        <span class='badge'>🚀 Memory-Safe Deployment</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════

weather_data = air_data = location = ml_prediction = None
using_backend = False

if st.session_state.city:
    if backend_ok:
        result, ok = fetch_weather_via_backend(st.session_state.city, selected_unit)
        if ok and result:
            location     = result["location"]
            air_data     = {"current": result["air_quality"]}
            ml_prediction = result.get("ml_prediction")
            weather_data = {"current": result["current"],
                            "daily":   result["daily"],
                            "hourly":  result["hourly"]}
            using_backend = True

    if not using_backend:
        weather_data, air_data, location = fetch_weather_direct(
            st.session_state.city, selected_unit
        )

    if weather_data and air_data and location:
        st.session_state.weather_data  = weather_data
        st.session_state.air_data      = air_data
        st.session_state.unit_symbol   = unit_symbol
        st.session_state.selected_unit = selected_unit
        st.session_state.location      = location
        st.session_state.ml_prediction = ml_prediction
        city_name = location["name"]
        if city_name not in st.session_state.saved_cities:
            if len(st.session_state.saved_cities) >= 8:
                st.session_state.saved_cities.pop(0)
            st.session_state.saved_cities.append(city_name)


# ══════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════

if "weather_data" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding:80px 20px; color:#636e72;'>
        <div style='font-size:5em; margin-bottom:16px;'>🌍</div>
        <h2 style='color:#74b9ff; font-size:1.8em; font-weight:800; margin:0 0 12px 0;'>Welcome to PyWeather AI</h2>
        <p style='font-size:1em; color:#b2bec3; max-width:480px; margin:0 auto; line-height:1.7;'>
            Search for any city using the sidebar to get real-time weather data,
            AI-powered insights, and ML-based severity predictions.
        </p>
        <div style='margin-top:32px; display:flex; gap:16px; justify-content:center; flex-wrap:wrap;'>
            <div style='background:rgba(116,185,255,0.06); border:1px solid rgba(116,185,255,0.15); border-radius:16px; padding:18px 24px; min-width:160px;'>
                <p style='font-size:2em; margin:0;'>🧠</p>
                <p style='color:#74b9ff; font-size:0.78em; font-weight:700; margin:8px 0 3px 0; text-transform:uppercase; letter-spacing:1px;'>ML Prediction</p>
                <p style='color:#636e72; font-size:0.78em; margin:0;'>Random Forest · 97.3%</p>
            </div>
            <div style='background:rgba(85,239,196,0.06); border:1px solid rgba(85,239,196,0.15); border-radius:16px; padding:18px 24px; min-width:160px;'>
                <p style='font-size:2em; margin:0;'>📡</p>
                <p style='color:#55efc4; font-size:0.78em; font-weight:700; margin:8px 0 3px 0; text-transform:uppercase; letter-spacing:1px;'>Live Data</p>
                <p style='color:#636e72; font-size:0.78em; margin:0;'>Open-Meteo Free API</p>
            </div>
            <div style='background:rgba(253,203,110,0.06); border:1px solid rgba(253,203,110,0.15); border-radius:16px; padding:18px 24px; min-width:160px;'>
                <p style='font-size:2em; margin:0;'>📊</p>
                <p style='color:#fdcb6e; font-size:0.78em; font-weight:700; margin:8px 0 3px 0; text-transform:uppercase; letter-spacing:1px;'>10-Day Forecast</p>
                <p style='color:#636e72; font-size:0.78em; margin:0;'>Patterns & Insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── UNPACK ──
    weather_data  = st.session_state.weather_data
    air_data      = st.session_state.air_data
    unit_symbol   = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location      = st.session_state.location
    ml_prediction = st.session_state.get("ml_prediction")

    current    = weather_data["current"]
    daily_df   = process_forecast(weather_data["daily"])
    hourly_df  = process_hourly(weather_data["hourly"])
    aqi        = air_data["current"]["us_aqi"]
    aqi_label, aqi_color, aqi_emoji = get_aqi_info(aqi)
    desc, icon = WMO_CODES.get(current["weather_code"], ("Unknown","❓"))
    wind_label = "km/h" if selected_unit == "metric" else "mph"
    rain_today = daily_df.iloc[0]["precipitation_probability_max"]
    country    = location.get("country","")

    st.markdown(
        f"<p style='color:#74b9ff; margin:0 0 16px 0; font-size:0.88em; font-weight:600;'>"
        f"📍 {location['name']}{', '+country if country else ''}"
        f" &nbsp;·&nbsp; Updated {datetime.now().strftime('%H:%M')}</p>",
        unsafe_allow_html=True
    )

    # ═══ TABS ═══
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠  Current", "📅  10-Day Forecast",
        "🤖  AI Summary", "🧠  ML Prediction",
        "📊  Pattern Analysis", "💡  Smart Tips"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — CURRENT WEATHER
    # ══════════════════════════════════════════════════════════
    with tab1:
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='temp-display'>{current['temperature_2m']:.0f}{unit_symbol}</p>
                <p style='font-size:1.5em; margin:10px 0 6px 0; font-weight:600;'>{icon} {desc}</p>
                <p style='color:#636e72; margin:0; font-size:0.88em;'>
                    ↑ <strong style='color:#ff7675;'>{daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol}</strong>
                    &nbsp;&nbsp;
                    ↓ <strong style='color:#74b9ff;'>{daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}</strong>
                </p>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='metric-card' style='height:100%;'>
                <p class='section-header'>📊 Live Conditions</p>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:18px;'>
                    <div>
                        <p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0;'>Humidity</p>
                        <p style='font-size:1.5em; font-weight:700; margin:3px 0;'>{current['relative_humidity_2m']}%</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0;'>Wind Speed</p>
                        <p style='font-size:1.5em; font-weight:700; margin:3px 0;'>{current['wind_speed_10m']} {wind_label}</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0;'>Precipitation</p>
                        <p style='font-size:1.5em; font-weight:700; margin:3px 0;'>{current['precipitation']:.1f} mm</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0;'>Rain Chance</p>
                        <p style='font-size:1.5em; font-weight:700; margin:3px 0;'>{rain_today}%</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card' style='margin-top:12px;'>
            <p class='section-header'>🌿 Air Quality Index</p>
            <div style='display:flex; align-items:center; gap:22px;'>
                <p style='font-size:3.2em; margin:0;'>{aqi_emoji}</p>
                <div style='flex:1;'>
                    <p style='font-size:1.7em; font-weight:800; color:{aqi_color}; margin:0 0 4px 0;'>{aqi_label}</p>
                    <p style='color:#636e72; margin:0; font-size:0.85em;'>
                        US AQI: <strong style='color:#dfe6e9;'>{aqi}</strong>
                        &nbsp;·&nbsp; 0–50 Good · 51–100 Moderate · 101–150 Sensitive · 151+ Unhealthy
                    </p>
                </div>
                <div style='text-align:right; min-width:160px;'>
                    <div style='background:rgba(255,255,255,0.05); border-radius:8px; height:8px; overflow:hidden;'>
                        <div style='background:{aqi_color}; width:{min(aqi/300*100,100):.0f}%; height:100%; border-radius:8px;'></div>
                    </div>
                    <p style='color:#636e72; font-size:0.75em; margin:4px 0 0 0;'>{aqi} / 300</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>📆 3-Day Outlook</p>", unsafe_allow_html=True)
        d_cols = st.columns(3, gap="small")
        for i, (_, row) in enumerate(daily_df.head(3).iterrows()):
            d, ic = WMO_CODES.get(row["weather_code"], ("Unknown","❓"))
            rp = row["precipitation_probability_max"]
            with d_cols[i]:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center; padding:20px 16px;'>
                    <p style='color:#74b9ff; font-size:0.78em; font-weight:700; margin:0;'>{row['day']}</p>
                    <p style='font-size:2.2em; margin:8px 0 4px 0;'>{ic}</p>
                    <p style='font-size:0.82em; margin:0 0 8px 0; color:#b2bec3;'>{d}</p>
                    <p style='margin:0 0 8px 0;'>
                        <span style='color:#ff7675; font-weight:700;'>{row['temperature_2m_max']:.0f}{unit_symbol}</span>
                        <span style='color:#4a5568;'> / </span>
                        <span style='color:#74b9ff; font-weight:700;'>{row['temperature_2m_min']:.0f}{unit_symbol}</span>
                    </p>
                    <div style='background:rgba(116,185,255,0.1); border-radius:6px; height:5px; overflow:hidden;'>
                        <div style='background:#74b9ff; width:{int(rp)}%; height:100%; border-radius:6px;'></div>
                    </div>
                    <p style='color:#636e72; font-size:0.75em; margin:4px 0 0 0;'>💧 {rp}% rain</p>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — 10-DAY FORECAST
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<p class='section-header'>📅 10-Day Daily Forecast</p>", unsafe_allow_html=True)
        for _, row in daily_df.iterrows():
            d, ic  = WMO_CODES.get(row["weather_code"], ("Unknown","❓"))
            rain   = row["precipitation_probability_max"]
            rcolor = "#ff7675" if rain > 70 else "#fdcb6e" if rain > 40 else "#74b9ff"
            st.markdown(f"""
            <div class='forecast-row'>
                <div style='display:flex; align-items:center; gap:16px;'>
                    <div style='width:116px; color:#b2bec3; font-size:0.88em; font-weight:500;'>{row['day']}</div>
                    <div style='width:32px; font-size:1.3em; text-align:center;'>{ic}</div>
                    <div style='flex:1; color:#dfe6e9; font-size:0.86em;'>{d}</div>
                    <div style='width:90px; text-align:right; font-size:0.9em;'>
                        <span style='color:#ff7675; font-weight:700;'>{row['temperature_2m_max']:.0f}{unit_symbol}</span>
                        <span style='color:#4a5568;'> / </span>
                        <span style='color:#74b9ff; font-weight:700;'>{row['temperature_2m_min']:.0f}{unit_symbol}</span>
                    </div>
                    <div style='width:110px;'>
                        <div style='background:rgba(255,255,255,0.05); border-radius:4px; height:6px; overflow:hidden;'>
                            <div style='background:{rcolor}; width:{int(rain)}%; height:100%; border-radius:4px;'></div>
                        </div>
                        <p style='color:#636e72; font-size:0.73em; margin:3px 0 0 0; text-align:right;'>💧 {rain}%</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><p class='section-header'>📈 Hourly Trends</p>", unsafe_allow_html=True)
        hc1, hc2 = st.columns([1, 3])
        with hc1:
            plot_choice = st.selectbox("", ["Temperature","Rain Chance (%)","Humidity"], label_visibility="collapsed")
        cmap  = {"Temperature":"#ff7675","Rain Chance (%)":"#74b9ff","Humidity":"#55efc4"}
        hr    = hourly_df.reset_index()
        fig_h = px.area(hr, x="Timestamp", y=plot_choice, color_discrete_sequence=[cmap[plot_choice]])
        fig_h.update_traces(opacity=0.78, line=dict(width=2))
        fig_h.update_layout(**CHART_BASE, height=300, showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis=dict(showgrid=False, color="#636e72", title=""),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                       color="#636e72", title=plot_choice))
        st.plotly_chart(fig_h, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — AI SUMMARY
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<p class='section-header'>🤖 AI Weather Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.88em; margin:0 0 16px 0;'>Rule-based intelligent analysis — lightweight, accurate, no model download required.</p>", unsafe_allow_html=True)

        ac1, ac2 = st.columns([6, 1])
        with ac2:
            if st.button("🔄 Refresh", key="ref_sum"):
                for k in ["ai_summary","ai_severity"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

        if not st.session_state.get("ai_summary"):
            st.session_state.ai_summary = smart_summary(
                desc, current["temperature_2m"], current["relative_humidity_2m"],
                rain_today, current["wind_speed_10m"], wind_label, aqi, location["name"], unit_symbol
            )

        st.markdown(f"""
        <div class='ai-box'>
            <p class='ai-label'>✦ &nbsp; AI WEATHER ANALYSIS</p>
            <p style='margin:0; line-height:1.8; color:#dfe6e9; font-size:0.97em;'>
                {st.session_state.ai_summary}
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>🚦 Severity Classification</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.85em; margin:0 0 14px 0;'>Meteorological rule engine classifying conditions across 4 severity levels.</p>", unsafe_allow_html=True)

        if not st.session_state.get("ai_severity"):
            st.session_state.ai_severity = rule_based_severity(
                current["weather_code"], current["temperature_2m"],
                aqi, rain_today, current["wind_speed_10m"]
            )

        sev   = st.session_state.ai_severity
        slbl  = sev.get("label","unknown")
        score = sev.get("score", 0)
        sc    = {"safe and pleasant":"#00b894","mild caution advised":"#fdcb6e",
                 "severe weather warning":"#e17055","extreme danger":"#d63031"}.get(slbl,"#74b9ff")
        se    = {"safe and pleasant":"✅","mild caution advised":"⚠️",
                 "severe weather warning":"🚨","extreme danger":"🆘"}.get(slbl,"🔍")
        sd    = {"safe and pleasant":"Conditions are comfortable and safe for all activities.",
                 "mild caution advised":"Exercise some caution, especially for sensitive groups.",
                 "severe weather warning":"Limit outdoor exposure — conditions are significantly impacted.",
                 "extreme danger":"Stay indoors — dangerous conditions posing serious risk."}.get(slbl,"")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{sc}18,{sc}05);
                    border:1px solid {sc}45; border-left:4px solid {sc};
                    border-radius:0 18px 18px 0; padding:24px 28px; margin:14px 0;
                    display:flex; align-items:center; gap:22px;'>
            <p style='font-size:3.2em; margin:0;'>{se}</p>
            <div style='flex:1;'>
                <p style='font-size:0.68em; font-weight:800; color:{sc}; text-transform:uppercase; letter-spacing:3px; margin:0 0 6px 0;'>Severity Classification</p>
                <p style='font-size:1.6em; font-weight:800; color:{sc}; margin:0 0 4px 0;'>{slbl.title()}</p>
                <p style='color:#b2bec3; margin:0; font-size:0.88em; line-height:1.5;'>{sd}</p>
            </div>
            <div style='text-align:right;'>
                <p style='color:#636e72; font-size:0.72em; margin:0 0 4px 0; text-transform:uppercase; font-weight:700;'>Confidence</p>
                <p style='font-size:1.8em; font-weight:800; color:{sc}; margin:0;'>{score*100:.0f}%</p>
            </div>
        </div>""", unsafe_allow_html=True)

        if sev.get("all"):
            sdf = pd.DataFrame({"Category":[x[0].title() for x in sev["all"]],
                                 "Confidence":[round(x[1]*100,1) for x in sev["all"]]})
            cmap2 = {"Safe And Pleasant":"#00b894","Mild Caution Advised":"#fdcb6e",
                     "Severe Weather Warning":"#e17055","Extreme Danger":"#d63031"}
            sdf["Color"] = sdf["Category"].map(cmap2).fillna("#74b9ff")
            fig_s = px.bar(sdf, x="Confidence", y="Category", orientation="h",
                           color="Category",
                           color_discrete_map=dict(zip(sdf["Category"], sdf["Color"])),
                           text=sdf["Confidence"].apply(lambda v: f"{v}%"))
            fig_s.update_traces(textposition="outside", marker_line_width=0)
            fig_s.update_layout(**CHART_BASE, height=190, showlegend=False,
                                margin=dict(l=0, r=70, t=10, b=0),
                                xaxis=dict(showgrid=False, visible=False, title=""),
                                yaxis=dict(showgrid=False, color="#b2bec3", title=""))
            st.plotly_chart(fig_s, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — ML PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("<p class='section-header'>🧠 Random Forest ML Prediction</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.88em; margin:0 0 16px 0;'>Trained on <strong style='color:#dfe6e9;'>5,000 synthetic samples</strong> · <strong style='color:#55efc4;'>97.3% accuracy</strong> · 6 meteorological features · 5 weather classes</p>", unsafe_allow_html=True)

        if ml_prediction:
            pred  = ml_prediction
            pc    = {"Clear":"#fdcb6e","Cloudy":"#74b9ff","Rainy":"#0984e3","Stormy":"#d63031","Foggy":"#b2bec3"}.get(pred.get("predicted_condition",""),"#74b9ff")
            pi    = {"Clear":"☀️","Cloudy":"☁️","Rainy":"🌧️","Stormy":"⛈️","Foggy":"🌫️"}.get(pred.get("predicted_condition",""),"🌤️")
            st.markdown(f"""
            <div class='ml-box'>
                <p class='ai-label' style='color:#55efc4;'>🧠 &nbsp; ML MODEL OUTPUT — RANDOM FOREST</p>
                <div style='display:flex; align-items:center; gap:18px; margin-bottom:16px;'>
                    <p style='font-size:2.8em; margin:0;'>{pi}</p>
                    <div>
                        <p style='font-size:1.8em; font-weight:800; color:{pc}; margin:0 0 4px 0;'>{pred.get('predicted_condition','')}</p>
                        <p style='color:#636e72; font-size:0.84em; margin:0;'>
                            Confidence: <strong style='color:#55efc4;'>{pred.get('confidence',0)*100:.1f}%</strong>
                            &nbsp;·&nbsp; Model Accuracy: <strong style='color:#55efc4;'>{pred.get('model_accuracy',0)*100:.1f}%</strong>
                        </p>
                    </div>
                </div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                    <div>
                        <p style='color:#636e72; font-size:0.72em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 4px 0;'>Severity</p>
                        <p style='color:#dfe6e9; margin:0; font-size:0.92em;'>{pred.get('severity','')}</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.72em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 4px 0;'>Recommendation</p>
                        <p style='color:#dfe6e9; margin:0; font-size:0.92em;'>{pred.get('recommendation','')}</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            if "probabilities" in pred:
                st.markdown("<p class='section-header'>📊 Class Probabilities</p>", unsafe_allow_html=True)
                pdf = pd.DataFrame(list(pred["probabilities"].items()), columns=["Condition","Probability"])
                pdf["Probability"] = (pdf["Probability"] * 100).round(1)
                pdf = pdf.sort_values("Probability", ascending=True)
                fig_p = px.bar(pdf, x="Probability", y="Condition", orientation="h",
                               color="Condition",
                               color_discrete_map={"Clear":"#fdcb6e","Cloudy":"#74b9ff",
                                                   "Rainy":"#0984e3","Stormy":"#d63031","Foggy":"#b2bec3"},
                               text=pdf["Probability"].apply(lambda v: f"{v}%"))
                fig_p.update_traces(textposition="outside", marker_line_width=0)
                fig_p.update_layout(**CHART_BASE, height=230, showlegend=False,
                                    margin=dict(l=0, r=70, t=10, b=0),
                                    xaxis=dict(showgrid=False, visible=False, title=""),
                                    yaxis=dict(showgrid=False, color="#b2bec3", title=""))
                st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <p class='ai-label' style='color:#55efc4;'>🧠 &nbsp; MODEL DETAILS</p>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px;'>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Algorithm</p>
                         <p style='color:#dfe6e9; margin:0; font-weight:600;'>Random Forest Classifier</p></div>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Accuracy</p>
                         <p style='color:#55efc4; margin:0; font-weight:700; font-size:1.1em;'>97.3%</p></div>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Training Samples</p>
                         <p style='color:#dfe6e9; margin:0; font-weight:600;'>4,000 (80%)</p></div>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Test Samples</p>
                         <p style='color:#dfe6e9; margin:0; font-weight:600;'>1,000 (20%)</p></div>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Input Features</p>
                         <p style='color:#dfe6e9; margin:0; font-weight:600;'>temp · humidity · wind · aqi · rain · precipitation</p></div>
                    <div><p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px 0;'>Output Classes</p>
                         <p style='color:#dfe6e9; margin:0; font-weight:600;'>Clear · Cloudy · Rainy · Stormy · Foggy</p></div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.info("💡 **ML predictions require the FastAPI backend.**  \nStart it with: `uvicorn backend:app --reload --port 8000`")

        # Model stats
        st.markdown("<p class='section-header'>📋 Model Performance Metrics</p>", unsafe_allow_html=True)
        try:
            with open("model_stats.json") as f:
                ms = json.load(f)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy",     f"{ms.get('accuracy',0)*100:.1f}%")
            mc2.metric("Training Set", str(ms.get("n_train","4,000")))
            mc3.metric("Test Set",     str(ms.get("n_test","1,000")))
            mc4.metric("N Estimators", "150")

            if "feature_importance" in ms:
                st.markdown("<p class='section-header'>🔬 Feature Importances</p>", unsafe_allow_html=True)
                fi_df = pd.DataFrame(list(ms["feature_importance"].items()), columns=["Feature","Importance"])
                fi_df["Importance"] = (fi_df["Importance"] * 100).round(2)
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Blues",
                                text=fi_df["Importance"].apply(lambda v: f"{v:.1f}%"))
                fig_fi.update_traces(textposition="outside", marker_line_width=0)
                fig_fi.update_layout(**CHART_BASE, height=280, showlegend=False,
                                     coloraxis_showscale=False,
                                     margin=dict(l=0, r=80, t=10, b=0),
                                     xaxis=dict(showgrid=False, visible=False, title=""),
                                     yaxis=dict(showgrid=False, color="#b2bec3", title=""))
                st.plotly_chart(fig_fi, use_container_width=True)

            if "report" in ms:
                st.markdown("<p class='section-header'>📊 Per-Class Performance</p>", unsafe_allow_html=True)
                rows = [{"Class": cls,
                         "Precision": f"{v.get('precision',0)*100:.1f}%",
                         "Recall":    f"{v.get('recall',0)*100:.1f}%",
                         "F1-Score":  f"{v.get('f1-score',0)*100:.1f}%",
                         "Support":   int(v.get("support",0))}
                        for cls, v in ms["report"].items() if isinstance(v, dict)]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    # TAB 5 — PATTERN ANALYSIS
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown("<p class='section-header'>📊 10-Day Pattern Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.88em; margin:0 0 16px 0;'>Statistical analysis and AI-generated insights from your 10-day forecast data.</p>", unsafe_allow_html=True)

        if not st.session_state.get("ai_analysis"):
            st.session_state.ai_analysis = analyze_patterns(daily_df, unit_symbol)
        an = st.session_state.ai_analysis

        kpi_cols = st.columns(4, gap="small")
        kpis = [("🌡️","Avg High",   f"{an['avg_max']:.1f}{unit_symbol}", "#ff7675"),
                ("❄️","Avg Low",    f"{an['avg_min']:.1f}{unit_symbol}", "#74b9ff"),
                ("🌧️","Rainy Days", f"{an['rainy_days']} / 10",          "#0984e3"),
                ("📊","Temp Swing", f"{an['temp_swing']:.1f}{unit_symbol}","#fdcb6e")]
        for col, (em, lbl, val, vc) in zip(kpi_cols, kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center; padding:20px 14px;'>
                    <p style='font-size:2em; margin:0;'>{em}</p>
                    <p style='color:#636e72; font-size:0.72em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:6px 0 4px 0;'>{lbl}</p>
                    <p style='font-size:1.4em; font-weight:800; margin:0; color:{vc};'>{val}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>💡 Key Insights</p>", unsafe_allow_html=True)
        for em, title, desc_text in an["insights"]:
            st.markdown(f"""
            <div class='insight-card'>
                <span style='font-size:1.1em;'>{em}</span>
                <strong style='color:#74b9ff;'>&nbsp;{title}:</strong>
                <span style='color:#b2bec3;'>&nbsp;{desc_text}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2, gap="medium")

        with pc1:
            st.markdown("<p class='section-header'>🌡️ Temperature Range</p>", unsafe_allow_html=True)
            tdf = pd.DataFrame({
                "Day": list(daily_df["day"]) * 2,
                "Temperature": list(daily_df["temperature_2m_max"]) + list(daily_df["temperature_2m_min"]),
                "Type": ["High"] * len(daily_df) + ["Low"] * len(daily_df)
            })
            fig_t = px.line(tdf, x="Day", y="Temperature", color="Type",
                            color_discrete_map={"High":"#ff7675","Low":"#74b9ff"}, markers=True)
            fig_t.update_traces(line=dict(width=2.5))
            fig_t.update_layout(**CHART_BASE, height=280, showlegend=True,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis=dict(showgrid=False, color="#636e72", title="",
                                           tickangle=-30, tickfont=dict(size=10)),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                           color="#636e72", title=f"Temp ({unit_symbol})"),
                                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b2bec3"),
                                            title="", orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_t, use_container_width=True)

        with pc2:
            st.markdown("<p class='section-header'>💧 Rain Probability</p>", unsafe_allow_html=True)
            rdf = daily_df[["day","precipitation_probability_max"]].copy()
            rdf["Level"] = rdf["precipitation_probability_max"].apply(
                lambda r: "Low (<40%)" if r < 40 else "Moderate (40–70%)" if r < 70 else "High (>70%)"
            )
            fig_r = px.bar(rdf, x="day", y="precipitation_probability_max", color="Level",
                           color_discrete_map={"Low (<40%)":"#74b9ff","Moderate (40–70%)":"#0984e3","High (>70%)":"#d63031"},
                           text=rdf["precipitation_probability_max"].apply(lambda r: f"{int(r)}%"))
            fig_r.update_traces(textposition="outside", marker_line_width=0)
            fig_r.update_layout(**CHART_BASE, height=280, showlegend=True,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis=dict(showgrid=False, color="#636e72", title="",
                                           tickangle=-30, tickfont=dict(size=10)),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                           color="#636e72", range=[0,120], title="Rain %"),
                                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b2bec3"),
                                            title="", orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_r, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 6 — SMART TIPS
    # ══════════════════════════════════════════════════════════
    with tab6:
        st.markdown("<p class='section-header'>💡 Personalized Recommendations</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#636e72; font-size:0.88em; margin:0 0 16px 0;'>Intelligent tips tailored to today's conditions in <strong style='color:#74b9ff;'>{location['name']}</strong>.</p>", unsafe_allow_html=True)

        tc1, tc2 = st.columns([6, 1])
        with tc2:
            if st.button("🔄 Refresh", key="ref_tips"):
                if "ai_tips" in st.session_state: del st.session_state["ai_tips"]
                st.rerun()

        if not st.session_state.get("ai_tips"):
            w, act, h = smart_tips(desc, current["temperature_2m"], aqi, rain_today)
            st.session_state.ai_tips = (w, act, h)

        wear, activity, health = st.session_state.ai_tips

        tip1, tip2 = st.columns(2, gap="medium")
        with tip1:
            st.markdown(f"""
            <div class='tip-wear'>
                <p class='tip-title' style='color:#74b9ff;'>👗 &nbsp; What to Wear</p>
                <p class='tip-body'>{wear}</p>
            </div>""", unsafe_allow_html=True)
        with tip2:
            st.markdown(f"""
            <div class='tip-activity'>
                <p class='tip-title' style='color:#55efc4;'>🏃 &nbsp; Activity Suggestions</p>
                <p class='tip-body'>{activity}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='tip-health'>
            <p class='tip-title' style='color:#fdcb6e;'>🏥 &nbsp; Health & Safety</p>
            <p class='tip-body'>{health}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>⚡ Today at a Glance</p>", unsafe_allow_html=True)
        umbrella = "✅ Recommended"  if rain_today > 40 else "❌ Not needed"
        outdoor  = ("✅ Great day"   if rain_today < 30 and aqi < 100 and current["temperature_2m"] < 36
                    else "⚠️ Moderate" if rain_today < 60 and aqi < 150 else "❌ Avoid outdoors")
        mask     = ("✅ Recommended" if aqi > 150 else "⚠️ Optional" if aqi > 100 else "❌ Not needed")
        uv       = ("🔴 Very High"   if current["temperature_2m"] > 35 else
                    "🟠 High"        if current["temperature_2m"] > 28 else
                    "🟡 Moderate"    if current["temperature_2m"] > 20 else "🟢 Low")

        g_cols = st.columns(4, gap="small")
        for col, (lbl, val) in zip(g_cols, [("☂️ Umbrella?",umbrella),
                                             ("🌳 Go Outside?",outdoor),
                                             ("😷 Mask?",mask),
                                             ("☀️ UV Risk",uv)]):
            with col:
                st.markdown(f"""
                <div class='glance-card'>
                    <p style='color:#636e72; font-size:0.72em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 10px 0;'>{lbl}</p>
                    <p style='font-size:0.95em; font-weight:700; margin:0; color:#dfe6e9;'>{val}</p>
                </div>""", unsafe_allow_html=True)
