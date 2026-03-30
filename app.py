"""
PyWeather AI — Streamlit Frontend
Intel Unnati Generative AI Major Project
Connects to FastAPI backend at http://localhost:8000
Falls back to direct API calls if backend is unavailable.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from transformers import pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(page_title="PyWeather AI", layout="wide", page_icon="🌤️")
BACKEND_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(145deg, #1c1c3a, #16213e);
    border: 1px solid rgba(116,185,255,0.1);
    border-radius: 18px; padding: 22px 26px; margin: 8px 0;
}
.temp-display {
    font-size: 6em; font-weight: 800;
    background: linear-gradient(135deg, #a8d8ff, #0984e3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 0; letter-spacing: -3px;
}
.section-header {
    font-size: 0.72em; font-weight: 700; color: #74b9ff;
    text-transform: uppercase; letter-spacing: 2.5px;
    margin: 20px 0 14px 0; padding-bottom: 10px;
    border-bottom: 1px solid rgba(116,185,255,0.15);
}
.ai-box {
    background: linear-gradient(135deg, rgba(116,185,255,0.06), rgba(9,132,227,0.03));
    border: 1px solid rgba(116,185,255,0.2); border-left: 3px solid #74b9ff;
    border-radius: 14px; padding: 20px 24px; margin: 14px 0;
}
.ai-label {
    font-size: 0.68em; font-weight: 800; color: #74b9ff;
    text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 10px;
}
.ml-box {
    background: linear-gradient(135deg, rgba(85,239,196,0.06), rgba(0,184,148,0.03));
    border: 1px solid rgba(85,239,196,0.2); border-left: 3px solid #55efc4;
    border-radius: 14px; padding: 20px 24px; margin: 14px 0;
}
.insight-card {
    background: rgba(116,185,255,0.04); border: 1px solid rgba(116,185,255,0.1);
    border-left: 3px solid #74b9ff; border-radius: 0 12px 12px 0;
    padding: 14px 20px; margin: 8px 0; font-size: 0.92em; color: #dfe6e9; line-height: 1.6;
}
.tip-wear {
    background: linear-gradient(135deg, rgba(41,128,185,0.12), rgba(41,128,185,0.05));
    border: 1px solid rgba(41,128,185,0.25); border-radius: 16px; padding: 22px 26px; margin: 12px 0;
}
.tip-activity {
    background: linear-gradient(135deg, rgba(39,174,96,0.12), rgba(39,174,96,0.05));
    border: 1px solid rgba(39,174,96,0.25); border-radius: 16px; padding: 22px 26px; margin: 12px 0;
}
.tip-health {
    background: linear-gradient(135deg, rgba(243,156,18,0.12), rgba(243,156,18,0.05));
    border: 1px solid rgba(243,156,18,0.25); border-radius: 16px; padding: 22px 26px; margin: 12px 0;
}
.tip-title { font-size: 0.8em; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; margin: 0 0 10px 0; }
.tip-body { color: #dfe6e9; line-height: 1.7; font-size: 0.95em; margin: 0; }
.glance-card {
    background: linear-gradient(145deg, #1c1c3a, #16213e);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    padding: 18px 14px; text-align: center; margin: 6px 0;
}
.forecast-row {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 14px 22px; margin: 5px 0;
}
.forecast-row:hover { background: rgba(116,185,255,0.05); }
.backend-badge-on {
    background: rgba(39,174,96,0.15); border: 1px solid rgba(39,174,96,0.3);
    color: #55efc4; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 700;
}
.backend-badge-off {
    background: rgba(255,118,117,0.15); border: 1px solid rgba(255,118,117,0.3);
    color: #ff7675; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 700;
}
.stButton > button { border-radius: 10px !important; font-weight: 600 !important; font-size: 0.85em !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b0f1a 0%, #111827 100%); border-right: 1px solid rgba(255,255,255,0.05); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(255,255,255,0.02); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; padding: 8px 18px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(116,185,255,0.15) !important; color: #74b9ff !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

WMO_CODES = {
    0: ("Clear Sky", "☀️"), 1: ("Mainly Clear", "🌤️"), 2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 48: ("Rime Fog", "🌫️"),
    51: ("Light Drizzle", "💧"), 53: ("Drizzle", "💧"), 55: ("Dense Drizzle", "💧"),
    61: ("Light Rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy Rain", "🌧️"),
    71: ("Light Snow", "❄️"), 73: ("Snow", "❄️"), 75: ("Heavy Snow", "❄️"),
    80: ("Showers", "🌦️"), 81: ("Showers", "🌦️"), 82: ("Violent Showers", "🌦️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm+Hail", "⛈️"), 99: ("Severe Storm", "⛈️"),
}

def get_aqi_info(aqi):
    if aqi <= 50:   return "Good", "#00b894", "🟢"
    elif aqi <= 100: return "Moderate", "#fdcb6e", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#e17055", "🟠"
    elif aqi <= 200: return "Unhealthy", "#d63031", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "#6c5ce7", "🟣"
    else:            return "Hazardous", "#2d3436", "⚫"

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
            return None, None, None
        loc = geo["results"][0]
        lat, lon, tz = loc["latitude"], loc["longitude"], loc["timezone"]
        unit_temp = "celsius" if units == "metric" else "fahrenheit"
        unit_wind = "kmh" if units == "metric" else "mph"
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": tz,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "temperature_unit": unit_temp, "wind_speed_unit": unit_wind, "forecast_days": 10
        }, timeout=10).json()
        a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                         params={"latitude": lat, "longitude": lon, "current": "us_aqi"}, timeout=10).json()
        return w, a, loc
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

def process_forecast(daily):
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df["day"] = df["date"].dt.strftime("%a, %b %d")
    return df

def process_hourly(hourly):
    df = pd.DataFrame(hourly)
    df["Timestamp"] = pd.to_datetime(df["time"])
    df.rename(columns={"temperature_2m": "Temperature",
                        "relative_humidity_2m": "Humidity",
                        "precipitation_probability": "Rain Chance (%)"}, inplace=True)
    return df.set_index("Timestamp")

def smart_summary(condition, temp, humidity, rain_chance, wind, wind_unit, aqi, city):
    feel = ("hot and humid" if temp > 32 and humidity > 70 else
            "warm and pleasant" if temp > 25 else
            "cool and comfortable" if temp > 15 else
            "cold" if temp > 5 else "freezing")
    rain_desc = ("high chance of rain — carry an umbrella" if rain_chance > 60 else
                 "possible showers later" if rain_chance > 30 else "mostly dry")
    aqi_label, _, _ = get_aqi_info(aqi)
    aqi_note = (f" Air quality is {aqi_label.lower()} — limit outdoor exposure." if aqi > 150 else
                f" Air quality is {aqi_label.lower()}." )
    _, icon = WMO_CODES.get(0, ("Clear", "☀️"))
    return (f"{city} is experiencing **{condition.lower()}** conditions "
            f"with a temperature of **{temp:.0f}°**, feeling {feel}. "
            f"Humidity at {humidity}% with winds of {wind} {wind_unit}. "
            f"Rain chance today: **{rain_chance}%** — {rain_desc}.{aqi_note}")

def smart_tips(condition, temp, aqi, rain_chance):
    cond = condition.lower()
    if "storm" in cond or "thunder" in cond:
        wear = "Stay indoors if possible. Heavy waterproof gear is essential if going out."
        activity = "Avoid all outdoor activities. Perfect for movies, cooking, or indoor work."
    elif "rain" in cond or "drizzle" in cond or "shower" in cond:
        wear = "Waterproof jacket and umbrella are a must. Water-resistant footwear recommended."
        activity = "Outdoor plans may be disrupted. Consider cafés, museums, or indoor workouts."
    elif "snow" in cond:
        wear = "Heavy coat, thermal layers, waterproof boots, gloves, and a warm hat."
        activity = "Snow activities if you enjoy them, otherwise stay warm indoors."
    elif temp > 35:
        wear = "Light breathable clothes in pale colors. Hat and sunglasses strongly recommended."
        activity = "Avoid midday heat. Morning/evening walks, swimming, or indoor activities."
    elif temp > 25:
        wear = "Light cotton or linen clothes. T-shirt and shorts will be comfortable."
        activity = "Great for picnics, cycling, jogging, or any outdoor sport."
    elif temp > 15:
        wear = "Comfortable casuals with a light jacket for evenings."
        activity = "Good for walks, sports, or outdoor dining."
    elif temp > 5:
        wear = "Warm coat, scarf, and consider thermal underlayers."
        activity = "Brisk walks are fine. Limit extended outdoor exposure."
    else:
        wear = "Full winter gear — heavy coat, thermals, gloves, and a warm hat."
        activity = "Minimize time outdoors. Stay warm and safe."

    health_parts = []
    if aqi > 200:    health_parts.append("Air quality is very unhealthy — wear an N95 mask outdoors.")
    elif aqi > 150:  health_parts.append("Unhealthy air quality — sensitive groups should stay inside.")
    elif aqi > 100:  health_parts.append("Moderate air quality — children and elderly should limit exposure.")
    else:            health_parts.append("Air quality is acceptable for most people today.")
    if temp > 35:    health_parts.append("Drink 3–4 litres of water to prevent heat exhaustion.")
    if rain_chance > 50: health_parts.append("Wet conditions — wear non-slip footwear.")
    return wear, activity, " ".join(health_parts)

def analyze_patterns(daily_df, unit_symbol):
    temps_max = daily_df["temperature_2m_max"].tolist()
    temps_min = daily_df["temperature_2m_min"].tolist()
    rain = daily_df["precipitation_probability_max"].tolist()
    avg_max, avg_min = sum(temps_max)/len(temps_max), sum(temps_min)/len(temps_min)
    avg_rain, max_temp, min_temp = sum(rain)/len(rain), max(temps_max), min(temps_min)
    rainy_days = sum(1 for r in rain if r > 50)
    temp_swing = max_temp - min_temp
    insights = []
    if avg_rain > 60:   insights.append(("🌧️","High Rainfall Week","Persistent rain expected — keep an umbrella all week."))
    elif avg_rain > 30: insights.append(("🌦️","Occasional Showers","Intermittent rain possible — check before heading out."))
    else:               insights.append(("☀️","Mostly Dry Week","Great week for outdoor plans with minimal rain."))
    if temp_swing > 12: insights.append(("🌡️","Large Temperature Variation",f"Swing of {temp_swing:.0f}{unit_symbol} — dress in layers."))
    else:               insights.append(("🌡️","Stable Temperatures","Consistent temperatures throughout the week."))
    if max_temp > 38:   insights.append(("🔥","Extreme Heat Warning","Stay indoors during peak hours. Hydrate frequently."))
    elif max_temp > 32: insights.append(("☀️","Hot Days Ahead","Stay hydrated and wear sunscreen."))
    if min_temp < 5:    insights.append(("🥶","Cold Nights","Near-freezing lows — keep warm clothing ready."))
    if rainy_days == 0: insights.append(("🌈","Zero Rain Forecast","Perfect outdoor week!"))
    elif rainy_days >= 7: insights.append(("☔","Very Rainy Week",f"Rain on {rainy_days}/10 days — plan indoors."))
    return {"avg_max": avg_max, "avg_min": avg_min, "avg_rain": avg_rain,
            "max_temp": max_temp, "min_temp": min_temp,
            "rainy_days": rainy_days, "temp_swing": temp_swing, "insights": insights}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 10px 0;'>
        <div style='font-size:3em; margin-bottom:6px;'>🌤️</div>
        <h2 style='margin:0 0 4px 0; color:#fff; font-size:1.3em; font-weight:800;'>PyWeather AI</h2>
        <p style='color:#636e72; font-size:0.75em; margin:0; letter-spacing:1px; text-transform:uppercase;'>Intelligent Weather Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    backend_ok, backend_info = check_backend()
    if backend_ok:
        st.markdown('<div class="backend-badge-on">⚡ Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="backend-badge-off">⚠️ Direct Mode (Backend offline)</div>', unsafe_allow_html=True)

    st.markdown("---")

    if "saved_cities" not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"]

    city = st.text_input("🔍 Search City", "Ghaziabad", placeholder="Enter city name...")

    def reset_cache():
        for k in ["ai_summary", "ai_severity", "ai_tips", "ai_analysis",
                  "weather_text", "ml_prediction", "model_info"]:
            st.session_state[k] = None

    if st.button("Search", use_container_width=True, type="primary"):
        st.session_state.city = city
        reset_cache()

    st.markdown("**📌 Saved Cities**")
    for saved_city in st.session_state.saved_cities:
        if st.button(f"🏙️ {saved_city}", use_container_width=True, key=saved_city):
            st.session_state.city = saved_city
            reset_cache()

    if "city" not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.markdown("---")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit = unit_options[st.radio("Temperature Unit", unit_options.keys(), label_visibility="collapsed")]
    unit_symbol = "°C" if selected_unit == "metric" else "°F"

    st.markdown("---")
    st.markdown("""
    <div style='border-top:1px solid rgba(255,255,255,0.05); padding-top:14px;'>
        <p style='color:#4a5568; font-size:0.72em; text-align:center; line-height:2; margin:0;'>
            🤖 Random Forest + HuggingFace<br>
            🌐 Open-Meteo API (Free)<br>
            ⚡ FastAPI Backend
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────

st.markdown(f"""
<div style='margin-bottom:16px;'>
    <h1 style='margin:0; font-size:2.4em; font-weight:800; letter-spacing:-1px;'>
        🌤️ {st.session_state.city}
    </h1>
</div>
""", unsafe_allow_html=True)

# Try backend first, fall back to direct
weather_data = air_data = location = None
ml_prediction = None
using_backend = False

if st.session_state.city:
    if backend_ok:
        result, ok = fetch_weather_via_backend(st.session_state.city, selected_unit)
        if ok and result:
            location = result["location"]
            air_data = {"current": result["air_quality"]}
            ml_prediction = result.get("ml_prediction")
            # Re-wrap to match expected structure
            weather_data = {
                "current": result["current"],
                "daily": result["daily"],
                "hourly": result["hourly"]
            }
            using_backend = True

    if not using_backend:
        weather_data, air_data, location = fetch_weather_direct(st.session_state.city, selected_unit)

    if weather_data and air_data and location:
        st.session_state.weather_data = weather_data
        st.session_state.air_data = air_data
        st.session_state.unit_symbol = unit_symbol
        st.session_state.selected_unit = selected_unit
        st.session_state.location = location
        st.session_state.ml_prediction = ml_prediction
        if st.session_state.city not in st.session_state.saved_cities:
            if len(st.session_state.saved_cities) >= 5:
                st.session_state.saved_cities.pop(0)
            st.session_state.saved_cities.append(st.session_state.city)

if "weather_data" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#636e72;'>
        <div style='font-size:4em;'>🌍</div>
        <h3 style='color:#74b9ff;'>Welcome to PyWeather AI</h3>
        <p>Search for any city to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    weather_data = st.session_state.weather_data
    air_data     = st.session_state.air_data
    unit_symbol  = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location     = st.session_state.location
    ml_prediction = st.session_state.get("ml_prediction")

    current   = weather_data["current"]
    daily_df  = process_forecast(weather_data["daily"])
    hourly_df = process_hourly(weather_data["hourly"])
    aqi       = air_data["current"]["us_aqi"]
    aqi_label, aqi_color, aqi_emoji = get_aqi_info(aqi)
    desc, icon = WMO_CODES.get(current["weather_code"], ("Unknown", "❓"))
    wind_label = "km/h" if selected_unit == "metric" else "mph"
    rain_today = daily_df.iloc[0]["precipitation_probability_max"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Current Weather",
        "📅 10-Day Forecast",
        "🤖 AI Summary",
        "🧠 ML Prediction",
        "📊 Pattern Analysis",
        "💡 Smart Tips"
    ])

    # ─── TAB 1: CURRENT WEATHER ───
    with tab1:
        st.markdown(f"<p style='color:#636e72; margin:0;'>📍 {location['name']}, {location.get('country', '')}</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='temp-display'>{current['temperature_2m']:.0f}{unit_symbol}</p>
                <p style='font-size:1.4em; margin:8px 0 4px 0;'>{icon} {desc}</p>
                <p style='color:#636e72; margin:0; font-size:0.9em;'>
                    H: {daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol} &nbsp;·&nbsp;
                    L: {daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}
                </p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card' style='height:100%;'>
                <p class='section-header'>Conditions</p>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px;'>
                    <div>
                        <p style='color:#636e72; font-size:0.8em; margin:0;'>HUMIDITY</p>
                        <p style='font-size:1.4em; font-weight:600; margin:2px 0;'>{current['relative_humidity_2m']}%</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.8em; margin:0;'>WIND</p>
                        <p style='font-size:1.4em; font-weight:600; margin:2px 0;'>{current['wind_speed_10m']} {wind_label}</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.8em; margin:0;'>PRECIPITATION</p>
                        <p style='font-size:1.4em; font-weight:600; margin:2px 0;'>{current['precipitation']:.1f} mm</p>
                    </div>
                    <div>
                        <p style='color:#636e72; font-size:0.8em; margin:0;'>RAIN CHANCE</p>
                        <p style='font-size:1.4em; font-weight:600; margin:2px 0;'>{rain_today}%</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card'>
            <p class='section-header'>Air Quality Index</p>
            <div style='display:flex; align-items:center; gap:20px;'>
                <p style='font-size:3em; margin:0;'>{aqi_emoji}</p>
                <div>
                    <p style='font-size:1.6em; font-weight:700; color:{aqi_color}; margin:0;'>{aqi_label}</p>
                    <p style='color:#636e72; margin:0; font-size:0.9em;'>US AQI: {aqi} &nbsp;·&nbsp; 0–50 Good · 51–100 Moderate · 101–150 Sensitive · 151+ Unhealthy</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>3-Day Outlook</p>", unsafe_allow_html=True)
        d_cols = st.columns(3)
        for i, (_, row) in enumerate(daily_df.head(3).iterrows()):
            d, ic = WMO_CODES.get(row["weather_code"], ("Unknown", "❓"))
            with d_cols[i]:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <p style='color:#636e72; font-size:0.8em; margin:0;'>{row['day']}</p>
                    <p style='font-size:2em; margin:4px 0;'>{ic}</p>
                    <p style='font-size:0.85em; margin:0; color:#b2bec3;'>{d}</p>
                    <p style='margin:4px 0;'><span style='color:#ff7675;'>{row['temperature_2m_max']:.0f}{unit_symbol}</span> / <span style='color:#74b9ff;'>{row['temperature_2m_min']:.0f}{unit_symbol}</span></p>
                    <p style='color:#636e72; font-size:0.8em; margin:0;'>💧 {row['precipitation_probability_max']}%</p>
                </div>""", unsafe_allow_html=True)

    # ─── TAB 2: FORECAST ───
    with tab2:
        st.markdown("<p class='section-header'>10-Day Daily Forecast</p>", unsafe_allow_html=True)
        for _, row in daily_df.iterrows():
            d, ic = WMO_CODES.get(row["weather_code"], ("Unknown", "❓"))
            rain = row["precipitation_probability_max"]
            st.markdown(f"""
            <div class='forecast-row'>
                <div style='display:flex; align-items:center; gap:16px;'>
                    <div style='width:110px; color:#b2bec3; font-size:0.9em;'>{row['day']}</div>
                    <div style='width:30px; font-size:1.3em; text-align:center;'>{ic}</div>
                    <div style='flex:1; color:#dfe6e9; font-size:0.88em;'>{d}</div>
                    <div style='width:80px; text-align:right;'>
                        <span style='color:#ff7675; font-weight:600;'>{row['temperature_2m_max']:.0f}{unit_symbol}</span>
                        <span style='color:#636e72;'> / </span>
                        <span style='color:#74b9ff; font-weight:600;'>{row['temperature_2m_min']:.0f}{unit_symbol}</span>
                    </div>
                    <div style='width:100px;'>
                        <div style='background:rgba(255,255,255,0.05); border-radius:4px; height:6px; overflow:hidden;'>
                            <div style='background:#74b9ff; width:{int(rain)}%; height:100%; border-radius:4px;'></div>
                        </div>
                        <p style='color:#636e72; font-size:0.75em; margin:2px 0 0 0; text-align:right;'>💧 {rain}%</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><p class='section-header'>Hourly Trends</p>", unsafe_allow_html=True)
        plot_choice = st.selectbox("", ["Temperature", "Rain Chance (%)", "Humidity"], label_visibility="collapsed")
        color_map = {"Temperature": "#ff7675", "Rain Chance (%)": "#74b9ff", "Humidity": "#55efc4"}
        hourly_reset = hourly_df.reset_index()
        fig = px.area(hourly_reset, x="Timestamp", y=plot_choice, color_discrete_sequence=[color_map[plot_choice]])
        fig.update_traces(opacity=0.75)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#b2bec3'),
                          xaxis=dict(showgrid=False, color='#636e72', title=""),
                          yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72'),
                          margin=dict(l=0, r=0, t=20, b=0), height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ─── TAB 3: AI SUMMARY ───
    with tab3:
        st.markdown("<p class='section-header'>AI Weather Summary</p>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Summary", key="ref_sum"):
            st.session_state.ai_summary = None
            st.session_state.ai_severity = None
            st.rerun()

        if not st.session_state.get("ai_summary"):
            st.session_state.ai_summary = smart_summary(
                desc, current["temperature_2m"], current["relative_humidity_2m"],
                rain_today, current["wind_speed_10m"], wind_label, aqi, location["name"]
            )

        st.markdown(f"""
        <div class='ai-box'>
            <p class='ai-label'>✦ AI Analysis</p>
            <p style='margin:0; line-height:1.7; color:#dfe6e9;'>{st.session_state.ai_summary}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>Weather Severity Classification</p>", unsafe_allow_html=True)
        if not st.session_state.get("ai_severity"):
            with st.spinner("🧠 Classifying weather severity..."):
                try:
                    clf = load_classifier()
                    weather_text = (f"{desc} weather with {current['temperature_2m']:.0f}°C temperature, "
                                    f"{current['relative_humidity_2m']}% humidity, AQI {aqi}, "
                                    f"{rain_today}% rain chance.")
                    result = clf(weather_text, ["safe and pleasant", "mild caution advised",
                                                "severe weather warning", "extreme danger"])
                    st.session_state.ai_severity = {
                        "label": result["labels"][0], "score": result["scores"][0],
                        "all": list(zip(result["labels"], result["scores"]))
                    }
                except Exception as e:
                    st.session_state.ai_severity = {"label": "unknown", "score": 0, "all": [], "error": str(e)}

        sev = st.session_state.ai_severity
        label, score = sev.get("label","unknown"), sev.get("score", 0)
        sev_color = {"safe and pleasant": "#00b894", "mild caution advised": "#fdcb6e",
                     "severe weather warning": "#e17055", "extreme danger": "#d63031"}.get(label, "#74b9ff")
        sev_emoji = {"safe and pleasant": "✅", "mild caution advised": "⚠️",
                     "severe weather warning": "🚨", "extreme danger": "🆘"}.get(label, "🔍")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{sev_color}15,{sev_color}05); border:1px solid {sev_color}40; border-radius:16px; padding:22px 26px; margin:14px 0; display:flex; align-items:center; gap:18px;'>
            <p style='font-size:3em; margin:0;'>{sev_emoji}</p>
            <div>
                <p style='font-size:0.7em; font-weight:800; color:{sev_color}; text-transform:uppercase; letter-spacing:2px; margin:0 0 4px 0;'>Severity Level</p>
                <p style='font-size:1.5em; font-weight:700; color:{sev_color}; margin:0;'>{label.title()}</p>
                <p style='color:#636e72; margin:4px 0 0 0; font-size:0.82em;'>Confidence: {score*100:.1f}%</p>
            </div>
        </div>""", unsafe_allow_html=True)

        if sev.get("all"):
            sev_df = pd.DataFrame({"Category": [x[0].title() for x in sev["all"]],
                                   "Confidence": [round(x[1]*100,1) for x in sev["all"]]})
            colors = ["#00b894","#fdcb6e","#e17055","#d63031"]
            sev_df["Color"] = colors[:len(sev_df)]
            fig2 = px.bar(sev_df, x="Confidence", y="Category", orientation="h",
                          color="Category",
                          color_discrete_map=dict(zip(sev_df["Category"], sev_df["Color"])),
                          text=sev_df["Confidence"].apply(lambda v: f"{v}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#b2bec3'),
                               xaxis=dict(showgrid=False, visible=False, title=""),
                               yaxis=dict(showgrid=False, color='#b2bec3', title=""),
                               margin=dict(l=0, r=60, t=10, b=0), height=180, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ─── TAB 4: ML PREDICTION ───
    with tab4:
        st.markdown("<p class='section-header'>🧠 Random Forest ML Prediction</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.9em;'>Trained on 5,000 synthetic weather samples · 97.3% accuracy · 6 meteorological features</p>", unsafe_allow_html=True)

        if ml_prediction:
            pred = ml_prediction
            pred_color = {"Clear": "#fdcb6e", "Cloudy": "#74b9ff", "Rainy": "#0984e3",
                          "Stormy": "#d63031", "Foggy": "#636e72"}.get(pred["predicted_condition"], "#74b9ff")
            pred_icon  = {"Clear": "☀️", "Cloudy": "☁️", "Rainy": "🌧️",
                          "Stormy": "⛈️", "Foggy": "🌫️"}.get(pred["predicted_condition"], "🌤️")

            st.markdown(f"""
            <div class='ml-box'>
                <p class='ai-label' style='color:#55efc4;'>🧠 ML Model Output</p>
                <div style='display:flex; align-items:center; gap:16px; margin-bottom:12px;'>
                    <p style='font-size:2.5em; margin:0;'>{pred_icon}</p>
                    <div>
                        <p style='font-size:1.6em; font-weight:700; color:{pred_color}; margin:0;'>{pred['predicted_condition']}</p>
                        <p style='color:#636e72; font-size:0.85em; margin:0;'>Confidence: {pred['confidence']*100:.1f}% &nbsp;·&nbsp; Model Accuracy: {pred['model_accuracy']*100:.1f}%</p>
                    </div>
                </div>
                <p style='color:#636e72; font-size:0.8em; margin:0 0 4px 0; text-transform:uppercase; font-weight:700; letter-spacing:1.5px;'>Severity</p>
                <p style='color:#dfe6e9; margin:0 0 10px 0;'>{pred['severity']}</p>
                <p style='color:#636e72; font-size:0.8em; margin:0 0 4px 0; text-transform:uppercase; font-weight:700; letter-spacing:1.5px;'>Recommendation</p>
                <p style='color:#dfe6e9; margin:0;'>{pred['recommendation']}</p>
            </div>""", unsafe_allow_html=True)

            # Probability chart
            st.markdown("<p class='section-header'>Class Probabilities</p>", unsafe_allow_html=True)
            prob_df = pd.DataFrame(list(pred["probabilities"].items()), columns=["Condition", "Probability"])
            prob_df["Probability"] = (prob_df["Probability"] * 100).round(1)
            prob_df = prob_df.sort_values("Probability", ascending=True)
            fig_prob = px.bar(prob_df, x="Probability", y="Condition", orientation="h",
                              color="Condition",
                              color_discrete_map={"Clear":"#fdcb6e","Cloudy":"#74b9ff","Rainy":"#0984e3","Stormy":"#d63031","Foggy":"#636e72"},
                              text=prob_df["Probability"].apply(lambda v: f"{v}%"))
            fig_prob.update_traces(textposition="outside")
            fig_prob.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#b2bec3'),
                                   xaxis=dict(showgrid=False, visible=False, title=""),
                                   yaxis=dict(showgrid=False, color='#b2bec3', title=""),
                                   margin=dict(l=0, r=60, t=10, b=0), height=220, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("ML prediction requires the FastAPI backend to be running. Start it with:\n\n```\ncd backend\nuvicorn main:app --reload\n```")

        # Model info from backend
        st.markdown("<p class='section-header'>Model Information</p>", unsafe_allow_html=True)
        if backend_ok:
            try:
                minfo = requests.get(f"{BACKEND_URL}/model-info", timeout=5).json()
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Accuracy",     f"{minfo['accuracy']*100:.1f}%")
                col_b.metric("Estimators",   minfo["n_estimators"])
                col_c.metric("Train Samples",minfo["train_samples"])
                col_d.metric("Test Samples", minfo["test_samples"])

                st.markdown("<p class='section-header'>Feature Importances</p>", unsafe_allow_html=True)
                fi = minfo["feature_importances"]
                fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                fi_df["Importance"] = (fi_df["Importance"] * 100).round(2)
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Blues",
                                text=fi_df["Importance"].apply(lambda v: f"{v}%"))
                fig_fi.update_traces(textposition="outside")
                fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color='#b2bec3'),
                                     xaxis=dict(showgrid=False, visible=False, title=""),
                                     yaxis=dict(showgrid=False, color='#b2bec3', title=""),
                                     coloraxis_showscale=False,
                                     margin=dict(l=0, r=60, t=10, b=0), height=260, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load model info: {e}")
        else:
            st.markdown("""
            <div class='metric-card'>
                <p style='color:#636e72; margin:0; font-size:0.9em;'>
                    Model: <strong style='color:#dfe6e9;'>Random Forest Classifier</strong><br>
                    Accuracy: <strong style='color:#55efc4;'>97.3%</strong><br>
                    Features: temperature, humidity, wind_speed, pressure, precipitation, aqi<br>
                    Classes: Clear, Cloudy, Rainy, Stormy, Foggy<br>
                    Dataset: 5,000 synthetic samples · 4,000 train / 1,000 test
                </p>
            </div>""", unsafe_allow_html=True)

    # ─── TAB 5: PATTERN ANALYSIS ───
    with tab5:
        st.markdown("<p class='section-header'>10-Day Weather Pattern Analysis</p>", unsafe_allow_html=True)
        if not st.session_state.get("ai_analysis"):
            st.session_state.ai_analysis = analyze_patterns(daily_df, unit_symbol)

        analysis = st.session_state.ai_analysis
        kpis = [("🌡️","Average High",f"{analysis['avg_max']:.1f}{unit_symbol}"),
                ("❄️","Average Low", f"{analysis['avg_min']:.1f}{unit_symbol}"),
                ("🌧️","Rainy Days",  f"{analysis['rainy_days']} / 10"),
                ("📊","Temp Variation",f"{analysis['temp_swing']:.1f}{unit_symbol}")]
        kpi_cols = st.columns(4)
        for col, (em, lbl, val) in zip(kpi_cols, kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <p style='font-size:1.8em; margin:0;'>{em}</p>
                    <p style='color:#636e72; font-size:0.75em; margin:4px 0 2px 0; text-transform:uppercase;'>{lbl}</p>
                    <p style='font-size:1.3em; font-weight:700; margin:0; color:#74b9ff;'>{val}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>Key Insights</p>", unsafe_allow_html=True)
        for em, title, desc_text in analysis["insights"]:
            st.markdown(f"""
            <div class='insight-card'>
                <span style='font-size:1.2em;'>{em}</span>
                <strong style='color:#74b9ff;'> {title}:</strong>
                <span> {desc_text}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>Temperature Range Forecast</p>", unsafe_allow_html=True)
        temp_plot_df = pd.DataFrame({
            "Day": list(daily_df["day"]) * 2,
            "Temperature": list(daily_df["temperature_2m_max"]) + list(daily_df["temperature_2m_min"]),
            "Type": ["High"] * len(daily_df) + ["Low"] * len(daily_df)
        })
        fig3 = px.line(temp_plot_df, x="Day", y="Temperature", color="Type",
                       color_discrete_map={"High": "#ff7675", "Low": "#74b9ff"}, markers=True)
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#b2bec3'),
                           xaxis=dict(showgrid=False, color='#636e72', title=""),
                           yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72'),
                           legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#b2bec3'), title=""),
                           margin=dict(l=0, r=0, t=10, b=0), height=260)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<p class='section-header'>Daily Rain Probability</p>", unsafe_allow_html=True)
        rain_df = daily_df[["day","precipitation_probability_max"]].copy()
        rain_df["Level"] = rain_df["precipitation_probability_max"].apply(
            lambda r: "Low" if r < 40 else "Moderate" if r < 70 else "High")
        fig4 = px.bar(rain_df, x="day", y="precipitation_probability_max", color="Level",
                      color_discrete_map={"Low":"#74b9ff","Moderate":"#0984e3","High":"#2d3436"},
                      text=rain_df["precipitation_probability_max"].apply(lambda r: f"{int(r)}%"))
        fig4.update_traces(textposition="outside")
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#b2bec3'),
                           xaxis=dict(showgrid=False, color='#636e72', title=""),
                           yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72', range=[0,120], title="Rain %"),
                           legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#b2bec3'), title=""),
                           margin=dict(l=0, r=0, t=20, b=0), height=240)
        st.plotly_chart(fig4, use_container_width=True)

    # ─── TAB 6: SMART TIPS ───
    with tab6:
        st.markdown("<p class='section-header'>Personalized Weather Tips</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#636e72; font-size:0.9em;'>Tailored to today's conditions in {location['name']}.</p>", unsafe_allow_html=True)

        if st.button("🔄 Refresh Tips", key="ref_tips"):
            st.session_state.ai_tips = None
            st.rerun()

        if not st.session_state.get("ai_tips"):
            wear, activity, health = smart_tips(desc, current["temperature_2m"], aqi, rain_today)
            st.session_state.ai_tips = (wear, activity, health)

        wear, activity, health = st.session_state.ai_tips
        for css, color, em, title, content in [
            ("tip-wear",     "#74b9ff", "👗", "What to Wear",          wear),
            ("tip-activity", "#55efc4", "🏃", "Activity Suggestions",  activity),
            ("tip-health",   "#fdcb6e", "🏥", "Health & Safety",       health),
        ]:
            st.markdown(f"""
            <div class='{css}'>
                <p class='tip-title' style='color:{color};'>{em} &nbsp;{title}</p>
                <p class='tip-body'>{content}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header'>Today at a Glance</p>", unsafe_allow_html=True)
        umbrella = "✅ Yes" if rain_today > 40 else "❌ No"
        outdoor  = ("✅ Great" if rain_today < 30 and aqi < 100 and current["temperature_2m"] < 36
                    else "⚠️ Moderate" if rain_today < 60 and aqi < 150 else "❌ Avoid")
        mask     = ("✅ Recommended" if aqi > 150 else "⚠️ Optional" if aqi > 100 else "❌ Not needed")
        glance_cols = st.columns(3)
        for col, (lbl, val) in zip(glance_cols, [("☂️ Umbrella?", umbrella),
                                                   ("🌳 Outdoor?",  outdoor),
                                                   ("😷 Mask?",     mask)]):
            with col:
                st.markdown(f"""
                <div class='glance-card'>
                    <p style='color:#636e72; font-size:0.75em; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 8px 0;'>{lbl}</p>
                    <p style='font-size:1.05em; font-weight:700; margin:0; color:#dfe6e9;'>{val}</p>
                </div>""", unsafe_allow_html=True)
