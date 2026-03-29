import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="PyWeather AI", layout="wide", page_icon="🌤️")

# ============================================================
# CUSTOM CSS — Professional Design
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
}

/* Temperature display */
.temp-display {
    font-size: 5.5em;
    font-weight: 700;
    background: linear-gradient(135deg, #74b9ff, #0984e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}

/* Section headers */
.section-header {
    font-size: 1.1em;
    font-weight: 600;
    color: #74b9ff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(116,185,255,0.2);
}

/* Info pills */
.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    margin: 3px;
}
.pill-blue { background: rgba(116,185,255,0.15); color: #74b9ff; border: 1px solid rgba(116,185,255,0.3); }
.pill-green { background: rgba(85,239,196,0.15); color: #55efc4; border: 1px solid rgba(85,239,196,0.3); }
.pill-orange { background: rgba(253,203,110,0.15); color: #fdcb6e; border: 1px solid rgba(253,203,110,0.3); }
.pill-red { background: rgba(255,118,117,0.15); color: #ff7675; border: 1px solid rgba(255,118,117,0.3); }

/* AI Summary box */
.ai-box {
    background: linear-gradient(135deg, rgba(116,185,255,0.08), rgba(9,132,227,0.05));
    border: 1px solid rgba(116,185,255,0.25);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 12px 0;
    position: relative;
}
.ai-label {
    font-size: 0.7em;
    font-weight: 700;
    color: #74b9ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-left: 3px solid #74b9ff;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin: 8px 0;
    font-size: 0.95em;
    color: #dfe6e9;
}

/* Tip cards */
.tip-card {
    border-radius: 14px;
    padding: 20px 24px;
    margin: 10px 0;
    color: #2d3436;
}
.tip-wear { background: linear-gradient(135deg, #e8f4fd, #d6eaf8); border-left: 4px solid #2980b9; }
.tip-activity { background: linear-gradient(135deg, #e9f7ef, #d5f5e3); border-left: 4px solid #27ae60; }
.tip-health { background: linear-gradient(135deg, #fef9e7, #fdebd0); border-left: 4px solid #f39c12; }

/* Severity bar */
.severity-safe { color: #00b894; font-weight: 700; font-size: 1.3em; }
.severity-mild { color: #fdcb6e; font-weight: 700; font-size: 1.3em; }
.severity-severe { color: #e17055; font-weight: 700; font-size: 1.3em; }
.severity-extreme { color: #d63031; font-weight: 700; font-size: 1.3em; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# WEATHER API
# ============================================================
BASE_URL = "https://api.open-meteo.com/v1/forecast"

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("text-generation", model="gpt2", max_new_tokens=80)

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource(show_spinner=False)
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def get_weather_data(city, units):
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        if "results" not in geo_data or not geo_data["results"]:
            st.error(f"City not found: {city}")
            return None, None, None
        location = geo_data["results"][0]
        lat, lon, timezone = location["latitude"], location["longitude"], location["timezone"]
        unit_temp = "celsius" if units == "metric" else "fahrenheit"
        unit_wind = "kmh" if units == "metric" else "mph"
        weather_response = requests.get(BASE_URL, params={
            "latitude": lat, "longitude": lon, "timezone": timezone,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "temperature_unit": unit_temp, "wind_speed_unit": unit_wind, "forecast_days": 10
        })
        weather_response.raise_for_status()
        air_response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "us_aqi"}
        )
        air_response.raise_for_status()
        return weather_response.json(), air_response.json(), location
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None


WMO_CODES = {
    0: ("Clear Sky", "☀️"), 1: ("Mainly Clear", "🌤️"), 2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 48: ("Rime Fog", "🌫️"),
    51: ("Light Drizzle", "💧"), 53: ("Drizzle", "💧"), 55: ("Heavy Drizzle", "💧"),
    61: ("Light Rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy Rain", "🌧️"),
    71: ("Light Snow", "❄️"), 73: ("Snow", "❄️"), 75: ("Heavy Snow", "❄️"),
    80: ("Rain Showers", "🌦️"), 81: ("Showers", "🌦️"), 82: ("Violent Showers", "🌦️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm + Hail", "⛈️"), 99: ("Severe Thunderstorm", "⛈️"),
}

def get_aqi_info(aqi):
    if aqi <= 50: return "Good", "#00b894", "🟢"
    elif aqi <= 100: return "Moderate", "#fdcb6e", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#e17055", "🟠"
    elif aqi <= 200: return "Unhealthy", "#d63031", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "#6c5ce7", "🟣"
    else: return "Hazardous", "#2d3436", "⚫"

def process_forecast_data(forecast_data):
    daily_df = pd.DataFrame(forecast_data["daily"])
    daily_df["date"] = pd.to_datetime(daily_df["time"])
    daily_df["day"] = daily_df["date"].dt.strftime("%a, %b %d")
    hourly_df = pd.DataFrame(forecast_data["hourly"])
    hourly_df["Timestamp"] = pd.to_datetime(hourly_df["time"])
    hourly_df.rename(columns={
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "precipitation_probability": "Rain Chance (%)"
    }, inplace=True)
    return daily_df, hourly_df.set_index("Timestamp")

def build_weather_text(weather_data, air_data, location, unit_symbol, selected_unit):
    current = weather_data["current"]
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    aqi_label, _, _ = get_aqi_info(aqi)
    weather_desc, _ = WMO_CODES.get(current["weather_code"], ("Unknown", ""))
    wind_unit = "km/h" if selected_unit == "metric" else "mph"
    forecast_text = " ".join([
        f"On {row['day']}, expect {WMO_CODES.get(row['weather_code'], ('Unknown',''))[0]} "
        f"with a high of {row['temperature_2m_max']:.0f}{unit_symbol} "
        f"and a low of {row['temperature_2m_min']:.0f}{unit_symbol}, "
        f"with a {row['precipitation_probability_max']}% chance of rain."
        for _, row in daily_df.head(4).iterrows()
    ])
    return (
        f"The weather in {location['name']} is currently {weather_desc} "
        f"with a temperature of {current['temperature_2m']:.1f}{unit_symbol}. "
        f"Humidity is {current['relative_humidity_2m']}% and wind speed is {current['wind_speed_10m']} {wind_unit}. "
        f"Air quality is {aqi_label} with an AQI of {aqi}. "
        f"Rain chance today is {daily_df.iloc[0]['precipitation_probability_max']}%. "
        f"Forecast: {forecast_text}"
    )


# ============================================================
# AI FUNCTIONS — Rule-based + HuggingFace
# ============================================================

def smart_weather_summary(weather_data, air_data, location, unit_symbol, selected_unit):
    """Generates a smart summary using rules + sentiment — no summarization model needed."""
    current = weather_data["current"]
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    aqi_label, _, _ = get_aqi_info(aqi)
    weather_desc, icon = WMO_CODES.get(current["weather_code"], ("Unknown", ""))
    temp = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    rain_chance = daily_df.iloc[0]["precipitation_probability_max"]
    wind = current["wind_speed_10m"]
    wind_unit = "km/h" if selected_unit == "metric" else "mph"

    # Smart rule-based summary
    feel = "hot and humid" if temp > 32 and humidity > 70 else \
           "warm and pleasant" if temp > 25 else \
           "cool and comfortable" if temp > 15 else \
           "cold" if temp > 5 else "freezing"

    rain_desc = "high chance of rain so carry an umbrella" if rain_chance > 60 else \
                "possible showers later in the day" if rain_chance > 30 else \
                "mostly dry conditions"

    aqi_advice = f"Air quality is {aqi_label.lower()}" + (
        " — limit outdoor exposure." if aqi > 150 else
        " — sensitive groups should be cautious." if aqi > 100 else "."
    )

    return (
        f"{location['name']} is experiencing **{weather_desc.lower()}** conditions {icon} "
        f"with a temperature of **{temp:.0f}{unit_symbol}**, feeling {feel}. "
        f"Humidity stands at {humidity}% with winds at {wind} {wind_unit}. "
        f"There is a **{rain_chance}%** chance of rain today — {rain_desc}. "
        f"{aqi_advice}"
    )


def classify_weather_severity(weather_text):
    """Zero-shot classification of weather severity."""
    try:
        classifier = load_classifier()
        labels = ["safe and pleasant", "mild caution advised", "severe weather warning", "extreme danger"]
        result = classifier(weather_text[:500], candidate_labels=labels)
        return {
            "label": result["labels"][0],
            "score": result["scores"][0],
            "all": list(zip(result["labels"], result["scores"]))
        }
    except Exception as e:
        return {"label": "unknown", "score": 0, "all": [], "error": str(e)}


def analyze_patterns(daily_df, unit_symbol):
    """10-day weather pattern analysis with insights."""
    temps_max = daily_df["temperature_2m_max"].tolist()
    temps_min = daily_df["temperature_2m_min"].tolist()
    rain = daily_df["precipitation_probability_max"].tolist()

    avg_max = sum(temps_max) / len(temps_max)
    avg_min = sum(temps_min) / len(temps_min)
    avg_rain = sum(rain) / len(rain)
    max_temp = max(temps_max)
    min_temp = min(temps_min)
    rainy_days = sum(1 for r in rain if r > 50)
    dry_days = 10 - rainy_days
    temp_swing = max_temp - min_temp

    insights = []
    if avg_rain > 60:
        insights.append(("🌧️", "High Rainfall Week", "Persistent rain expected — keep an umbrella handy all week."))
    elif avg_rain > 30:
        insights.append(("🌦️", "Occasional Showers", "Intermittent rain possible — check forecasts before heading out."))
    else:
        insights.append(("☀️", "Mostly Dry Week", "Great week for outdoor plans with minimal rain expected."))

    if temp_swing > 12:
        insights.append(("🌡️", "Large Temperature Variation", f"Temperatures swing by {temp_swing:.0f}{unit_symbol} — dress in layers."))
    elif temp_swing > 6:
        insights.append(("🌡️", "Moderate Temperature Change", f"Expect a {temp_swing:.0f}{unit_symbol} variation — light layers recommended."))
    else:
        insights.append(("🌡️", "Stable Temperatures", "Consistent temperatures throughout the week."))

    if max_temp > 38:
        insights.append(("🔥", "Extreme Heat Warning", "Dangerous heat expected — stay indoors during peak hours and hydrate frequently."))
    elif max_temp > 32:
        insights.append(("☀️", "Hot Days Ahead", "High temperatures forecast — stay hydrated and wear sunscreen."))
    elif min_temp < 2:
        insights.append(("🧊", "Freezing Temperatures", "Near-freezing lows expected — protect pipes and wear heavy winter clothing."))
    elif min_temp < 10:
        insights.append(("🥶", "Cool Nights", "Cold nights ahead — keep warm clothing ready for evenings."))

    if dry_days >= 8:
        insights.append(("🌈", "Excellent Outdoor Week", f"{dry_days} dry days forecast — perfect for outdoor activities and travel."))
    elif rainy_days >= 7:
        insights.append(("☔", "Rainy Week", f"Rain expected on {rainy_days} out of 10 days — plan indoor alternatives."))

    return {
        "avg_max": avg_max, "avg_min": avg_min, "avg_rain": avg_rain,
        "max_temp": max_temp, "min_temp": min_temp,
        "rainy_days": rainy_days, "dry_days": dry_days,
        "temp_swing": temp_swing, "insights": insights
    }


def generate_smart_tips(weather_data, air_data, unit_symbol, selected_unit):
    """Smart rule-based weather tips — reliable, no model errors."""
    current = weather_data["current"]
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    temp = current["temperature_2m"]
    weather_code = current["weather_code"]
    rain_chance = daily_df.iloc[0]["precipitation_probability_max"]
    condition, _ = WMO_CODES.get(weather_code, ("Unknown", ""))
    condition_lower = condition.lower()

    # Clothing tip
    if "snow" in condition_lower:
        wear = "Heavy winter coat, thermal layers, waterproof boots, gloves, and a warm hat are essential today."
    elif "rain" in condition_lower or "drizzle" in condition_lower or "shower" in condition_lower:
        wear = "Waterproof jacket or raincoat is a must. Carry an umbrella and wear water-resistant footwear."
    elif "storm" in condition_lower:
        wear = "Stay indoors if possible. If going out, wear a heavy waterproof jacket and avoid open areas."
    elif temp > 35:
        wear = "Light, breathable clothing in light colors. A hat and sunglasses are strongly recommended."
    elif temp > 28:
        wear = "Light cotton or linen clothes. A t-shirt and shorts or light trousers will be comfortable."
    elif temp > 20:
        wear = "Comfortable casuals — a light jacket or cardigan in the evening would be a good idea."
    elif temp > 12:
        wear = "Layer up with a sweater or hoodie. A light jacket for outdoor time is advisable."
    elif temp > 5:
        wear = "Wear a warm coat, scarf, and consider thermal underlayers for extended outdoor time."
    else:
        wear = "Full winter gear — heavy coat, thermal layers, gloves, and a warm hat are essential."

    # Activity tip
    if "storm" in condition_lower or "thunder" in condition_lower:
        activity = "Avoid outdoor activities entirely. This is a great day for indoor hobbies, reading, or catching up on work. If you must go out, stay away from trees and open fields."
    elif "rain" in condition_lower or "drizzle" in condition_lower:
        activity = "Outdoor plans may get disrupted. Consider visiting a museum, café, or doing indoor workouts. If rain is light, a short walk with proper gear is still fine."
    elif "snow" in condition_lower:
        activity = "Snow sports like skiing or building a snowman are fun! Otherwise, indoor activities are recommended — roads may be slippery."
    elif temp > 35 or aqi > 150:
        activity = "Avoid strenuous outdoor activity during peak afternoon hours. Morning or evening walks are better. Swimming is a great option on hot days."
    elif temp > 22 and rain_chance < 30 and aqi < 100:
        activity = "Perfect conditions for outdoor activities! Jogging, cycling, picnics, or sports — today is ideal for anything you enjoy outside."
    elif temp > 15:
        activity = "Good weather for moderate outdoor activities. A park walk, light jog, or outdoor dining would be enjoyable."
    else:
        activity = "Cold but manageable. Brisk walks, winter sports, or outdoor photography can be fun. Warm up with hot drinks afterward."

    # Health tip
    health_parts = []
    if aqi > 200:
        health_parts.append("Air quality is very unhealthy — wear an N95 mask outdoors and keep windows closed.")
    elif aqi > 150:
        health_parts.append("Unhealthy air quality today — sensitive individuals (asthma, allergies) should avoid outdoor exposure.")
    elif aqi > 100:
        health_parts.append("Moderate air quality — children and elderly should limit extended outdoor activity.")
    else:
        health_parts.append("Air quality is acceptable for most people today.")

    if temp > 35:
        health_parts.append("Drink at least 3–4 litres of water to prevent heat exhaustion.")
    elif temp < 5:
        health_parts.append("Risk of hypothermia if outdoors for long periods — limit exposure and stay warm.")

    if rain_chance > 50:
        health_parts.append("Wet conditions increase risk of slipping — wear non-slip footwear.")

    health = " ".join(health_parts)

    return wear, activity, health


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 5px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/1779/1779940.png' width='70'>
        <h2 style='margin:8px 0 2px 0; color:#74b9ff; font-size:1.4em;'>PyWeather AI</h2>
        <p style='color:#636e72; font-size:0.8em; margin:0;'>Intelligent Weather Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if "saved_cities" not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"]

    city = st.text_input("🔍 Search City", "Ghaziabad", placeholder="Enter city name...")

    def reset_cache():
        for key in ["ai_summary", "ai_severity", "ai_tips", "ai_analysis", "weather_text"]:
            st.session_state[key] = None

    if st.button("Search", use_container_width=True, type="primary"):
        st.session_state.city = city
        reset_cache()

    st.markdown("**📌 Saved Cities**")
    cols = st.columns(1)
    for saved_city in st.session_state.saved_cities:
        if st.button(f"🏙️ {saved_city}", use_container_width=True, key=saved_city):
            st.session_state.city = saved_city
            reset_cache()

    if "city" not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit = unit_options[st.radio("Temperature Unit", unit_options.keys(), label_visibility="collapsed")]
    unit_symbol = "°C" if selected_unit == "metric" else "°F"

    st.markdown("---")
    st.markdown("""
    <div style='color:#636e72; font-size:0.75em; text-align:center; padding:8px 0;'>
        🤖 Powered by HuggingFace<br>
        🌐 Weather by Open-Meteo<br>
        ✅ No API Key Required
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FETCH WEATHER
# ============================================================
st.markdown(f"<h1 style='margin-bottom:4px;'>🌤️ {st.session_state.city}</h1>", unsafe_allow_html=True)

if st.session_state.city:
    weather_data, air_data, location = get_weather_data(st.session_state.city, selected_unit)
    if weather_data and air_data and location:
        st.session_state.weather_data = weather_data
        st.session_state.air_data = air_data
        st.session_state.unit_symbol = unit_symbol
        st.session_state.selected_unit = selected_unit
        st.session_state.location = location
        if st.session_state.city not in st.session_state.saved_cities:
            if len(st.session_state.saved_cities) >= 5:
                st.session_state.saved_cities.pop(0)
            st.session_state.saved_cities.append(st.session_state.city)

if "weather_data" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#636e72;'>
        <div style='font-size:4em;'>🌍</div>
        <h3 style='color:#74b9ff;'>Welcome to PyWeather AI</h3>
        <p>Search for any city to get started with intelligent weather insights.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    weather_data = st.session_state.weather_data
    air_data = st.session_state.air_data
    unit_symbol = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location = st.session_state.location

    current = weather_data["current"]
    daily_df, hourly_df = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    aqi_label, aqi_color, aqi_emoji = get_aqi_info(aqi)
    desc, icon = WMO_CODES.get(current["weather_code"], ("Unknown", "❓"))
    wind_label = "km/h" if selected_unit == "metric" else "mph"

    # Build weather text once
    if not st.session_state.get("weather_text"):
        st.session_state.weather_text = build_weather_text(
            weather_data, air_data, location, unit_symbol, selected_unit
        )

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Current Weather",
        "📅 10-Day Forecast",
        "🤖 AI Summary",
        "📊 Pattern Analysis",
        "💡 Smart Tips"
    ])

    # ============================================================
    # TAB 1 — CURRENT WEATHER
    # ============================================================
    with tab1:
        st.markdown(f"<p style='color:#636e72; margin:0;'>📍 {location['name']}, {location.get('country', '')} &nbsp;·&nbsp; Last updated just now</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_main, col_details = st.columns([1, 1])

        with col_main:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='temp-display'>{current['temperature_2m']:.0f}{unit_symbol}</p>
                <p style='font-size:1.4em; margin:8px 0 4px 0;'>{icon} {desc}</p>
                <p style='color:#636e72; margin:0; font-size:0.9em;'>
                    Feels like {current['temperature_2m'] - 2:.0f}{unit_symbol} &nbsp;·&nbsp;
                    H: {daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol} &nbsp;·&nbsp;
                    L: {daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_details:
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
                        <p style='font-size:1.4em; font-weight:600; margin:2px 0;'>{daily_df.iloc[0]['precipitation_probability_max']}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AQI Card
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
        </div>
        """, unsafe_allow_html=True)

        # Quick 3-day outlook
        st.markdown("<br>", unsafe_allow_html=True)
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
                </div>
                """, unsafe_allow_html=True)

    # ============================================================
    # TAB 2 — 10-DAY FORECAST
    # ============================================================
    with tab2:
        st.markdown("<p class='section-header'>10-Day Daily Forecast</p>", unsafe_allow_html=True)

        for _, row in daily_df.iterrows():
            d, ic = WMO_CODES.get(row["weather_code"], ("Unknown", "❓"))
            rain = row["precipitation_probability_max"]
            bar_width = int(rain)
            st.markdown(f"""
            <div class='metric-card' style='padding:14px 20px; margin:5px 0;'>
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
                            <div style='background:#74b9ff; width:{bar_width}%; height:100%; border-radius:4px;'></div>
                        </div>
                        <p style='color:#636e72; font-size:0.75em; margin:2px 0 0 0; text-align:right;'>💧 {rain}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Hourly Trends</p>", unsafe_allow_html=True)
        plot_choice = st.selectbox("", ["Temperature", "Rain Chance (%)", "Humidity"], label_visibility="collapsed")

        color_map = {"Temperature": "#ff7675", "Rain Chance (%)": "#74b9ff", "Humidity": "#55efc4"}
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_df.index, y=hourly_df[plot_choice],
            fill='tozeroy',
            line=dict(color=color_map[plot_choice], width=2),
            fillcolor=color_map[plot_choice].replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color_map[plot_choice] else color_map[plot_choice] + "22",
            name=plot_choice
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b2bec3'),
            xaxis=dict(showgrid=False, color='#636e72'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72'),
            margin=dict(l=0, r=0, t=20, b=0),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 3 — AI SUMMARY
    # ============================================================
    with tab3:
        st.markdown("<p class='section-header'>AI Weather Summary</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.9em;'>Intelligent analysis of current conditions and what they mean for your day.</p>", unsafe_allow_html=True)

        if st.button("🔄 Refresh Summary", key="refresh_summary"):
            st.session_state.ai_summary = None
            st.session_state.ai_severity = None

        # Smart Summary (rule-based, always works)
        if not st.session_state.get("ai_summary"):
            st.session_state.ai_summary = smart_weather_summary(
                weather_data, air_data, location, unit_symbol, selected_unit
            )

        st.markdown(f"""
        <div class='ai-box'>
            <p class='ai-label'>✦ AI Analysis</p>
            <p style='margin:0; line-height:1.7; color:#dfe6e9;'>{st.session_state.ai_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Weather Severity Classification</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.9em;'>AI model classifies current weather into safety categories.</p>", unsafe_allow_html=True)

        if not st.session_state.get("ai_severity"):
            with st.spinner("🧠 Classifying weather severity..."):
                st.session_state.ai_severity = classify_weather_severity(st.session_state.weather_text)

        severity = st.session_state.ai_severity
        label = severity.get("label", "unknown")
        score = severity.get("score", 0)

        sev_color = {
            "safe and pleasant": "#00b894",
            "mild caution advised": "#fdcb6e",
            "severe weather warning": "#e17055",
            "extreme danger": "#d63031"
        }.get(label, "#74b9ff")

        sev_emoji = {
            "safe and pleasant": "✅",
            "mild caution advised": "⚠️",
            "severe weather warning": "🚨",
            "extreme danger": "🆘"
        }.get(label, "🔍")

        st.markdown(f"""
        <div class='metric-card'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <p style='font-size:2.5em; margin:0;'>{sev_emoji}</p>
                <div>
                    <p style='font-size:1.3em; font-weight:700; color:{sev_color}; margin:0;'>{label.title()}</p>
                    <p style='color:#636e72; margin:0; font-size:0.85em;'>Confidence: {score*100:.1f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if severity.get("all"):
            labels_list = [x[0].title() for x in severity["all"]]
            scores_list = [round(x[1] * 100, 1) for x in severity["all"]]
            colors = ["#00b894", "#fdcb6e", "#e17055", "#d63031"]
            fig2 = go.Figure(go.Bar(
                x=scores_list, y=labels_list, orientation='h',
                marker_color=colors[:len(labels_list)],
                text=[f"{s}%" for s in scores_list],
                textposition='outside'
            ))
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#b2bec3'),
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, color='#b2bec3'),
                margin=dict(l=0, r=60, t=10, b=0),
                height=180, showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # TAB 4 — PATTERN ANALYSIS
    # ============================================================
    with tab4:
        st.markdown("<p class='section-header'>10-Day Weather Pattern Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.9em;'>Statistical analysis and AI-generated insights from your 10-day forecast.</p>", unsafe_allow_html=True)

        if not st.session_state.get("ai_analysis"):
            st.session_state.ai_analysis = analyze_patterns(daily_df, unit_symbol)

        analysis = st.session_state.ai_analysis

        # KPI Row
        kpi_cols = st.columns(4)
        kpis = [
            ("🌡️", "Average High", f"{analysis['avg_max']:.1f}{unit_symbol}"),
            ("❄️", "Average Low", f"{analysis['avg_min']:.1f}{unit_symbol}"),
            ("🌧️", "Rainy Days", f"{analysis['rainy_days']} / 10"),
            ("📊", "Temp Variation", f"{analysis['temp_swing']:.1f}{unit_symbol}"),
        ]
        for col, (em, label, val) in zip(kpi_cols, kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <p style='font-size:1.8em; margin:0;'>{em}</p>
                    <p style='color:#636e72; font-size:0.75em; margin:4px 0 2px 0; text-transform:uppercase;'>{label}</p>
                    <p style='font-size:1.3em; font-weight:700; margin:0; color:#74b9ff;'>{val}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Key Insights</p>", unsafe_allow_html=True)

        for emoji, title, desc_text in analysis["insights"]:
            st.markdown(f"""
            <div class='insight-card'>
                <span style='font-size:1.2em;'>{emoji}</span>
                <strong style='color:#74b9ff;'> {title}:</strong>
                <span> {desc_text}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Temperature Range Chart
        st.markdown("<p class='section-header'>Temperature Range Forecast</p>", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=daily_df["day"], y=daily_df["temperature_2m_max"],
            name="High", line=dict(color="#ff7675", width=2),
            fill=None
        ))
        fig3.add_trace(go.Scatter(
            x=daily_df["day"], y=daily_df["temperature_2m_min"],
            name="Low", line=dict(color="#74b9ff", width=2),
            fill='tonexty', fillcolor='rgba(116,185,255,0.08)'
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b2bec3'),
            xaxis=dict(showgrid=False, color='#636e72'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#b2bec3')),
            margin=dict(l=0, r=0, t=10, b=0), height=260
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Rain Probability Chart
        st.markdown("<p class='section-header'>Daily Rain Probability</p>", unsafe_allow_html=True)
        bar_colors = ["#74b9ff" if r < 40 else "#0984e3" if r < 70 else "#2d3436" for r in daily_df["precipitation_probability_max"]]
        fig4 = go.Figure(go.Bar(
            x=daily_df["day"],
            y=daily_df["precipitation_probability_max"],
            marker_color=bar_colors,
            text=[f"{int(r)}%" for r in daily_df["precipitation_probability_max"]],
            textposition='outside'
        ))
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b2bec3'),
            xaxis=dict(showgrid=False, color='#636e72'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#636e72', range=[0, 120]),
            margin=dict(l=0, r=0, t=20, b=0), height=240, showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ============================================================
    # TAB 5 — SMART TIPS
    # ============================================================
    with tab5:
        st.markdown("<p class='section-header'>Personalized Weather Tips</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#636e72; font-size:0.9em;'>AI-powered recommendations tailored to today's conditions in {}.".format(location['name']) + "</p>", unsafe_allow_html=True)

        if st.button("🔄 Refresh Tips", key="refresh_tips"):
            st.session_state.ai_tips = None

        if not st.session_state.get("ai_tips"):
            with st.spinner("Generating personalized tips..."):
                wear, activity, health = generate_smart_tips(
                    weather_data, air_data, unit_symbol, selected_unit
                )
                st.session_state.ai_tips = (wear, activity, health)

        wear, activity, health = st.session_state.ai_tips

        tips = [
            ("👗", "What to Wear", wear, "tip-wear", "#2980b9"),
            ("🏃", "Activity Suggestions", activity, "tip-activity", "#27ae60"),
            ("🏥", "Health & Safety", health, "tip-health", "#f39c12"),
        ]

        for em, title, content, css_class, color in tips:
            st.markdown(f"""
            <div class='tip-card {css_class}'>
                <p style='font-size:1em; font-weight:700; color:{color}; margin:0 0 8px 0;'>{em} {title}</p>
                <p style='margin:0; line-height:1.6; font-size:0.95em;'>{content}</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick reference card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Today at a Glance</p>", unsafe_allow_html=True)
        glance_cols = st.columns(3)
        rain_today = daily_df.iloc[0]['precipitation_probability_max']
        umbrella = "✅ Yes" if rain_today > 40 else "❌ No"
        outdoor = "✅ Great" if rain_today < 30 and aqi < 100 and current['temperature_2m'] < 36 else "⚠️ Moderate" if rain_today < 60 and aqi < 150 else "❌ Avoid"
        mask = "✅ Recommended" if aqi > 150 else "⚠️ Optional" if aqi > 100 else "❌ Not needed"

        for col, (label, val) in zip(glance_cols, [("☂️ Umbrella?", umbrella), ("🌳 Outdoor?", outdoor), ("😷 Mask?", mask)]):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <p style='color:#636e72; font-size:0.85em; margin:0;'>{label}</p>
                    <p style='font-size:1.1em; font-weight:600; margin:6px 0 0 0;'>{val}</p>
                </div>
                """, unsafe_allow_html=True)
