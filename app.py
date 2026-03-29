import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from transformers import pipeline
import re

# ============================================================
# ✅ Intel Unnati Project 3 — News Summarizer (NO API KEY!)
# ✅ Intel Unnati Project 15 — Performance Analyzer (NO API KEY!)
# Uses HuggingFace local models — runs completely free & offline
# ============================================================

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# --- Load HuggingFace Models (cached so they load only once) ---
@st.cache_resource
def load_summarizer():
    """Project 3: News/Text Summarizer — facebook/bart-large-cnn"""
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_classifier():
    """Project 3: Zero-shot classifier for weather condition analysis"""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_text_generator():
    """Text generation for recommendations — no API needed"""
    return pipeline("text2text-generation", model="google/flan-t5-base")


# --- Weather API ---
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
        st.error(f"Error fetching weather: {e}")
        return None, None, None


WMO_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅️"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "💧"), 53: ("Moderate drizzle", "💧"), 55: ("Dense drizzle", "💧"),
    61: ("Slight rain", "🌧️"), 63: ("Moderate rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    71: ("Slight snow fall", "❄️"), 73: ("Moderate snow fall", "❄️"), 75: ("Heavy snow fall", "❄️"),
    80: ("Slight rain showers", "🌦️"), 81: ("Moderate rain showers", "🌦️"), 82: ("Violent rain showers", "🌦️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm with light hail", "⛈️"), 99: ("Thunderstorm with heavy hail", "⛈️"),
}

def get_aqi_level(aqi):
    if aqi <= 50: return "Good", "green"
    elif aqi <= 100: return "Moderate", "yellow"
    elif aqi <= 150: return "Unhealthy (SG)", "orange"
    elif aqi <= 200: return "Unhealthy", "red"
    elif aqi <= 300: return "Very Unhealthy", "purple"
    else: return "Hazardous", "maroon"

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
    """Builds a long natural language text about the weather — used for summarization."""
    current = weather_data["current"]
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    aqi_level, _ = get_aqi_level(aqi)
    weather_desc, _ = WMO_CODES.get(current["weather_code"], ("Unknown", ""))
    wind_unit = "km/h" if selected_unit == "metric" else "mph"

    forecast_text = " ".join([
        f"On {row['day']}, expect {WMO_CODES.get(row['weather_code'], ('Unknown',''))[0]} "
        f"with a high of {row['temperature_2m_max']:.0f}{unit_symbol} "
        f"and a low of {row['temperature_2m_min']:.0f}{unit_symbol}, "
        f"with a {row['precipitation_probability_max']}% chance of rain."
        for _, row in daily_df.head(5).iterrows()
    ])

    return (
        f"Current weather in {location['name']}: {weather_desc} with a temperature of "
        f"{current['temperature_2m']:.1f}{unit_symbol}. Humidity is {current['relative_humidity_2m']}% "
        f"and wind speed is {current['wind_speed_10m']} {wind_unit}. "
        f"Precipitation today is {current['precipitation']:.1f} mm. "
        f"The air quality index is {aqi}, which is considered {aqi_level}. "
        f"Rain probability today is {daily_df.iloc[0]['precipitation_probability_max']}%. "
        f"Here is the forecast for the coming days: {forecast_text}"
    )


# ============================================================
# ✅ PROJECT 3 — AI Weather Summarizer (HuggingFace BART)
# ============================================================
def get_ai_summary(weather_text: str) -> str:
    """Summarizes weather data using BART model — no API key needed."""
    try:
        summarizer = load_summarizer()
        # BART works best with 100-1000 words
        result = summarizer(
            weather_text,
            max_length=80,
            min_length=30,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception as e:
        return f"(Summary error: {e})"


# ============================================================
# ✅ PROJECT 3 — Weather Condition Classifier (Zero-Shot)
# ============================================================
def classify_weather_severity(weather_text: str) -> dict:
    """Classifies weather severity using zero-shot classification."""
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
        return {"label": "unknown", "score": 0, "all": []}


# ============================================================
# ✅ PROJECT 15 — Weather Performance Analyzer
# ============================================================
def analyze_weather_patterns(daily_df, unit_symbol) -> dict:
    """Analyzes 10-day weather patterns — no AI model needed, pure data analysis."""
    temps_max = daily_df["temperature_2m_max"].tolist()
    temps_min = daily_df["temperature_2m_min"].tolist()
    rain_chances = daily_df["precipitation_probability_max"].tolist()

    avg_max = sum(temps_max) / len(temps_max)
    avg_min = sum(temps_min) / len(temps_min)
    avg_rain = sum(rain_chances) / len(rain_chances)
    max_temp = max(temps_max)
    min_temp = min(temps_min)
    rainy_days = sum(1 for r in rain_chances if r > 50)
    temp_swing = max_temp - min_temp

    insights = []

    if avg_rain > 60:
        insights.append("🌧️ High rainfall expected — carry an umbrella all week.")
    elif avg_rain > 30:
        insights.append("🌦️ Moderate rain chances — be prepared for occasional showers.")
    else:
        insights.append("☀️ Mostly dry week ahead — great for outdoor activities.")

    if temp_swing > 10:
        insights.append(f"🌡️ Large temperature swing of {temp_swing:.0f}{unit_symbol} — dress in layers.")
    else:
        insights.append(f"🌡️ Stable temperatures throughout the week.")

    if max_temp > 35:
        insights.append("🔥 Heatwave risk — stay hydrated and avoid midday sun.")
    elif min_temp < 5:
        insights.append("🥶 Cold snap expected — bundle up warmly.")

    if rainy_days >= 7:
        insights.append(f"☔ Rain expected on {rainy_days} out of 10 days.")
    elif rainy_days == 0:
        insights.append("🌈 Zero rain days forecast — perfect outdoor week!")

    return {
        "avg_max": avg_max,
        "avg_min": avg_min,
        "avg_rain": avg_rain,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "rainy_days": rainy_days,
        "temp_swing": temp_swing,
        "insights": insights
    }


# ============================================================
# ✅ PROJECT 8 — Weather Flashcard Generator (Flan-T5)
# ============================================================
def generate_weather_tips(weather_text: str, city: str) -> list:
    """Generates weather tips using Flan-T5 — no API key needed."""
    try:
        generator = load_text_generator()
        questions = [
            f"What clothing should someone wear in {city} based on this weather: {weather_text[:300]}?",
            f"What outdoor activities are suitable for this weather in {city}: {weather_text[:300]}?",
            f"What health precautions should be taken for this weather: {weather_text[:300]}?",
        ]
        tips = []
        for q in questions:
            result = generator(q, max_length=80)
            tips.append(result[0]["generated_text"])
        return tips
    except Exception as e:
        return [f"(Tip generation error: {e})"]


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="PyWeather AI", layout="wide", page_icon="🌤️")

st.markdown("""
<style>
.badge-blue {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    color: white; padding: 3px 12px; border-radius: 12px;
    font-size: 0.75em; font-weight: 700;
}
.badge-green {
    background: linear-gradient(135deg, #34A853, #1e7e34);
    color: white; padding: 3px 12px; border-radius: 12px;
    font-size: 0.75em; font-weight: 700;
}
.insight-box {
    background: #1e1e2e; border-left: 4px solid #4285F4;
    padding: 10px 16px; border-radius: 8px; margin: 6px 0;
    font-size: 0.95em;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=80)
    st.title("PyWeather AI")
    st.success("✅ No API Key Needed!")
    st.caption("Powered by HuggingFace local models")

    st.markdown("""
    **Intel Unnati Projects Used:**
    - 📰 Project 3: News Summarizer
    - 📊 Project 15: Performance Analyzer
    - 📚 Project 8: Flashcard/Tips Generator
    """)

    if "saved_cities" not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"]

    city = st.text_input("Enter City Name", "Ghaziabad")

    def reset_cache():
        for key in ["ai_summary", "ai_severity", "ai_tips", "ai_analysis"]:
            st.session_state[key] = None

    if st.button("Search City", use_container_width=True):
        st.session_state.city = city
        reset_cache()

    st.write("---")
    st.write("**Favorite Cities:**")
    for saved_city in st.session_state.saved_cities:
        if st.button(saved_city, use_container_width=True, key=saved_city):
            st.session_state.city = saved_city
            reset_cache()

    if "city" not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.write("---")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit = unit_options[st.radio("Temperature Unit", unit_options.keys())]
    unit_symbol = "°C" if selected_unit == "metric" else "°F"


# --- Main Content ---
st.header(f"🌤️ Weather in {st.session_state.city}")

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
    st.info("Welcome! ☀️ Enter a city and click 'Search City' to start.")
else:
    weather_data = st.session_state.weather_data
    air_data = st.session_state.air_data
    unit_symbol = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location = st.session_state.location

    current = weather_data["current"]
    daily_df, hourly_df = process_forecast_data(weather_data)
    weather_text = build_weather_text(weather_data, air_data, location, unit_symbol, selected_unit)

    tab_today, tab_forecast, tab_summary, tab_analysis, tab_tips = st.tabs([
        "☀️ Today",
        "📅 10-Day Forecast",
        "📰 AI Summary (Project 3)",
        "📊 Pattern Analysis (Project 15)",
        "📚 Weather Tips (Project 8)"
    ])

    # --- Tab 1: Today ---
    with tab_today:
        st.subheader(f"Current Conditions — {location['name']}, {location.get('country_code', '')}")
        st.write("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<h1 style='font-size:5em;margin:0'>{current['temperature_2m']:.0f}{unit_symbol}</h1>", unsafe_allow_html=True)
            desc, icon = WMO_CODES.get(current["weather_code"], ("Unknown", "❓"))
            st.markdown(f"**{desc} {icon}**")
        with col2:
            st.markdown(f"Min/Max: **{daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}** / **{daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol}**")
            st.markdown(f"Precipitation: **{current['precipitation']:.1f} mm**")
            st.markdown(f"Rain Chance Today: **{daily_df.iloc[0]['precipitation_probability_max']}%**")
            st.write("---")
            aqi = air_data["current"]["us_aqi"]
            aqi_level, aqi_color = get_aqi_level(aqi)
            st.markdown(f"Air Quality (AQI): <span style='color:{aqi_color};font-weight:bold'>{aqi_level} ({aqi})</span>", unsafe_allow_html=True)
            st.caption("0–50 Good · 51–100 Moderate · 101–150 Unhealthy (SG) · 151+ Unhealthy")
        st.write("---")
        cols = st.columns(3)
        cols[0].metric("Humidity", f"{current['relative_humidity_2m']}%")
        cols[1].metric("Wind Speed", f"{current['wind_speed_10m']} {'km/h' if selected_unit == 'metric' else 'mph'}")
        cols[2].metric("Coordinates", f"{location['latitude']:.2f}°, {location['longitude']:.2f}°")

    # --- Tab 2: Forecast ---
    with tab_forecast:
        st.header("10-Day Forecast")
        for _, row in daily_df.iterrows():
            cols = st.columns([1, 1, 2, 2])
            cols[0].markdown(f"**{row['day']}**")
            d, i = WMO_CODES.get(row["weather_code"], ("Unknown", "❓"))
            cols[1].markdown(f"**{i}**")
            cols[2].markdown(d)
            cols[3].markdown(f"**{row['temperature_2m_max']:.0f}{unit_symbol}** / {row['temperature_2m_min']:.0f}{unit_symbol}")
        st.write("---")
        plot_choice = st.selectbox("Select graph:", ["Temperature", "Rain Chance (%)", "Humidity"])
        fig = px.line(hourly_df, y=plot_choice, title=f"{plot_choice} Trend")
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: AI Summary (Project 3) ---
    with tab_summary:
        st.header("📰 AI Weather Summarizer")
        st.markdown('<span class="badge-blue">📰 Intel Unnati Project 3 — News Summarizer</span>', unsafe_allow_html=True)
        st.caption("Uses HuggingFace `facebook/bart-large-cnn` locally — NO API KEY required.")

        with st.expander("📄 Raw Weather Text (Input to AI)"):
            st.write(weather_text)

        if st.button("🔄 Regenerate Summary"):
            st.session_state.ai_summary = None
            st.session_state.ai_severity = None

        # BART Summary
        st.subheader("🤖 AI-Generated Summary")
        if not st.session_state.get("ai_summary"):
            with st.spinner("🧠 BART model summarizing weather data..."):
                st.session_state.ai_summary = get_ai_summary(weather_text)

        st.info(f"📋 **Summary:** {st.session_state.ai_summary}")

        st.write("---")

        # Zero-Shot Classification
        st.subheader("🚦 Weather Severity Classification")
        st.caption("Zero-shot classification using `facebook/bart-large-mnli`")

        if not st.session_state.get("ai_severity"):
            with st.spinner("🧠 Classifying weather severity..."):
                st.session_state.ai_severity = classify_weather_severity(weather_text)

        severity = st.session_state.ai_severity
        label = severity.get("label", "unknown")
        score = severity.get("score", 0)

        color_map = {
            "safe and pleasant": "🟢",
            "mild caution advised": "🟡",
            "severe weather warning": "🟠",
            "extreme danger": "🔴"
        }
        emoji = color_map.get(label, "⚪")
        st.markdown(f"### {emoji} **{label.title()}** ({score*100:.1f}% confidence)")

        # Show all scores as a bar chart
        if severity.get("all"):
            labels_list = [x[0] for x in severity["all"]]
            scores_list = [x[1] * 100 for x in severity["all"]]
            fig = px.bar(
                x=scores_list, y=labels_list,
                orientation="h",
                title="Severity Classification Confidence",
                labels={"x": "Confidence (%)", "y": "Category"},
                color=scores_list,
                color_continuous_scale="RdYlGn_r"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: Pattern Analysis (Project 15) ---
    with tab_analysis:
        st.header("📊 Weather Pattern Analyzer")
        st.markdown('<span class="badge-green">📊 Intel Unnati Project 15 — Performance Analyzer</span>', unsafe_allow_html=True)
        st.caption("Analyzes 10-day weather patterns and generates insights — NO API KEY required.")

        if not st.session_state.get("ai_analysis"):
            st.session_state.ai_analysis = analyze_weather_patterns(daily_df, unit_symbol)

        analysis = st.session_state.ai_analysis

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg High", f"{analysis['avg_max']:.1f}{unit_symbol}")
        col2.metric("Avg Low", f"{analysis['avg_min']:.1f}{unit_symbol}")
        col3.metric("Rainy Days", f"{analysis['rainy_days']} / 10")
        col4.metric("Temp Swing", f"{analysis['temp_swing']:.1f}{unit_symbol}")

        st.write("---")

        # AI Insights
        st.subheader("🔍 AI-Generated Insights")
        for insight in analysis["insights"]:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.write("---")

        # Temperature trend chart
        st.subheader("📈 10-Day Temperature Range")
        temp_fig = px.area(
            daily_df,
            x="day",
            y=["temperature_2m_max", "temperature_2m_min"],
            title="Max & Min Temperature Forecast",
            labels={"value": f"Temperature ({unit_symbol})", "day": "Day", "variable": ""},
            color_discrete_map={
                "temperature_2m_max": "#FF6B6B",
                "temperature_2m_min": "#4ECDC4"
            }
        )
        st.plotly_chart(temp_fig, use_container_width=True)

        # Rain probability chart
        st.subheader("🌧️ Rain Probability Trend")
        rain_fig = px.bar(
            daily_df,
            x="day",
            y="precipitation_probability_max",
            title="Daily Rain Probability (%)",
            color="precipitation_probability_max",
            color_continuous_scale="Blues",
            labels={"precipitation_probability_max": "Rain Chance (%)", "day": "Day"}
        )
        rain_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(rain_fig, use_container_width=True)

    # --- Tab 5: Weather Tips (Project 8) ---
    with tab_tips:
        st.header("📚 AI Weather Tips Generator")
        st.markdown('<span class="badge-blue">📚 Intel Unnati Project 8 — Flashcard Generator</span>', unsafe_allow_html=True)
        st.caption("Uses HuggingFace `google/flan-t5-base` locally — NO API KEY required.")

        if st.button("🔄 Generate New Tips"):
            st.session_state.ai_tips = None

        if not st.session_state.get("ai_tips"):
            with st.spinner("🧠 Flan-T5 generating weather tips..."):
                st.session_state.ai_tips = generate_weather_tips(weather_text, st.session_state.city)

        tips = st.session_state.ai_tips
        tip_titles = ["👗 What to Wear", "🏃 Activity Suggestions", "🏥 Health & Safety"]
        tip_colors = ["#e3f2fd", "#e8f5e9", "#fff3e0"]

        for i, (title, tip) in enumerate(zip(tip_titles, tips)):
            st.markdown(f"""
            <div style='background:{tip_colors[i]};padding:16px;border-radius:10px;margin:10px 0;color:#000'>
                <strong>{title}</strong><br>{tip}
            </div>
            """, unsafe_allow_html=True)
