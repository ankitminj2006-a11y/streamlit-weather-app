import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# --- API Configuration (Open-Meteo) ---
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# --- Gemini AI Setup ---
# Get your FREE key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyBZxOyWKGQqkcUtQUH7K6J5u4V_M1DDch8")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")# Free & fast model
else:
    model = None


# --- Helper Functions for API Calls ---

def get_weather_data(city, units):
    try:
        geo_params = {"name": city, "count": 1}
        geo_response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=geo_params)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if "results" not in geo_data or len(geo_data['results']) == 0:
            st.error(f"City not found: {city}. Please try again.")
            return None, None, None

        location = geo_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        timezone = location['timezone']

        unit_temp = "celsius" if units == "metric" else "fahrenheit"
        unit_wind = "kmh" if units == "metric" else "mph"

        weather_params = {
            "latitude": lat, "longitude": lon, "timezone": timezone,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "temperature_unit": unit_temp, "wind_speed_unit": unit_wind, "forecast_days": 10
        }

        weather_response = requests.get(BASE_URL, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        air_params = {"latitude": lat, "longitude": lon, "current": "us_aqi"}
        air_response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=air_params)
        air_response.raise_for_status()
        air_data = air_response.json()

        return weather_data, air_data, location

    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error: {err}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")

    return None, None, None


# --- WMO Weather Code Mapping ---
WMO_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅️"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "💧"), 53: ("Moderate drizzle", "💧"), 55: ("Dense drizzle", "💧"),
    56: ("Light freezing drizzle", "❄️💧"), 57: ("Dense freezing drizzle", "❄️💧"),
    61: ("Slight rain", "🌧️"), 63: ("Moderate rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "❄️🌧️"), 67: ("Heavy freezing rain", "❄️🌧️"),
    71: ("Slight snow fall", "❄️"), 73: ("Moderate snow fall", "❄️"), 75: ("Heavy snow fall", "❄️"),
    77: ("Snow grains", "❄️"), 80: ("Slight rain showers", "🌦️"), 81: ("Moderate rain showers", "🌦️"),
    82: ("Violent rain showers", "🌦️"), 85: ("Slight snow showers", "❄️🌦️"),
    86: ("Heavy snow showers", "❄️🌦️"), 95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with light hail", "⛈️"), 99: ("Thunderstorm with heavy hail", "⛈️"),
}

def get_aqi_level(aqi):
    if aqi <= 50: return "Good", "green"
    elif aqi <= 100: return "Moderate", "yellow"
    elif aqi <= 150: return "Unhealthy (SG)", "orange"
    elif aqi <= 200: return "Unhealthy", "red"
    elif aqi <= 300: return "Very Unhealthy", "purple"
    else: return "Hazardous", "maroon"

def process_forecast_data(forecast_data):
    daily_df = pd.DataFrame(forecast_data['daily'])
    daily_df['date'] = pd.to_datetime(daily_df['time'])
    daily_df['day'] = daily_df['date'].dt.strftime("%a, %b %d")

    hourly_df = pd.DataFrame(forecast_data['hourly'])
    hourly_df['Timestamp'] = pd.to_datetime(hourly_df['time'])
    hourly_df.rename(columns={
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "precipitation_probability": "Rain Chance (%)"
    }, inplace=True)

    return daily_df, hourly_df.set_index('Timestamp')


# --- AI Helper Functions (Gemini) ---

def build_weather_context(weather_data, air_data, location, unit_symbol, selected_unit):
    current = weather_data['current']
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data['current']['us_aqi']
    aqi_level, _ = get_aqi_level(aqi)
    weather_desc, weather_icon = WMO_CODES.get(current['weather_code'], ("Unknown", ""))
    wind_unit = "km/h" if selected_unit == "metric" else "mph"

    forecast_lines = []
    for _, row in daily_df.head(4).iterrows():
        desc, icon = WMO_CODES.get(row['weather_code'], ("Unknown", ""))
        forecast_lines.append(
            f"  - {row['day']}: {desc} {icon}, High {row['temperature_2m_max']:.0f}{unit_symbol}, "
            f"Low {row['temperature_2m_min']:.0f}{unit_symbol}, "
            f"Rain chance {row['precipitation_probability_max']}%\n"
        )

    context = f"""Current weather for {location['name']}, {location.get('country', '')}:
- Condition: {weather_desc} {weather_icon}
- Temperature: {current['temperature_2m']:.1f}{unit_symbol}
- Humidity: {current['relative_humidity_2m']}%
- Wind Speed: {current['wind_speed_10m']} {wind_unit}
- Precipitation: {current['precipitation']:.1f} mm
- Air Quality Index (US AQI): {aqi} — {aqi_level}
- Today's rain chance: {daily_df.iloc[0]['precipitation_probability_max']}%

4-Day Forecast:
{"".join(forecast_lines)}""".strip()
    return context


def get_ai_weather_summary(weather_context, city):
    if not model:
        return "⚠️ Add your GEMINI_API_KEY to enable AI summaries."
    prompt = f"""You are a friendly weather presenter. Write a concise, engaging 2-3 sentence 
weather summary for {city}. Be conversational and highlight what matters most for someone 
planning their day. Don't just list facts — paint a picture.

Weather Data:
{weather_context}

Write only the summary, no preamble."""
    response = model.generate_content(prompt)
    return response.text


def get_ai_recommendations(weather_context, city):
    if not model:
        return "⚠️ Add your GEMINI_API_KEY in secrets to enable AI recommendations."
    prompt = f"""Based on the weather data below for {city}, provide:

**👗 What to Wear**
2-3 sentences with specific clothing advice.

**🏃 Activity Suggestions**
2-3 sentences with 2-3 specific activity ideas suited to the weather.

**🏥 Health & Safety Tips**
1-2 sentences about any weather-related health considerations, especially regarding air quality.

Be specific, practical, and friendly.

Weather Data:
{weather_context}"""
    response = model.generate_content(prompt)
    return response.text


def get_ai_chat_response(user_message, weather_context, city, chat_history):
    if not model:
        return "⚠️ Add your GEMINI_API_KEY in secrets to enable AI chat."

    # Build conversation history for Gemini
    history = []
    for msg in chat_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    full_message = f"""You are PyWeather AI, a friendly weather assistant for {city}.
You have access to current real-time weather data. Answer questions about the weather,
give advice, or chat about weather-related topics. Be concise (2-4 sentences) and conversational.

Current Weather Data:
{weather_context}

User question: {user_message}"""

    response = chat.send_message(full_message)
    return response.text


# --- Main Application UI ---
st.set_page_config(page_title="PyWeather AI", layout="wide", page_icon="🌤️")

st.markdown("""
<style>
.chat-user {
    background: #e3f2fd;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    text-align: right;
    font-size: 0.9em;
}
.chat-ai {
    background: #e8f5e9;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=80)
    st.title("PyWeather AI")
    st.caption("✨ Powered by Google Gemini (Free)")

    if not GEMINI_API_KEY:
        st.warning("⚠️ No Gemini API key found!\n\nGet a FREE key at:\naistudio.google.com/app/apikey\n\nThen add it to your secrets.")

    if 'saved_cities' not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"]

    city = st.text_input("Enter City Name", "Ghaziabad")

    if st.button("Search City", use_container_width=True):
        st.session_state.city = city
        st.session_state.ai_summary = None
        st.session_state.ai_recommendations = None
        st.session_state.chat_history = []

    st.write("---")
    st.write("**Favorite Cities:**")
    for saved_city in st.session_state.saved_cities:
        if st.button(saved_city, use_container_width=True, key=saved_city):
            st.session_state.city = saved_city
            st.session_state.ai_summary = None
            st.session_state.ai_recommendations = None
            st.session_state.chat_history = []

    if 'city' not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.write("---")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit_label = st.radio("Temperature Unit", unit_options.keys())
    selected_unit = unit_options[selected_unit_label]
    unit_symbol = "°C" if selected_unit == "metric" else "°F"

    # --- AI Chat Widget ---
    st.write("---")
    st.markdown("### 🤖 Ask Gemini AI")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask about the weather...",
            placeholder="Will it rain tomorrow?",
            label_visibility="collapsed"
        )
        send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and user_input:
        if "weather_data" in st.session_state:
            weather_context = build_weather_context(
                st.session_state.weather_data, st.session_state.air_data,
                st.session_state.location, st.session_state.unit_symbol,
                st.session_state.selected_unit
            )
            with st.spinner("Thinking..."):
                try:
                    ai_response = get_ai_chat_response(
                        user_input, weather_context,
                        st.session_state.city, st.session_state.chat_history
                    )
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {e}")
        else:
            st.warning("Search for a city first!")


# --- Main Content ---
st.header(f"Weather in {st.session_state.city}")

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
else:
    st.info("Please enter a city name in the sidebar.")

if "weather_data" not in st.session_state:
    st.info("Welcome! ☀️ Enter a city and click 'Search City' to start.")
else:
    weather_data = st.session_state.weather_data
    air_data = st.session_state.air_data
    unit_symbol = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location = st.session_state.location

    current = weather_data['current']
    daily_df, hourly_df = process_forecast_data(weather_data)
    weather_context = build_weather_context(weather_data, air_data, location, unit_symbol, selected_unit)

    tab_today, tab_forecast, tab_recommendations = st.tabs(
        ["☀️ Today", "📅 10-Day Forecast", "🤖 AI Recommendations"]
    )

    # --- Tab 1: Today ---
    with tab_today:
        st.subheader(f"Current Conditions — {location['name']}, {location.get('country_code', '')}")

        if 'ai_summary' not in st.session_state or st.session_state.ai_summary is None:
            with st.spinner("✨ Generating AI summary..."):
                try:
                    st.session_state.ai_summary = get_ai_weather_summary(weather_context, st.session_state.city)
                except Exception as e:
                    st.session_state.ai_summary = f"(AI summary error: {e})"

        st.info(f"🤖 **AI Summary:** {st.session_state.ai_summary}")
        st.write("---")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"<h1 style='font-size:5em;margin:0'>{current['temperature_2m']:.0f}{unit_symbol}</h1>",
                unsafe_allow_html=True
            )
            weather_desc, weather_icon = WMO_CODES.get(current['weather_code'], ("Unknown", "❓"))
            st.markdown(f"**{weather_desc} {weather_icon}**")

        with col2:
            st.markdown(f"Min/Max: **{daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}** / **{daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol}**")
            st.markdown(f"Precipitation: **{current['precipitation']:.1f} mm**")
            st.markdown(f"Rain Chance Today: **{daily_df.iloc[0]['precipitation_probability_max']}%**")
            st.write("---")
            aqi = air_data['current']['us_aqi']
            aqi_level, aqi_color = get_aqi_level(aqi)
            st.markdown(
                f"Air Quality (US AQI): <span style='color:{aqi_color};font-weight:bold'>{aqi_level} ({aqi})</span>",
                unsafe_allow_html=True
            )
            st.caption("0–50 Good · 51–100 Moderate · 101–150 Unhealthy (SG) · 151+ Unhealthy")

        st.write("---")
        st.subheader("Additional Details")
        cols = st.columns(3)
        cols[0].metric("Humidity", f"{current['relative_humidity_2m']}%")
        wind_label = "km/h" if selected_unit == "metric" else "mph"
        cols[1].metric("Wind Speed", f"{current['wind_speed_10m']} {wind_label}")
        cols[2].metric("Coordinates", f"{location['latitude']:.2f}°, {location['longitude']:.2f}°")

    # --- Tab 2: Forecast ---
    with tab_forecast:
        st.header("10-Day Forecast")
        st.subheader("Daily Summary")

        for _, row in daily_df.iterrows():
            cols = st.columns([1, 1, 2, 2])
            cols[0].markdown(f"**{row['day']}**")
            day_desc, day_icon = WMO_CODES.get(row['weather_code'], ("Unknown", "❓"))
            cols[1].markdown(f"**{day_icon}**")
            cols[2].markdown(day_desc)
            cols[3].markdown(f"**{row['temperature_2m_max']:.0f}{unit_symbol}** / {row['temperature_2m_min']:.0f}{unit_symbol}")

        st.write("---")
        st.subheader("Hourly Trends (Next 10 Days)")
        plot_choice = st.selectbox("Select graph:", ["Temperature", "Rain Chance (%)", "Humidity"])
        fig = px.line(hourly_df, y=plot_choice, title=f"{plot_choice} Trend", markers=False)
        fig.update_layout(xaxis_title="Date & Time", yaxis_title=plot_choice)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: AI Recommendations ---
    with tab_recommendations:
        st.header("🤖 AI-Powered Recommendations")
        st.caption("Personalized insights from Google Gemini based on real-time weather data")

        if st.button("🔄 Regenerate Recommendations"):
            st.session_state.ai_recommendations = None

        if 'ai_recommendations' not in st.session_state or st.session_state.ai_recommendations is None:
            with st.spinner("🤖 Gemini is analyzing the weather for you..."):
                try:
                    st.session_state.ai_recommendations = get_ai_recommendations(
                        weather_context, st.session_state.city
                    )
                except Exception as e:
                    st.session_state.ai_recommendations = f"(AI recommendations error: {e})"

        st.markdown(st.session_state.ai_recommendations)

        st.write("---")
        st.subheader("⚡ Quick Questions")

        quick_questions = [
            "Should I carry an umbrella?",
            "Is it good weather for a run outside?",
            "Is the air quality safe for kids today?"
        ]
        q_cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            if q_cols[i].button(question, use_container_width=True):
                with st.spinner("Getting answer..."):
                    try:
                        answer = get_ai_chat_response(question, weather_context, st.session_state.city, [])
                        st.info(f"**{question}**\n\n{answer}")
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.caption("💬 For a full conversation, use the **Ask Gemini AI** panel in the sidebar.")
