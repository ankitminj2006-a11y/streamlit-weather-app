import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os


from google import genai
from google.genai import types

# 👇 PASTE YOUR API KEY HERE (get it free from aistudio.google.com/app/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", AIzaSyB0eyztEnRbdZw_AG30_1AQhiqU1ErZkho)

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY != AIzaSyB0eyztEnRbdZw_AG30_1AQhiqU1ErZkho else None
MODEL_ID = "gemini-2.0-flash"

# --- Weather API (Free, no key needed) ---
BASE_URL = "https://api.open-meteo.com/v1/forecast"

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
        st.error(f"Error fetching weather data: {e}")
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

def build_weather_context(weather_data, air_data, location, unit_symbol, selected_unit):
    current = weather_data["current"]
    daily_df, _ = process_forecast_data(weather_data)
    aqi = air_data["current"]["us_aqi"]
    aqi_level, _ = get_aqi_level(aqi)
    weather_desc, _ = WMO_CODES.get(current["weather_code"], ("Unknown", ""))
    wind_unit = "km/h" if selected_unit == "metric" else "mph"
    forecast_lines = "".join([
        f"  - {row['day']}: {WMO_CODES.get(row['weather_code'], ('Unknown',''))[0]}, "
        f"High {row['temperature_2m_max']:.0f}{unit_symbol}, "
        f"Low {row['temperature_2m_min']:.0f}{unit_symbol}, "
        f"Rain {row['precipitation_probability_max']}%\n"
        for _, row in daily_df.head(4).iterrows()
    ])
    return f"""Weather for {location['name']}, {location.get('country', '')}:
- Condition: {weather_desc}
- Temperature: {current['temperature_2m']:.1f}{unit_symbol}
- Humidity: {current['relative_humidity_2m']}%
- Wind: {current['wind_speed_10m']} {wind_unit}
- Precipitation: {current['precipitation']:.1f} mm
- AQI: {aqi} ({aqi_level})
- Rain chance today: {daily_df.iloc[0]['precipitation_probability_max']}%
4-Day Forecast:
{forecast_lines}""".strip()


# ============================================================
# ✅ AI FUNCTIONS — Google AI Cookbook Style
# ============================================================

def ai_generate(prompt: str) -> str:
    """Basic text generation using Google AI Cookbook SDK."""
    if not client:
        return "⚠️ Please add your API key. Get it free at aistudio.google.com/app/apikey"
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return response.text


def ai_generate_with_search(prompt: str):
    """
    Google Search Grounding — Google AI Cookbook style.
    Fetches live results from Google Search in real time.
    """
    if not client:
        return "⚠️ Please add your API key. Get it free at aistudio.google.com/app/apikey", []

    # ✅ Cookbook-style Search Grounding Tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool]
        )
    )

    # Extract grounding sources
    sources = []
    try:
        candidate = response.candidates[0]
        if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
            for chunk in candidate.grounding_metadata.grounding_chunks:
                if chunk.web:
                    sources.append({
                        "title": chunk.web.title or "Source",
                        "url": chunk.web.uri or "#"
                    })
    except Exception:
        pass

    return response.text, sources


def ai_chat(messages: list, system_prompt: str) -> str:
    """Multi-turn chat using Google AI Cookbook SDK."""
    if not client:
        return "⚠️ Please add your API key. Get it free at aistudio.google.com/app/apikey"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="model", parts=[types.Part.from_text(text="Understood! Ready to help with weather questions.")])
    ]
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    response = client.models.generate_content(model=MODEL_ID, contents=contents)
    return response.text


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="PyWeather AI", layout="wide", page_icon="🌤️")

st.markdown("""
<style>
.chat-user {
    background: #e3f2fd; border-radius: 12px 12px 2px 12px;
    padding: 10px 14px; margin: 6px 0; text-align: right; font-size: 0.9em;
}
.chat-ai {
    background: #e8f5e9; border-radius: 12px 12px 12px 2px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.9em;
}
.badge {
    background: linear-gradient(135deg, #4285F4, #34A853);
    color: white; padding: 3px 12px; border-radius: 12px;
    font-size: 0.75em; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=80)
    st.title("PyWeather AI")
    st.caption("✅ Google AI Cookbook SDK")

    # API Key status
    if not client:
        st.error("⚠️ API Key Missing!\n\nGet FREE key:\naistudio.google.com/app/apikey\n\nThen add to Codespace secrets as GEMINI_API_KEY")
    else:
        st.success("✅ API Key Connected")

    if "saved_cities" not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"]

    city = st.text_input("Enter City Name", "Ghaziabad")

    def reset_ai_cache():
        for key in ["ai_summary", "ai_recommendations", "ai_news", "ai_context"]:
            st.session_state[key] = None
        st.session_state.ai_news_sources = []
        st.session_state.chat_history = []

    if st.button("Search City", use_container_width=True):
        st.session_state.city = city
        reset_ai_cache()

    st.write("---")
    st.write("**Favorite Cities:**")
    for saved_city in st.session_state.saved_cities:
        if st.button(saved_city, use_container_width=True, key=saved_city):
            st.session_state.city = saved_city
            reset_ai_cache()

    if "city" not in st.session_state:
        st.session_state.city = "Ghaziabad"

    st.write("---")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit = unit_options[st.radio("Temperature Unit", unit_options.keys())]
    unit_symbol = "°C" if selected_unit == "metric" else "°F"

    # Chat
    st.write("---")
    st.markdown("### 🤖 Ask AI")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history[-6:]:
        css = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about the weather...", placeholder="Will it rain tomorrow?", label_visibility="collapsed")
        send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and user_input:
        if "weather_data" in st.session_state:
            ctx = build_weather_context(
                st.session_state.weather_data, st.session_state.air_data,
                st.session_state.location, st.session_state.unit_symbol,
                st.session_state.selected_unit
            )
            system = f"You are PyWeather AI for {st.session_state.city}. Answer weather questions in 2-4 sentences.\n\nWeather:\n{ctx}"
            with st.spinner("Thinking..."):
                try:
                    reply = ai_chat(
                        st.session_state.chat_history + [{"role": "user", "content": user_input}],
                        system
                    )
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
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
    weather_context = build_weather_context(weather_data, air_data, location, unit_symbol, selected_unit)

    tab_today, tab_forecast, tab_rec, tab_news = st.tabs([
        "☀️ Today", "📅 10-Day Forecast", "🤖 AI Recommendations", "🔍 Live Weather News"
    ])

    # --- Tab 1: Today ---
    with tab_today:
        st.subheader(f"Current Conditions — {location['name']}, {location.get('country_code', '')}")

        if not st.session_state.get("ai_summary"):
            with st.spinner("✨ Generating AI summary..."):
                try:
                    st.session_state.ai_summary = ai_generate(
                        f"Write a friendly 2-3 sentence weather summary for {st.session_state.city}. "
                        f"Be conversational, paint a picture.\n\nData:\n{weather_context}"
                    )
                except Exception as e:
                    st.session_state.ai_summary = f"(Error: {e})"

        st.info(f"🤖 **AI Summary:** {st.session_state.ai_summary}")
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

    # --- Tab 3: AI Recommendations ---
    with tab_rec:
        st.header("🤖 AI Recommendations")
        st.caption("Powered by Google AI Cookbook — client.models.generate_content()")

        if st.button("🔄 Regenerate"):
            st.session_state.ai_recommendations = None

        if not st.session_state.get("ai_recommendations"):
            with st.spinner("Generating recommendations..."):
                try:
                    st.session_state.ai_recommendations = ai_generate(
                        f"""For {st.session_state.city} weather, provide:
**👗 What to Wear** — 2-3 sentences of clothing advice.
**🏃 Activity Suggestions** — 2-3 activity ideas for this weather.
**🏥 Health & Safety** — 1-2 sentences, mention AQI if relevant.
Weather Data:\n{weather_context}"""
                    )
                except Exception as e:
                    st.session_state.ai_recommendations = f"(Error: {e})"

        st.markdown(st.session_state.get("ai_recommendations", ""))

        st.write("---")
        st.subheader("⚡ Quick Questions")
        quick_qs = ["Should I carry an umbrella?", "Good weather for a run?", "Safe air quality for kids?"]
        q_cols = st.columns(3)
        for i, q in enumerate(quick_qs):
            if q_cols[i].button(q, use_container_width=True):
                with st.spinner("Asking AI..."):
                    try:
                        ans = ai_generate(f"Weather:\n{weather_context}\n\nQuestion: {q}\nAnswer in 2-3 sentences.")
                        st.info(f"**{q}**\n\n{ans}")
                    except Exception as e:
                        st.error(str(e))

    # --- Tab 4: Live Weather News ---
    with tab_news:
        st.header("🔍 Live Weather News")
        st.markdown('<span class="badge">🌐 Google Search Grounding — Google AI Cookbook</span>', unsafe_allow_html=True)
        st.caption("Uses `types.Tool(google_search=types.GoogleSearch())` to fetch live results from Google.")

        with st.expander("📖 Cookbook Code Used"):
            st.code("""from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_KEY")  # aistudio.google.com/app/apikey

# Google Search Grounding Tool
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Latest weather alerts for Delhi?",
    config=types.GenerateContentConfig(
        tools=[grounding_tool]
    )
)

print(response.text)

# Access live sources:
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(chunk.web.title, chunk.web.uri)
""", language="python")

        if st.button("🔄 Refresh Live News"):
            st.session_state.ai_news = None
            st.session_state.ai_news_sources = []
            st.session_state.ai_context = None

        # Live alerts
        st.subheader(f"⚠️ Live Alerts & News — {st.session_state.city}")
        if not st.session_state.get("ai_news"):
            with st.spinner("🌐 Searching Google for live weather news..."):
                try:
                    news, sources = ai_generate_with_search(
                        f"What are the latest weather alerts, warnings, and news for {st.session_state.city} right now? "
                        f"List 3-5 key bullet points. Be factual and current."
                    )
                    st.session_state.ai_news = news
                    st.session_state.ai_news_sources = sources
                except Exception as e:
                    st.session_state.ai_news = f"(Search error: {e})"
                    st.session_state.ai_news_sources = []

        st.markdown(st.session_state.get("ai_news", ""))

        if st.session_state.get("ai_news_sources"):
            st.write("---")
            st.caption("**Live Sources from Google Search:**")
            for src in st.session_state.ai_news_sources[:5]:
                st.markdown(f"🔗 [{src['title']}]({src['url']})")

        st.write("---")

        # Seasonal context
        st.subheader("🌍 Seasonal Weather Context")
        if not st.session_state.get("ai_context"):
            with st.spinner("🌐 Fetching context from Google..."):
                try:
                    ctx_text, _ = ai_generate_with_search(
                        f"Is the current weather in {st.session_state.city} unusual for this time of year? "
                        f"What seasonal patterns are expected? Answer in 3-4 sentences."
                    )
                    st.session_state.ai_context = ctx_text
                except Exception as e:
                    st.session_state.ai_context = f"(Error: {e})"

        st.info(st.session_state.get("ai_context", ""))

        st.write("---")

        # Custom live search
        st.subheader("🔎 Custom Live Search")
        with st.form("search_form", clear_on_submit=True):
            query = st.text_input(
                "Search...",
                placeholder=f"Is there a heatwave in {st.session_state.city}?",
                label_visibility="collapsed"
            )
            search_btn = st.form_submit_button("🔍 Search Google")

        if search_btn and query:
            with st.spinner("🌐 Searching Google in real time..."):
                try:
                    result, sources = ai_generate_with_search(query)
                    st.success(result)
                    if sources:
                        st.caption("**Sources:**")
                        for src in sources[:4]:
                            st.markdown(f"🔗 [{src['title']}]({src['url']})")
                except Exception as e:
                    st.error(f"Search error: {e}")
