import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import random

# --- API Configuration (Open-Meteo) ---
# No API key needed!
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# --- Helper Functions for API Calls ---

def get_weather_data(city, units):
    """
    Fetches 10-day forecast and air quality data from Open-Meteo.
    We first need to get lat/lon from a geocoding API.
    """
    try:
        # 1. Geocoding: Get lat/lon for the city
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
        
        # 2. Weather Forecast
        unit_temp = "celsius" if units == "metric" else "fahrenheit"
        unit_wind = "kmh" if units == "metric" else "mph"

        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "temperature_unit": unit_temp,
            "wind_speed_unit": unit_wind,
            "forecast_days": 10 # Get 10 days of forecast data
        }
        
        weather_response = requests.get(BASE_URL, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # 3. Air Quality
        air_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "us_aqi"
        }
        air_response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=air_params)
        air_response.raise_for_status()
        air_data = air_response.json()

        return weather_data, air_data, location
        
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error: {err}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
    
    return None, None, None

# --- WMO Weather Code to Icon/Description Mapping ---
WMO_CODES = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅️"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "💧"),
    53: ("Moderate drizzle", "💧"),
    55: ("Dense drizzle", "💧"),
    56: ("Light freezing drizzle", "❄️💧"),
    57: ("Dense freezing drizzle", "❄️💧"),
    61: ("Slight rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "❄️🌧️"),
    67: ("Heavy freezing rain", "❄️🌧️"),
    71: ("Slight snow fall", "❄️"),
    73: ("Moderate snow fall", "❄️"),
    75: ("Heavy snow fall", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌦️"),
    82: ("Violent rain showers", "🌦️"),
    85: ("Slight snow showers", "❄️🌦️"),
    86: ("Heavy snow showers", "❄️🌦️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with light hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}

def get_aqi_level(aqi):
    """Returns a human-readable AQI level and color."""
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy (SG)", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def process_forecast_data(forecast_data):
    """Processes the daily forecast data for plotting and display."""
    daily_df = pd.DataFrame(forecast_data['daily'])
    # Convert time to datetime objects
    daily_df['date'] = pd.to_datetime(daily_df['time'])
    daily_df['day'] = daily_df['date'].dt.strftime("%a, %b %d")
    
    # Process hourly data for graph
    hourly_df = pd.DataFrame(forecast_data['hourly'])
    hourly_df['Timestamp'] = pd.to_datetime(hourly_df['time'])
    hourly_df.rename(columns={
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "precipitation_probability": "Rain Chance (%)"
    }, inplace=True)
    
    return daily_df, hourly_df.set_index('Timestamp')


# --- AI/Logic Functions ---

def get_weather_mood(weather_code, temp, unit):
    """Returns a creative "Weather Mood" string."""
    temp_c = temp if unit == 'metric' else (temp - 32) * 5/9
    condition = WMO_CODES.get(weather_code, ("Unknown", ""))[0].lower()
    
    if "rain" in condition or "storm" in condition or "drizzle" in condition:
        return "Cozy indoor reading weather 📚"
    if "snow" in condition:
        return "Hot chocolate & blanket vibes ☕️"
    if temp_c > 30:
        return "Bright, breezy, and sunglasses on 😎"
    if temp_c < 10:
        return "Crisp air & warm jacket day 🧣"
    if "cloudy" in condition or "overcast" in condition:
        return "Perfectly soft and cloudy skies ☁️"
    if "clear" in condition:
        return "Blue skies and good times ahead ☀️"
    return "A great day to be you! ✨"

def get_clothing_recommendation(weather_code, temp, unit):
    """Suggests clothes based on weather."""
    temp_c = temp if unit == 'metric' else (temp - 32) * 5/9
    condition = WMO_CODES.get(weather_code, ("Unknown", ""))[0].lower()

    if "rain" in condition or "drizzle" in condition:
        return "Waterproof jacket and an umbrella are a must! ☂️"
    if "snow" in condition:
        return "Heavy coat, gloves, and a warm hat. Stay bundled!"
    if "storm" in condition:
        return "Stay indoors! But if you must go out, waterproof gear is essential."
    if temp_c > 28:
        return "Light cotton or linen clothes. Stay hydrated!"
    if 20 <= temp_c <= 28:
        return "A t-shirt and jeans/shorts will be comfortable."
    if 10 <= temp_c < 20:
        return "A light jacket or sweater is a good idea. 🧥"
    if temp_c < 10:
        return "Wear a warm coat, layers are your friend!"
    return "Check the conditions and dress accordingly."

def get_activity_recommendation(weather_code, aqi_level):
    """Suggests activities based on weather and AQI."""
    condition = WMO_CODES.get(weather_code, ("Unknown", ""))[0].lower()
    aqi_level = aqi_level.lower()

    if aqi_level in ["unhealthy", "very unhealthy", "hazardous"]:
        return f"High pollution! ({aqi_level.title()}) 😷 Best to stay indoors. Good day for study or indoor exercise."
    if "rain" in condition or "storm" in condition or "drizzle" in condition:
        return "Looks wet out there! Perfect time for a movie marathon, baking, or visiting an indoor cafe. ☕"
    if "snow" in condition:
        return "It's a winter wonderland! ❄️ Great for building a snowman or cozying up by the fire."
    if "clear" in condition or "mainly clear" in condition:
        return "Clear skies! ☀️ Fantastic day for jogging, a picnic in the park, or photography."
    if "cloudy" in condition or "overcast" in condition or "fog" in condition:
        return "Pleasantly overcast. Great for a long walk, gardening, or outdoor sports without the harsh sun."
    return "A versatile day! Most outdoor or indoor activities are on the table."


# --- Main Application UI ---

st.set_page_config(page_title="Weather Assistant", layout="wide")

# --- Sidebar for Inputs ---
with st.sidebar:
    # *** THIS IS THE CORRECTED LINE WITH A WORKING IMAGE URL ***
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=100) 
    st.title("PyWeather")
    
    # Check for saved cities in session state
    if 'saved_cities' not in st.session_state:
        st.session_state.saved_cities = ["London", "New York", "Tokyo"] # Default favorites

    city = st.text_input("Enter City Name", "Ghaziabad")
    
    if st.button("Search City", use_container_width=True):
        st.session_state.city = city
    
    st.write("---")
    st.write("Favorite Cities:")
    
    # Display favorite cities as buttons
    for saved_city in st.session_state.saved_cities:
        if st.button(saved_city, use_container_width=True, key=saved_city):
            st.session_state.city = saved_city # Set current city to this favorite
            
    if 'city' not in st.session_state:
        st.session_state.city = "Ghaziabad" # Default city

    st.write("---")
    unit_options = {"Celsius (°C)": "metric", "Fahrenheit (°F)": "imperial"}
    selected_unit_label = st.radio("Select Temperature Unit", unit_options.keys())
    selected_unit = unit_options[selected_unit_label]
    
    unit_symbol = "°C" if selected_unit == "metric" else "°F"


# --- Main Content Area ---
st.header(f"Weather in {st.session_state.city}")

# Fetch data only if the city name is not empty
if st.session_state.city:
    weather_data, air_data, location = get_weather_data(st.session_state.city, selected_unit)

    if weather_data and air_data and location:
        # Store data in session state to persist across tab clicks
        st.session_state.weather_data = weather_data
        st.session_state.air_data = air_data
        st.session_state.unit_symbol = unit_symbol
        st.session_state.selected_unit = selected_unit
        st.session_state.location = location

        # Add to favorites if it's a new city
        if st.session_state.city not in st.session_state.saved_cities:
            if len(st.session_state.saved_cities) >= 5: # Limit to 5 favorites
                st.session_state.saved_cities.pop(0)
            st.session_state.saved_cities.append(st.session_state.city)
    else:
        # Don't clear old data if new fetch fails, just show the error
        pass
else:
    st.info("Please enter a city name in the sidebar.")


if "weather_data" not in st.session_state:
    st.info("Welcome! ☀️ Enter a city and click 'Search City' to start.")
else:
    # Retrieve data from session state
    weather_data = st.session_state.weather_data
    air_data = st.session_state.air_data
    unit_symbol = st.session_state.unit_symbol
    selected_unit = st.session_state.selected_unit
    location = st.session_state.location
    
    current = weather_data['current']
    daily_df, hourly_df = process_forecast_data(weather_data)

    # --- Create Tabs ---
    tab_today, tab_forecast, tab_recommendations = st.tabs(
        ["Today", "10-Day Forecast", "Recommendations"]
    )

    # --- Tab 1: Today's Weather ---
    with tab_today:
        st.subheader(f"Current Conditions in {location['name']}, {location.get('country_code', '')}")
        
        # Main info: Temp, Icon, Description
        col1, col2 = st.columns([1, 2])
        with col1:
            temp_str = f"{current['temperature_2m']:.0f}{unit_symbol}"
            st.markdown(f"<h1 style='text-align: left; font-size: 6em; margin-top: 0; padding-top: 0;'>{temp_str}</h1>", unsafe_allow_html=True)
            
            weather_desc, weather_icon = WMO_CODES.get(current['weather_code'], ("Unknown", "❓"))
            st.markdown(f"**{weather_desc} {weather_icon}**")

        with col2:
            st.markdown(f"Min/Max: **{daily_df.iloc[0]['temperature_2m_min']:.0f}{unit_symbol}** / **{daily_df.iloc[0]['temperature_2m_max']:.0f}{unit_symbol}**")
            st.markdown(f"Precipitation: **{current['precipitation']:.1f} mm**")
            st.markdown(f"Chance of Rain Today: **{daily_df.iloc[0]['precipitation_probability_max']}%**")
            st.write("---")
            
            # Air Quality
            aqi = air_data['current']['us_aqi']
            aqi_level, aqi_color = get_aqi_level(aqi)
            st.markdown(f"Air Quality (US AQI): <span style='color:{aqi_color}; font-weight:bold;'>{aqi_level} ({aqi})</span>", unsafe_allow_html=True)
            st.caption("AQI Index: 0-50=Good, 51-100=Moderate, 101-150=Unhealthy (SG), 151+=Unhealthy")

        st.write("---")
        
        # Additional Details
        st.subheader("Additional Details")
        cols = st.columns(3)
        cols[0].metric("Humidity", f"{current['relative_humidity_2m']}%")
        wind_unit_label = "km/h" if selected_unit == "metric" else "mph"
        cols[1].metric("Wind Speed", f"{current['wind_speed_10m']} {wind_unit_label}")
        cols[2].metric("Latitude", f"{location['latitude']:.2f}°")
        cols[2].metric("Longitude", f"{location['longitude']:.2f}°")

    # --- Tab 2: 10-Day Forecast ---
    with tab_forecast:
        st.header("10-Day Forecast")
        
        # Display daily summaries
        st.subheader("Daily Summary")
        
        for _, row in daily_df.iterrows():
            cols = st.columns([1, 1, 2, 2])
            cols[0].markdown(f"**{row['day']}**")
            
            day_desc, day_icon = WMO_CODES.get(row['weather_code'], ("Unknown", "❓"))
            cols[1].markdown(f"**{day_icon}**")
            cols[2].markdown(f"{day_desc}")
            cols[3].markdown(f"**{row['temperature_2m_max']:.0f}{unit_symbol}** / {row['temperature_2m_min']:.0f}{unit_symbol}")
        
        st.write("---")
        
        # Display interactive hourly chart
        st.subheader("Hourly Trends (Next 10 Days)")
        
        # Create a dropdown to select what to plot
        plot_choice = st.selectbox("Select graph:", ["Temperature", "Rain Chance (%)", "Humidity"])
        
        fig = px.line(
            hourly_df, 
            y=plot_choice, 
            title=f"{plot_choice} Trend",
            markers=False
        )
        fig.update_layout(xaxis_title="Date & Time", yaxis_title=f"{plot_choice}")
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Recommendations ---
    with tab_recommendations:
        st.header("Smart Insights & Recommendations")
        
        condition_code = current['weather_code']
        temp = current['temperature_2m']
        aqi_level, _ = get_aqi_level(air_data['current']['us_aqi'])

        # Get recommendations
        mood = get_weather_mood(condition_code, temp, selected_unit)
        clothing_rec = get_clothing_recommendation(condition_code, temp, selected_unit)
        activity_rec = get_activity_recommendation(condition_code, aqi_level)
        
        # Display in cards
        st.subheader("Today's Weather Mood")
        st.info(f"**{mood}**")
        
        st.subheader("What to Wear")
        st.success(f"**{clothing_rec}**")
        
        st.subheader("Activity Suggestion")
        st.warning(f"**{activity_rec}**")
