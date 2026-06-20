import requests
import pandas as pd
from datetime import datetime, timedelta

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_coordinates(city_name):
    """
    Fetches latitude, longitude, country, and timezone for a given city name.
    """
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "name": result.get("name"),
                "country": result.get("country"),
                "timezone": result.get("timezone", "auto")
            }
        return None
    except Exception as e:
        print(f"Error fetching coordinates for {city_name}: {e}")
        return None

def fetch_historical_data(lat, lon, start_date, end_date):
    """
    Fetches historical temperature (max and min daily) for a location and date range.
    Returns a pandas DataFrame with ['date', 'temp_max', 'temp_min', 'temp_mean'] or None on error.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "auto",
        "models": "era5"
    }
    try:
        response = requests.get(HIST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            daily = data["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "temp_max": daily["temperature_2m_max"],
                "temp_min": daily["temperature_2m_min"]
            })
            # Clean missing data if any
            df = df.dropna().reset_index(drop=True)
            # Calculate mean temperature
            df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
            return df
        return None
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def fetch_forecast_data(lat, lon, forecast_days=7):
    """
    Fetches official upcoming weather forecast from Open-Meteo Forecast API.
    Returns a pandas DataFrame with ['date', 'forecast_max', 'forecast_min', 'forecast_mean'] or None on error.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            daily = data["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "forecast_max": daily["temperature_2m_max"],
                "forecast_min": daily["temperature_2m_min"]
            })
            df = df.dropna().reset_index(drop=True)
            df["forecast_mean"] = (df["forecast_max"] + df["forecast_min"]) / 2
            return df
        return None
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return None
