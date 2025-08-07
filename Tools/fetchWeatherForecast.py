import requests
from datetime import datetime

API_URL = "https://api.open-meteo.com/v1/forecast"

def get_weather_forecast(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,uv_index_max",
        "timezone": "auto"
    }

    response = requests.get(API_URL, params=params)
    if response.status_code != 200: return {"error": "Failed to fetch data"}

    data = response.json()
    forecast = []
    
    days = data["daily"]["time"]
    for i in range(len(days)):
        forecast.append({
            "date": days[i],
            "temperature_max": data["daily"]["temperature_2m_max"][i],
            "temperature_min": data["daily"]["temperature_2m_min"][i],
            "precipitation_sum_mm": data["daily"]["precipitation_sum"][i],
            "wind_speed_max_kmh": data["daily"]["windspeed_10m_max"][i],
            "uv_index_max": data["daily"]["uv_index_max"][i]
        })

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": data.get("timezone")
        },
        "forecast_days": forecast
    }

if __name__ == "__main__":
    lat, lon = 28.6139, 77.2090 
    forecast_data = get_weather_forecast(lat, lon)
    print(forecast_data)
