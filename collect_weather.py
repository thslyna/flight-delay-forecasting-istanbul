import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# ✅ Make sure you've exported your OpenWeather API key first:
# export OPENWEATHER_KEY="your_api_key_here"

API_KEY = os.getenv("OPENWEATHER_KEY")
if not API_KEY:
    raise SystemExit("❌ ERROR: You must export OPENWEATHER_KEY before running this script.")

# 🌍 Istanbul Airport coordinates (more accurate for aviation context)
LAT, LON = 41.275278, 28.751944

# 🌦️ API URL
URL = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

# 🚀 Make the request
resp = requests.get(URL, timeout=20)
print("HTTP Status:", resp.status_code)

if resp.status_code == 200:
    data = resp.json()

    # 🕒 Convert timestamps from API
    dt_utc = datetime.fromtimestamp(data["dt"], tz=timezone.utc)
    tz_offset = timedelta(seconds=data["timezone"])
    dt_local = dt_utc + tz_offset

    # 📊 Extract and organize data
    weather = {
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather": data["weather"][0]["main"],
        "api_utc_time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "api_local_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 💾 Save data to CSV
    out_path = os.path.expanduser("~/thesis/data/raw/weather_all.csv")
    df = pd.DataFrame([weather])
    df.to_csv(out_path, index=False, mode="a", header=not os.path.exists(out_path))

    print("✅ Weather data appended to:", out_path)
    print(df)
else:
    print("❌ API error:", resp.text)
