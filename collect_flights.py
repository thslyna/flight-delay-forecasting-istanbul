import os
import requests
import pandas as pd
from datetime import datetime

# Export your key first:
# export AVIATIONSTACK_KEY="your_real_key_here"

KEY = os.getenv("AVIATIONSTACK_KEY")
if not KEY:
    raise SystemExit("❌ ERROR: You must export AVIATIONSTACK_KEY before running this script.")

URL = "http://api.aviationstack.com/v1/flights"
params = {
    "access_key": KEY,
    "dep_iata": "IST",   # departing from Istanbul (IST)
    "limit": 100
}

resp = requests.get(URL, params=params, timeout=30)
print("HTTP Status:", resp.status_code)

if resp.status_code != 200:
    print("❌ API error:", resp.status_code, resp.text)
    raise SystemExit()

data = resp.json().get("data", [])
if not data:
    print("⚠️ No flight data returned.")
else:
    df = pd.json_normalize(data)

    # Select / rename a stable set of columns to persist (keeps CSV consistent)
    cols = {
        "flight_date": "flight_date",
        "flight_status": "flight_status",
        "departure.iata": "dep_iata",
        "departure.icao": "dep_icao",
        "departure.airport": "dep_airport",
        "departure.scheduled": "dep_scheduled",
        "departure.estimated": "dep_estimated",
        "departure.actual": "dep_actual",
        "departure.delay": "dep_delay",
        "arrival.iata": "arr_iata",
        "arrival.icao": "arr_icao",
        "arrival.airport": "arr_airport",
        "arrival.scheduled": "arr_scheduled",
        "arrival.estimated": "arr_estimated",
        "arrival.actual": "arr_actual",
        "arrival.delay": "arr_delay",
        "airline.name": "airline_name",
        "flight.number": "flight_number",
        "flight.iata": "flight_iata",
        "flight.icao": "flight_icao"
    }

    # Keep only available columns from the response
    keep = [c for c in cols.keys() if c in df.columns]
    out_df = df[keep].rename(columns={k: cols[k] for k in keep})

    # add collection timestamp
    out_df["collected_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure column order is stable
    ordered_cols = list(out_df.columns)

    out_path = os.path.expanduser("~/thesis/data/raw/flights_all.csv")

    # Append to CSV with header only if file doesn't exist
    out_df.to_csv(out_path, index=False, mode="a", header=not os.path.exists(out_path))

    print(f"✅ Flight data appended to: {out_path}")
    print(f"🛫 {len(out_df)} flights saved.")