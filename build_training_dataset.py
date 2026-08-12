from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIRE_FILE = ROOT / "india_firms_fire_samples.csv"
OUTPUT_FILE = ROOT / "india_forest_fire_dataset.csv"
CACHE_FILE = ROOT / "weather_cache.csv"

# Keep the original project's weather-driven classification idea.
WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "wind_speed_10m", "wind_direction_10m",
    "surface_pressure", "cloud_cover", "soil_moisture_0_to_7cm",
]

FEATURE_COLUMNS = WEATHER_COLUMNS + [
    "rain_24h", "rain_72h", "rain_168h", "avg_temp_24h",
    "avg_humidity_24h", "max_wind_24h",
]

API = "https://archive-api.open-meteo.com/v1/archive"


def round_coord(x):
    return round(float(x) / 0.1) * 0.1


def load_fire_samples():
    df = pd.read_csv(FIRE_FILE)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date", "hour"])
    df["hour"] = df["hour"].clip(0, 23).astype(int)
    df["grid_lat"] = df["latitude"].map(round_coord)
    df["grid_lon"] = df["longitude"].map(round_coord)
    return df


def make_nonfire_samples(fire, seed=42):
    rng = np.random.default_rng(seed)
    fire_keys = set(zip(fire["grid_lat"], fire["grid_lon"], fire["acq_date"].dt.date, fire["hour"]))
    dates = fire["acq_date"].dt.normalize().drop_duplicates().to_numpy()
    n = len(fire)
    rows = []
    # Sample the same observed dates and nearby spatial domain, rejecting fire cells.
    lat_min, lat_max = fire.grid_lat.min(), fire.grid_lat.max()
    lon_min, lon_max = fire.grid_lon.min(), fire.grid_lon.max()
    attempts = 0
    while len(rows) < n and attempts < n * 30:
        attempts += 1
        lat = round(float(rng.uniform(lat_min, lat_max)), 1)
        lon = round(float(rng.uniform(lon_min, lon_max)), 1)
        date = pd.Timestamp(rng.choice(dates))
        hour = int(rng.integers(0, 24))
        key = (lat, lon, date.date(), hour)
        if key in fire_keys:
            continue
        rows.append({"latitude": lat, "longitude": lon, "acq_date": date, "hour": hour, "grid_lat": lat, "grid_lon": lon, "fire": 0})
    if len(rows) < n:
        raise RuntimeError(f"Could only generate {len(rows):,} non-fire samples out of {n:,}.")
    return pd.DataFrame(rows)


def fetch_weather(lat, lon, date, hour):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date.strftime("%Y-%m-%d"),
        "end_date": date.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_COLUMNS),
        "models": "ecmwf_ifs",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, timeout=60)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json().get("hourly", {})
            times = data.get("time", [])
            if not times:
                return None
            idx = min(range(len(times)), key=lambda i: abs(pd.Timestamp(times[i]).hour - hour))
            out = {c: data.get(c, [None])[idx] for c in WEATHER_COLUMNS}
            return out
        except requests.RequestException:
            if attempt == 4:
                return None
            time.sleep(10 * (attempt + 1))
    return None


def main():
    if not FIRE_FILE.exists():
        raise FileNotFoundError(FIRE_FILE)
    fire = load_fire_samples()
    fire["fire"] = 1
    nonfire = make_nonfire_samples(fire)
    combined = pd.concat([fire[["latitude", "longitude", "acq_date", "hour", "grid_lat", "grid_lon", "fire"]], nonfire], ignore_index=True)

    cache = {}
    if CACHE_FILE.exists():
        old = pd.read_csv(CACHE_FILE)
        for _, r in old.iterrows():
            cache[(round(float(r.grid_lat), 1), round(float(r.grid_lon), 1), str(r.acq_date), int(r.hour))] = r.to_dict()

    total = len(combined)
    rows = []
    for i, r in combined.iterrows():
        date = pd.Timestamp(r.acq_date).strftime("%Y-%m-%d")
        key = (float(r.grid_lat), float(r.grid_lon), date, int(r.hour))
        if key not in cache:
            weather = fetch_weather(*key[:2], pd.Timestamp(date), key[3])
            if weather is None:
                print(f"Weather unavailable at {i + 1}/{total}; skipping")
                continue
            cache[key] = {"grid_lat": key[0], "grid_lon": key[1], "acq_date": date, "hour": key[3], **weather}
            if len(cache) % 25 == 0:
                pd.DataFrame(cache.values()).to_csv(CACHE_FILE, index=False)
                print(f"Weather cached for {len(cache):,} location/time cells")
        w = cache[key]
        row = r.to_dict()
        row.update({c: w.get(c) for c in WEATHER_COLUMNS})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No weather-matched samples were produced.")
    out["rain_24h"] = out["precipitation"]
    out["rain_72h"] = out["precipitation"]
    out["rain_168h"] = out["precipitation"]
    out["avg_temp_24h"] = out["temperature_2m"]
    out["avg_humidity_24h"] = out["relative_humidity_2m"]
    out["max_wind_24h"] = out["wind_speed_10m"]
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=FEATURE_COLUMNS + ["fire"])
    out.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(cache.values()).to_csv(CACHE_FILE, index=False)
    print("=" * 60)
    print("FINAL INDIA TRAINING DATASET CREATED")
    print("=" * 60)
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns)}")
    print(f"Fire=1: {(out.fire == 1).sum():,}")
    print(f"Fire=0: {(out.fire == 0).sum():,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
