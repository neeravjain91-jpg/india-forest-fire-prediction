from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIRE_FILE = ROOT / "india_firms_fire_samples.csv"
OUTPUT_FILE = ROOT / "india_forest_fire_dataset.csv"
CACHE_FILE = ROOT / "weather_cache.csv"

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
BATCH_SIZE = 20


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
        rows.append({"latitude": lat, "longitude": lon, "acq_date": date, "hour": hour,
                     "grid_lat": lat, "grid_lon": lon, "fire": 0})
    if len(rows) < n:
        raise RuntimeError(f"Could only generate {len(rows):,} non-fire samples out of {n:,}.")
    return pd.DataFrame(rows)


def request_batch(points, date):
    """Request up to BATCH_SIZE coordinates for one date in a single API call."""
    lats = ",".join(f"{p[0]:.1f}" for p in points)
    lons = ",".join(f"{p[1]:.1f}" for p in points)
    params = {
        "latitude": lats, "longitude": lons,
        "start_date": date.strftime("%Y-%m-%d"),
        "end_date": date.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_COLUMNS),
        "models": "ecmwf_ifs", "timezone": "UTC",
        "temperature_unit": "celsius", "wind_speed_unit": "ms",
        "precipitation_unit": "mm", "cell_selection": "land",
    }
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code in (429, 502, 503, 504):
                wait = min(60, 10 * (attempt + 1))
                print(f"Batch request {r.status_code}; retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                payload = [payload]
            return payload
        except requests.RequestException as exc:
            if attempt == 5:
                print(f"Batch failed permanently: {exc}")
                return []
            wait = min(60, 10 * (attempt + 1))
            print(f"Request error; retrying in {wait}s...")
            time.sleep(wait)
    return []


def build_weather_cache(keys, cache):
    missing = [k for k in keys if k not in cache]
    grouped = {}
    for lat, lon, date, hour in missing:
        grouped.setdefault(date, set()).add((lat, lon))

    completed = 0
    for date, coords in grouped.items():
        coords = list(coords)
        for start in range(0, len(coords), BATCH_SIZE):
            batch = coords[start:start + BATCH_SIZE]
            payloads = request_batch(batch, pd.Timestamp(date))
            if not payloads:
                continue
            for (lat, lon), data in zip(batch, payloads):
                times = data.get("hourly", {}).get("time", [])
                if not times:
                    continue
                hourly = data["hourly"]
                # Cache every hour returned for the requested date.
                for i, t in enumerate(times):
                    h = pd.Timestamp(t).hour
                    key = (round(lat, 1), round(lon, 1), date, h)
                    cache[key] = {
                        "grid_lat": lat, "grid_lon": lon, "acq_date": date, "hour": h,
                        **{c: hourly.get(c, [None] * len(times))[i] for c in WEATHER_COLUMNS}
                    }
            completed += len(batch)
            if completed % 100 == 0 or completed == len(coords):
                pd.DataFrame(cache.values()).to_csv(CACHE_FILE, index=False)
                print(f"Weather batched: {completed:,}/{len(coords):,} coordinates for {date}; cache rows={len(cache):,}")
            time.sleep(2)


def main():
    if not FIRE_FILE.exists():
        raise FileNotFoundError(FIRE_FILE)

    fire = load_fire_samples()
    fire["fire"] = 1
    nonfire = make_nonfire_samples(fire)
    combined = pd.concat([
        fire[["latitude", "longitude", "acq_date", "hour", "grid_lat", "grid_lon", "fire"]],
        nonfire
    ], ignore_index=True)

    cache = {}
    if CACHE_FILE.exists():
        old = pd.read_csv(CACHE_FILE)
        for _, r in old.iterrows():
            cache[(round(float(r.grid_lat), 1), round(float(r.grid_lon), 1),
                   str(r.acq_date), int(r.hour))] = r.to_dict()
        print(f"Loaded existing weather cache: {len(cache):,} rows")

    keys = []
    for _, r in combined.iterrows():
        keys.append((float(r.grid_lat), float(r.grid_lon),
                     pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour)))
    unique_keys = list(dict.fromkeys(keys))
    print(f"Observations: {len(combined):,}")
    print(f"Unique weather location/time cells: {len(unique_keys):,}")
    print(f"Already cached: {sum(k in cache for k in unique_keys):,}")
    print(f"Remaining weather cells: {sum(k not in cache for k in unique_keys):,}")

    build_weather_cache(unique_keys, cache)

    rows = []
    for _, r in combined.iterrows():
        key = (float(r.grid_lat), float(r.grid_lon),
               pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour))
        w = cache.get(key)
        if not w:
            continue
        row = r.to_dict()
        row.update({c: w.get(c) for c in WEATHER_COLUMNS})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No weather-matched samples were produced.")

    # These are deliberately explicit baseline features for the original-model adaptation.
    # A later research version can calculate true rolling windows from hourly weather history.
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
