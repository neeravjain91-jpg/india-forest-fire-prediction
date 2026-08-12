from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIRE_FILE = ROOT / "india_firms_fire_samples.csv"
OUTPUT_FILE = ROOT / "india_forest_fire_dataset.csv"
CACHE_FILE = ROOT / "weather_cache.csv"

BASE_WEATHER = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "wind_speed_10m", "wind_direction_10m",
    "surface_pressure", "cloud_cover", "soil_moisture_0_to_7cm",
]
DERIVED = [
    "rain_24h", "rain_72h", "rain_168h", "avg_temp_24h",
    "avg_humidity_24h", "max_wind_24h",
]
FEATURE_COLUMNS = BASE_WEATHER + DERIVED
API = "https://archive-api.open-meteo.com/v1/archive"
COORD_BATCH = 50
WEEK_DAYS = 7


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
    fire_keys = set(zip(fire.grid_lat, fire.grid_lon, fire.acq_date.dt.date, fire.hour))
    dates = fire.acq_date.dt.normalize().drop_duplicates().to_numpy()
    n = len(fire)
    lat_min, lat_max = fire.grid_lat.min(), fire.grid_lat.max()
    lon_min, lon_max = fire.grid_lon.min(), fire.grid_lon.max()
    rows, attempts = [], 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        lat = round(float(rng.uniform(lat_min, lat_max)), 1)
        lon = round(float(rng.uniform(lon_min, lon_max)), 1)
        date = pd.Timestamp(rng.choice(dates))
        hour = int(rng.integers(0, 24))
        key = (lat, lon, date.date(), hour)
        if key in fire_keys:
            continue
        rows.append({"latitude": lat, "longitude": lon, "acq_date": date,
                     "hour": hour, "grid_lat": lat, "grid_lon": lon, "fire": 0})
    if len(rows) < n:
        raise RuntimeError(f"Could only generate {len(rows):,} non-fire samples out of {n:,}.")
    return pd.DataFrame(rows)


def request_period(points, start_date, end_date):
    params = {
        "latitude": ",".join(f"{p[0]:.1f}" for p in points),
        "longitude": ",".join(f"{p[1]:.1f}" for p in points),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(BASE_WEATHER),
        "models": "ecmwf_ifs",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=180)
            if r.status_code in (429, 502, 503, 504):
                wait = min(90, 15 * (attempt + 1))
                print(f"Request {r.status_code}; retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            return payload if isinstance(payload, list) else [payload]
        except requests.RequestException as exc:
            if attempt == 5:
                print(f"Request failed permanently: {exc}")
                return []
            wait = min(90, 15 * (attempt + 1))
            print(f"Request error; retrying in {wait}s...")
            time.sleep(wait)
    return []


def feature_at_target(hourly, target_ts):
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        return None

    # target_ts may already be timezone-aware. Normalize it safely to UTC.
    target_ts = pd.Timestamp(target_ts)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")
    else:
        target_ts = target_ts.tz_convert("UTC")

    matches = np.where(times == target_ts)[0]
    if len(matches) == 0:
        return None
    i = int(matches[0])
    values = {}
    for col in BASE_WEATHER:
        arr = hourly.get(col, [])
        values[col] = arr[i] if i < len(arr) else np.nan

    def arr_window(col, hours):
        arr = pd.to_numeric(pd.Series(hourly.get(col, [])), errors="coerce")
        start = max(0, i - hours + 1)
        return arr.iloc[start:i + 1]

    rain = arr_window("precipitation", 24)
    rain72 = arr_window("precipitation", 72)
    rain168 = arr_window("precipitation", 168)
    temp = arr_window("temperature_2m", 24)
    hum = arr_window("relative_humidity_2m", 24)
    wind = arr_window("wind_speed_10m", 24)

    values["rain_24h"] = rain.sum(min_count=1)
    values["rain_72h"] = rain72.sum(min_count=1)
    values["rain_168h"] = rain168.sum(min_count=1)
    values["avg_temp_24h"] = temp.mean()
    values["avg_humidity_24h"] = hum.mean()
    values["max_wind_24h"] = wind.max()
    return values


def load_cache():
    cache = {}
    if not CACHE_FILE.exists():
        return cache
    old = pd.read_csv(CACHE_FILE)
    if not set(DERIVED).issubset(old.columns):
        print("Existing cache is from the old format; it will be rebuilt.")
        return cache
    for _, r in old.iterrows():
        key = (round(float(r.grid_lat), 1), round(float(r.grid_lon), 1),
               str(r.acq_date), int(r.hour))
        cache[key] = r.to_dict()
    return cache


def save_cache(cache):
    pd.DataFrame(cache.values()).to_csv(CACHE_FILE, index=False)


def build_weather_features(keys, cache):
    missing = [k for k in keys if k not in cache]
    groups = {}
    for lat, lon, date, hour in missing:
        d = pd.Timestamp(date)
        week_start = d - pd.Timedelta(days=d.dayofweek)
        groups.setdefault(week_start.strftime("%Y-%m-%d"), []).append((lat, lon, date, hour))

    group_items = list(groups.items())
    print(f"Weather blocks required: {len(group_items):,}")

    for block_no, (week_key, target_keys) in enumerate(group_items, 1):
        week_start = pd.Timestamp(week_key)
        week_end = week_start + pd.Timedelta(days=6)
        api_start = week_start - pd.Timedelta(days=7)
        api_end = week_end
        coords = list(dict.fromkeys((float(k[0]), float(k[1])) for k in target_keys))

        for start in range(0, len(coords), COORD_BATCH):
            batch = coords[start:start + COORD_BATCH]
            payloads = request_period(batch, api_start, api_end)
            if not payloads:
                continue

            for (lat, lon), data in zip(batch, payloads):
                hourly = data.get("hourly", {}) if isinstance(data, dict) else {}
                for k_lat, k_lon, date, hour in target_keys:
                    if float(k_lat) != lat or float(k_lon) != lon:
                        continue
                    target = pd.Timestamp(date) + pd.Timedelta(hours=int(hour))
                    vals = feature_at_target(hourly, target)
                    if vals is not None:
                        key = (float(k_lat), float(k_lon), date, int(hour))
                        cache[key] = {
                            "grid_lat": float(k_lat), "grid_lon": float(k_lon),
                            "acq_date": date, "hour": int(hour), **vals
                        }

            time.sleep(2)

        save_cache(cache)
        print(f"Weather block {block_no}/{len(group_items)} complete; cache rows={len(cache):,}")


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

    cache = load_cache()
    if cache:
        print(f"Loaded complete weather cache: {len(cache):,} rows")

    keys = [(float(r.grid_lat), float(r.grid_lon),
             pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour))
            for _, r in combined.iterrows()]
    unique_keys = list(dict.fromkeys(keys))
    cached = sum(k in cache for k in unique_keys)
    print(f"Observations: {len(combined):,}")
    print(f"Unique weather location/time cells: {len(unique_keys):,}")
    print(f"Already cached: {cached:,}")
    print(f"Remaining weather cells: {len(unique_keys) - cached:,}")

    build_weather_features(unique_keys, cache)

    rows = []
    for _, r in combined.iterrows():
        key = (float(r.grid_lat), float(r.grid_lon),
               pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour))
        w = cache.get(key)
        if not w:
            continue
        row = r.to_dict()
        row.update({c: w.get(c) for c in FEATURE_COLUMNS})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No weather-matched samples were produced.")
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=FEATURE_COLUMNS + ["fire"])
    out.to_csv(OUTPUT_FILE, index=False)
    save_cache(cache)

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
