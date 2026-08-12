from pathlib import Path
import argparse
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "research" / "research_fire_samples.csv"
DEFAULT_CACHE = ROOT / "research" / "weather" / "research_weather_cache.csv"
API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm",
]


def request_weather(points, start_date, end_date):
    params = {
        "latitude": ",".join(f"{x[0]:.1f}" for x in points),
        "longitude": ",".join(f"{x[1]:.1f}" for x in points),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY),
        "models": "ecmwf_ifs",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    for attempt in range(8):
        try:
            response = requests.get(API, params=params, timeout=180)
            if response.status_code == 429:
                wait = min(600, 120 * (attempt + 1))
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                print(f"429 rate limit; waiting {wait}s (retry {attempt + 1}/8)")
                time.sleep(wait)
                continue
            if response.status_code in (502, 503, 504):
                wait = min(300, 60 * (attempt + 1))
                print(f"HTTP {response.status_code}; waiting {wait}s (retry {attempt + 1}/8)")
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else [payload]
        except requests.RequestException as exc:
            if attempt == 7:
                print(f"Request failed: {exc}")
                return []
            wait = min(300, 60 * (attempt + 1))
            print(f"Request error; waiting {wait}s (retry {attempt + 1}/8)")
            time.sleep(wait)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--coord-batch", type=int, default=50)
    parser.add_argument("--pause", type=int, default=20)
    parser.add_argument("--days", type=int, default=7, help="Target window size; 7 days also fetches 7 prior days for rolling features.")
    args = parser.parse_args()

    sample = pd.read_csv(args.input)
    sample["acq_date"] = pd.to_datetime(sample["acq_date"], errors="coerce")
    sample["grid_lat"] = pd.to_numeric(sample["grid_lat"], errors="coerce").round(1)
    sample["grid_lon"] = pd.to_numeric(sample["grid_lon"], errors="coerce").round(1)
    sample["hour"] = pd.to_numeric(sample["hour"], errors="coerce").astype(int)
    sample = sample.dropna(subset=["acq_date", "grid_lat", "grid_lon", "hour"])

    keys = sample[["grid_lat", "grid_lon", "acq_date", "hour"]].drop_duplicates()
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        cache["acq_date"] = pd.to_datetime(cache["acq_date"], errors="coerce")
        cache["grid_lat"] = pd.to_numeric(cache["grid_lat"], errors="coerce").round(1)
        cache["grid_lon"] = pd.to_numeric(cache["grid_lon"], errors="coerce").round(1)
        cache["hour"] = pd.to_numeric(cache["hour"], errors="coerce").astype("Int64")
    else:
        cache = pd.DataFrame(columns=["grid_lat", "grid_lon", "acq_date", "hour"] + HOURLY)

    cached_keys = set(zip(cache.grid_lat, cache.grid_lon, cache.acq_date.dt.strftime("%Y-%m-%d"), cache.hour.astype("Int64")))
    missing = [tuple(x) for x in keys.itertuples(index=False, name=None)
               if (float(x[0]), float(x[1]), pd.Timestamp(x[2]).strftime("%Y-%m-%d"), int(x[3])) not in cached_keys]

    print(f"Research observations: {len(sample):,}")
    print(f"Unique location/time cells: {len(keys):,}")
    print(f"Cached cells: {len(keys) - len(missing):,}")
    print(f"Missing cells: {len(missing):,}")
    if not missing:
        print("Weather cache already covers all requested cells.")
        return

    # Group target observations by ISO week. Each request fetches a 14-day period:
    # seven days before the target week plus the seven-day target week.
    groups = {}
    for lat, lon, date, hour in missing:
        date = pd.Timestamp(date)
        week = date - pd.Timedelta(days=date.dayofweek)
        groups.setdefault(week.strftime("%Y-%m-%d"), []).append((float(lat), float(lon), date, int(hour)))

    blocks = list(groups.items())
    print(f"Research weather blocks: {len(blocks):,}")
    request_no = 0

    for block_no, (week_text, targets) in enumerate(blocks, 1):
        week = pd.Timestamp(week_text)
        start_date = week - pd.Timedelta(days=7)
        end_date = week + pd.Timedelta(days=6)
        coords = sorted(set((x[0], x[1]) for x in targets))

        for offset in range(0, len(coords), args.coord_batch):
            batch = coords[offset:offset + args.coord_batch]
            payloads = request_weather(batch, start_date, end_date)
            request_no += 1
            rows = []
            for coord, payload in zip(batch, payloads):
                hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
                times = hourly.get("time", [])
                if not times:
                    continue
                for lat, lon, date, hour in targets:
                    if (lat, lon) != coord:
                        continue
                    target = pd.Timestamp(date).replace(hour=hour)
                    target_text = target.strftime("%Y-%m-%dT%H:%M")
                    try:
                        idx = times.index(target_text)
                    except ValueError:
                        continue
                    row = {"grid_lat": lat, "grid_lon": lon, "acq_date": date, "hour": hour}
                    for col in HOURLY:
                        values = hourly.get(col, [])
                        row[col] = values[idx] if idx < len(values) else None
                    rows.append(row)
            if rows:
                cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
                cache = cache.drop_duplicates(["grid_lat", "grid_lon", "acq_date", "hour"], keep="last")
                cache.to_csv(cache_path, index=False)
            print(f"Block {block_no}/{len(blocks)} | request {request_no} | cached rows={len(cache):,}")
            time.sleep(args.pause)

    print("Weather acquisition complete or exhausted available requests.")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()
