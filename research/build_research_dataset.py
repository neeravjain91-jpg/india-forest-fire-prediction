from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRE = ROOT / "research" / "research_fire_samples.csv"
DEFAULT_WEATHER = ROOT / "research" / "weather" / "research_weather_cache.csv"
DEFAULT_OUTPUT = ROOT / "research" / "processed" / "research_training_dataset.csv"
BASE = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm",
]
DERIVED = ["rain_24h", "rain_72h", "rain_168h", "avg_temp_24h", "avg_humidity_24h", "max_wind_24h"]


def add_rolling_features(weather):
    weather = weather.sort_values(["grid_lat", "grid_lon", "acq_date", "hour"]).copy()
    weather["timestamp"] = weather["acq_date"] + pd.to_timedelta(weather["hour"], unit="h")
    weather = weather.sort_values(["grid_lat", "grid_lon", "timestamp"])
    group = weather.groupby(["grid_lat", "grid_lon"], sort=False)
    for hours, name in [(24, "rain_24h"), (72, "rain_72h"), (168, "rain_168h")]:
        weather[name] = group["precipitation"].transform(lambda s: s.rolling(hours, min_periods=1).sum())
    weather["avg_temp_24h"] = group["temperature_2m"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    weather["avg_humidity_24h"] = group["relative_humidity_2m"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    weather["max_wind_24h"] = group["wind_speed_10m"].transform(lambda s: s.rolling(24, min_periods=1).max())
    return weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire", default=str(DEFAULT_FIRE))
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    fire = pd.read_csv(args.fire)
    weather = pd.read_csv(args.weather)
    fire["acq_date"] = pd.to_datetime(fire["acq_date"], errors="coerce")
    weather["acq_date"] = pd.to_datetime(weather["acq_date"], errors="coerce")
    for col in BASE:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.dropna(subset=["grid_lat", "grid_lon", "acq_date", "hour"])
    weather = add_rolling_features(weather)

    fire["timestamp"] = fire["acq_date"] + pd.to_timedelta(fire["hour"], unit="h")
    weather["timestamp"] = weather["acq_date"] + pd.to_timedelta(weather["hour"], unit="h")
    weather = weather.drop_duplicates(["grid_lat", "grid_lon", "timestamp"], keep="last")

    # Fire samples are retained as provided. Non-fire generation is deliberately
    # a separate step so its sampling assumptions can be audited and reproduced.
    out = fire.merge(
        weather[["grid_lat", "grid_lon", "timestamp"] + BASE + DERIVED],
        on=["grid_lat", "grid_lon", "timestamp"], how="inner",
    )
    out["month"] = out["acq_date"].dt.month
    out["day_of_year"] = out["acq_date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out = out.drop(columns=["timestamp"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("=" * 60)
    print("PROJECT 2 FIRE WEATHER DATASET CREATED")
    print("=" * 60)
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns)}")
    print(f"Fire rows: {(out.fire == 1).sum():,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
