from pathlib import Path

import numpy as np
import pandas as pd

FIRE_FILE = Path("india_firms_weather.csv")
CANDIDATE_FILE = Path("india_weather_candidates.csv")
OUTPUT_FILE = Path("india_forest_fire_dataset.csv")

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm", "rain_24h", "rain_72h", "rain_168h",
    "avg_temp_24h", "avg_humidity_24h", "max_wind_24h",
]
REQUIRED = ["latitude", "longitude", "acq_date", "acq_time"] + FEATURES


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def normalize_time(df):
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce").fillna(-1).astype(int)
    df["hour"] = (df["acq_time"] // 100).astype(int)
    return df


def validate(df, name):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def build_dataset():
    if not FIRE_FILE.exists():
        raise FileNotFoundError(f"Missing {FIRE_FILE}")
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CANDIDATE_FILE}. It must contain weather rows for candidate "
            "locations/times, including locations where FIRMS detected no fire."
        )

    fire = pd.read_csv(FIRE_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    fire.columns = fire.columns.str.strip()
    candidates.columns = candidates.columns.str.strip()
    validate(fire, "Fire dataset")
    validate(candidates, "Candidate weather dataset")

    fire = normalize_time(fire)
    candidates = normalize_time(candidates)
    fire["fire"] = 1

    # Reject candidate locations within 5 km of a detected fire in the same hour.
    keep = np.ones(len(candidates), dtype=bool)
    for key, group in fire.groupby(["acq_date", "hour"], sort=False):
        idx = candidates.index[
            (candidates["acq_date"] == key[0]) & (candidates["hour"] == key[1])
        ]
        if len(idx) == 0:
            continue
        c = candidates.loc[idx, ["latitude", "longitude"]].to_numpy()
        nearby = np.zeros(len(c), dtype=bool)
        for lat, lon in group[["latitude", "longitude"]].to_numpy():
            nearby |= haversine_km(c[:, 0], c[:, 1], lat, lon) < 5.0
        keep[candidates.index.get_indexer(idx)] = ~nearby

    negatives = candidates.loc[keep].copy()
    negatives["fire"] = 0
    n = min(len(fire), len(negatives))
    if n == 0:
        raise ValueError("No valid non-fire samples were available.")

    fire = fire.sample(n=n, random_state=42)
    negatives = negatives.sample(n=n, random_state=42)
    final = pd.concat([fire, negatives], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)
    final.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("INDIA DATASET CREATED")
    print("=" * 60)
    print(f"Rows: {len(final):,}")
    print(f"Fire=1: {(final['fire'] == 1).sum():,}")
    print(f"Fire=0: {(final['fire'] == 0).sum():,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()
