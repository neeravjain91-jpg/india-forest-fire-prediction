from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRE = ROOT / "research" / "research_fire_samples.csv"
DEFAULT_OUTPUT = ROOT / "research" / "research_nonfire_samples.csv"


def main():
    parser = argparse.ArgumentParser(description="Create matched non-fire samples for Project 2.")
    parser.add_argument("--fire", default=str(DEFAULT_FIRE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ratio", type=float, default=1.0, help="Non-fire/fire ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fire = pd.read_csv(args.fire)
    fire["acq_date"] = pd.to_datetime(fire["acq_date"], errors="coerce")
    fire["grid_lat"] = pd.to_numeric(fire["grid_lat"], errors="coerce").round(1)
    fire["grid_lon"] = pd.to_numeric(fire["grid_lon"], errors="coerce").round(1)
    fire["hour"] = pd.to_numeric(fire["hour"], errors="coerce").astype(int)
    fire = fire.dropna(subset=["grid_lat", "grid_lon", "acq_date", "hour"])

    n = int(round(len(fire) * args.ratio))
    rng = np.random.default_rng(args.seed)
    fire_keys = set(zip(fire.grid_lat, fire.grid_lon, fire.acq_date.dt.strftime("%Y-%m-%d"), fire.hour))

    # Match the observed fire-date and hour distributions while sampling from
    # the geographic bounding box of the fire data. This is a baseline negative
    # sampling strategy; later experiments can replace it with land-cover or
    # distance-matched controls.
    dates = fire.acq_date.dt.normalize().to_numpy()
    hours = fire.hour.to_numpy()
    lat_values = fire.grid_lat.to_numpy()
    lon_values = fire.grid_lon.to_numpy()
    lat_min, lat_max = lat_values.min(), lat_values.max()
    lon_min, lon_max = lon_values.min(), lon_values.max()

    rows = []
    attempts = 0
    max_attempts = max(n * 50, 1000)
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        idx = int(rng.integers(0, len(fire)))
        date = pd.Timestamp(dates[idx])
        hour = int(hours[idx])
        lat = round(float(rng.uniform(lat_min, lat_max)), 1)
        lon = round(float(rng.uniform(lon_min, lon_max)), 1)
        key = (lat, lon, date.strftime("%Y-%m-%d"), hour)
        if key in fire_keys:
            continue
        rows.append({"grid_lat": lat, "grid_lon": lon, "acq_date": date, "hour": hour, "fire": 0})

    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows):,} non-fire samples out of {n:,} requested.")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("=" * 60)
    print("PROJECT 2 NON-FIRE SAMPLE CREATED")
    print("=" * 60)
    print(f"Fire samples: {len(fire):,}")
    print(f"Non-fire samples: {len(out):,}")
    print(f"Ratio: {len(out) / len(fire):.2f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
