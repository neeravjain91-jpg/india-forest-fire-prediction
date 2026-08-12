from pathlib import Path
import argparse
import pandas as pd

RAW_FILE = Path("data/raw/FIRMS_VIIRS_SNPP_India_2018_2025.csv")
OUTPUT_FILE = Path("india_firms_fire_samples.csv")

USECOLS = ["latitude", "longitude", "bright_ti4", "bright_ti5", "scan", "track",
           "acq_date", "acq_time", "satellite", "instrument", "confidence",
           "version", "frp", "daynight", "type"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-fire-samples", type=int, default=20_000)
    args = parser.parse_args()

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw FIRMS file not found: {RAW_FILE}")

    parts = []
    total = 0
    for chunk in pd.read_csv(RAW_FILE, usecols=lambda c: c in USECOLS,
                             chunksize=args.chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
        chunk["acq_date"] = pd.to_datetime(chunk["acq_date"], errors="coerce")
        chunk["acq_time"] = pd.to_numeric(chunk["acq_time"], errors="coerce")
        chunk = chunk.dropna(subset=["latitude", "longitude", "acq_date", "acq_time"])
        chunk = chunk[(chunk.latitude >= 6) & (chunk.latitude <= 37) &
                      (chunk.longitude >= 68) & (chunk.longitude <= 98)]
        chunk["hour"] = (chunk["acq_time"] // 100).astype(int)
        chunk["fire"] = 1
        parts.append(chunk)
        total += len(chunk)
        print(f"Processed {total:,} valid FIRMS rows...")

    if not parts:
        raise RuntimeError("No valid FIRMS observations found.")

    fire = pd.concat(parts, ignore_index=True)
    fire = fire.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    if len(fire) > args.max_fire_samples:
        fire = fire.sample(args.max_fire_samples, random_state=42)
    fire = fire.sort_values(["acq_date", "hour", "latitude", "longitude"]).reset_index(drop=True)
    fire.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("FIRMS FIRE SAMPLE CREATED")
    print("=" * 60)
    print(f"Rows: {len(fire):,}")
    print(f"Output: {OUTPUT_FILE}")
    print("Next stage: generate matched non-fire samples and weather features.")


if __name__ == "__main__":
    main()
