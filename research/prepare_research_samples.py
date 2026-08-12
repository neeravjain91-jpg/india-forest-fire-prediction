from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "firms_india" / "FIRMS_VIIRS_SNPP_India_2018_2025.csv"
DEFAULT_OUTPUT = ROOT / "research" / "research_fire_samples.csv"


def round_coord(x):
    return (pd.to_numeric(x, errors="coerce") / 0.1).round() * 0.1


def main():
    parser = argparse.ArgumentParser(description="Create a reproducible stratified FIRMS sample for Project 2.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fire-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Raw FIRMS file not found: {path}")

    usecols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])
    df["hour"] = (df["acq_time"] // 100).clip(0, 23).astype(int)
    df["grid_lat"] = round_coord(df["latitude"])
    df["grid_lon"] = round_coord(df["longitude"])

    grouped = (
        df.groupby(["grid_lat", "grid_lon", "acq_date", "hour"], as_index=False)
        .agg(
            fire_detections=("latitude", "size"),
            mean_confidence=("confidence", "mean"),
            max_frp=("frp", "max"),
        )
    )
    grouped["fire"] = 1
    grouped["year"] = grouped["acq_date"].dt.year

    n = min(args.fire_samples, len(grouped))
    years = sorted(grouped["year"].unique())
    base = n // len(years)
    remainder = n % len(years)
    selected_indices = []
    for pos, year in enumerate(years):
        part = grouped[grouped["year"] == year]
        take = min(len(part), base + (1 if pos < remainder else 0))
        selected_indices.extend(part.sample(n=take, random_state=args.seed + int(year)).index.tolist())

    selected = grouped.loc[sorted(set(selected_indices))]
    if len(selected) < n:
        remaining = grouped.drop(index=selected.index)
        extra = remaining.sample(n=n-len(selected), random_state=args.seed)
        selected = pd.concat([selected, extra], ignore_index=False)
    fire = selected.sample(n=n, random_state=args.seed).reset_index(drop=True)

    out = fire[["grid_lat", "grid_lon", "acq_date", "hour", "fire_detections", "mean_confidence", "max_frp", "fire"]]
    out.to_csv(args.output, index=False)

    print("=" * 60)
    print("PROJECT 2 FIRMS SAMPLE CREATED")
    print("=" * 60)
    print(f"Raw detections: {len(df):,}")
    print(f"Unique grid/time fire cells: {len(grouped):,}")
    print(f"Selected fire cells: {len(out):,}")
    print(f"Years: {out.acq_date.dt.year.min()}–{out.acq_date.dt.year.max()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
