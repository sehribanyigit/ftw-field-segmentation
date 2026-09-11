from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


MASK_DIR = Path("data/raw/spain_instance_masks")
OUTPUT_CSV = Path("data/interim/spain_instance_density_pilot.csv")


def get_group(filename: str) -> str:
    return filename.split("_")[0]


def count_instances(mask_path: Path) -> dict:
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        ids = np.unique(arr)
        n_instances = int(np.sum(ids != 0))

        return {
            "chip": mask_path.name,
            "group": get_group(mask_path.name),
            "n_instances": n_instances,
            "crs": str(src.crs),
            "height": src.height,
            "width": src.width,
            "res_x": abs(src.res[0]),
            "res_y": abs(src.res[1]),
        }


def print_summary(df: pd.DataFrame) -> None:
    s = df["n_instances"]

    print("\n=== SPAIN PILOT — OVERALL ===")
    print(f"Chips:   {len(df)}")
    print(f"Mean:    {s.mean():.2f}")
    print(f"Median:  {s.median():.2f}")
    print(f"Q1:      {s.quantile(0.25):.2f}")
    print(f"Q3:      {s.quantile(0.75):.2f}")
    print(f"Min:     {s.min()}")
    print(f"Max:     {s.max()}")
    print(f"Total chip-instance occurrences: {s.sum()}")

    print("\n=== BY GROUP ===")
    summary = (
        df.groupby("group")["n_instances"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )
    print(summary)


def main():
    files = sorted(MASK_DIR.glob("*.tif"))

    if not files:
        raise FileNotFoundError(f"No .tif files found in: {MASK_DIR}")

    print(f"Found {len(files)} masks.")

    records = [count_instances(f) for f in files]
    df = pd.DataFrame(records)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print_summary(df)

    print("\n=== METADATA QC ===")
    print("Unique CRS:", df["crs"].nunique())
    print("Unique shapes:", df[["height", "width"]].drop_duplicates().shape[0])
    print("Unique resolutions:", df[["res_x", "res_y"]].drop_duplicates().shape[0])

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()