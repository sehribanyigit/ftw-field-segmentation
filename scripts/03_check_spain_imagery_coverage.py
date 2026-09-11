from pathlib import Path
import pandas as pd


METADATA = Path("data/raw/chips_spain.parquet")
INVENTORY = Path("/tmp/spain_s2_inventory.txt")


def main():
    df = pd.read_parquet(METADATA)

    expected = set(df["aoi_id"].astype(str))

    window_a = set()
    window_b = set()

    with open(INVENTORY, encoding="utf-8") as f:
        for line in f:
            if ".tif" not in line:
                continue

            path = line.strip().split()[-1]
            chip = Path(path).stem

            if "/window_a/" in path:
                window_a.add(chip)
            elif "/window_b/" in path:
                window_b.add(chip)

    print("=== FTW SPAIN IMAGERY COVERAGE ===")
    print(f"Metadata chips: {len(expected)}")
    print(f"window_a:       {len(window_a)}")
    print(f"window_b:       {len(window_b)}")

    for window, available in [
        ("window_a", window_a),
        ("window_b", window_b),
    ]:
        missing = expected - available
        extra = available - expected

        print(f"\n=== {window} ===")
        print(f"Missing: {len(missing)}")
        print(f"Extra:   {len(extra)}")

        if missing:
            missing_df = df[df["aoi_id"].isin(missing)]

            print("\nMissing by split:")
            print(missing_df["split"].value_counts())

            print("\nMissing IDs:")
            print(sorted(missing))


if __name__ == "__main__":
    main()