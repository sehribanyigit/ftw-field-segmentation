from pathlib import Path

import pandas as pd


METADATA = Path("data/raw/metadata/chips_spain.parquet")
IMAGE_DIR = Path("data/raw/imagery/window_a")
MASK_DIR = Path("data/raw/masks")
OUTPUT = Path("data/interim/spain_dataset_index.csv")


def main():
    df = pd.read_parquet(METADATA)

    rows = []

    for _, row in df.iterrows():
        aoi_id = str(row["aoi_id"])

        image_path = IMAGE_DIR / f"{aoi_id}.tif"
        mask_path = MASK_DIR / f"{aoi_id}.tif"

        rows.append(
            {
                "aoi_id": aoi_id,
                "split": row["split"],
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_exists": image_path.exists(),
                "mask_exists": mask_path.exists(),
            }
        )

    index = pd.DataFrame(rows)

    print("\n=== FTW SPAIN DATASET INDEX ===")
    print(f"Metadata chips: {len(index)}")
    print(f"Images found:   {index['image_exists'].sum()}")
    print(f"Masks found:    {index['mask_exists'].sum()}")

    usable = index[index["image_exists"] & index["mask_exists"]].copy()

    print(f"Usable pairs:   {len(usable)}")

    print("\n=== USABLE BY SPLIT ===")
    print(usable["split"].value_counts())

    missing = index[~(index["image_exists"] & index["mask_exists"])]

    if not missing.empty:
        print("\n=== EXCLUDED ===")
        print(
            missing[
                ["aoi_id", "split", "image_exists", "mask_exists"]
            ].to_string(index=False)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    usable[
        ["aoi_id", "split", "image_path", "mask_path"]
    ].to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()