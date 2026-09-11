from pathlib import Path
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import CRS, Transformer


MASK_DIR = Path("data/raw/spain_instance_masks")
OUTPUT_CSV = Path("data/interim/spain_morphology_pilot_v2.csv")

N_CHIPS_PER_GROUP = 100
RANDOM_SEED = 42


def get_group(filename: str) -> str:
    return filename.split("_")[0]


def select_balanced_sample(files):
    df = pd.DataFrame(
        {
            "path": files,
            "group": [get_group(f.name) for f in files],
        }
    )

    samples = []

    for group, subset in df.groupby("group"):
        n = min(N_CHIPS_PER_GROUP, len(subset))

        sampled = subset.sample(
            n=n,
            random_state=RANDOM_SEED,
        )

        samples.append(sampled)

    return pd.concat(samples, ignore_index=True)


def touches_chip_edge(mask):
    edge_values = np.concatenate(
        [
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ]
    )

    return set(np.unique(edge_values)) - {0}


def local_utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def project_geometry(geom, source_crs):
    centroid = geom.centroid
    target_crs = local_utm_crs(
        centroid.x,
        centroid.y,
    )

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    return transform(
        transformer.transform,
        geom,
    )


def remap_to_int32(mask):
    original_ids = np.unique(mask)
    original_ids = original_ids[original_ids != 0]

    safe = np.zeros(
        mask.shape,
        dtype=np.int32,
    )

    mapping = {}

    for new_id, old_id in enumerate(
        original_ids,
        start=1,
    ):
        safe[mask == old_id] = new_id
        mapping[new_id] = int(old_id)

    return safe, mapping


def process_mask(mask_path):
    records = []

    with rasterio.open(mask_path) as src:
        original_mask = src.read(1)
        edge_ids_original = touches_chip_edge(
            original_mask
        )

        safe_mask, id_mapping = remap_to_int32(
            original_mask
        )

        components = {}

        for geom_json, value in shapes(
            safe_mask,
            mask=(safe_mask != 0),
            transform=src.transform,
        ):
            safe_id = int(value)
            geom = shape(geom_json)

            if geom.is_empty:
                continue

            components.setdefault(
                safe_id,
                []
            ).append(geom)

        for safe_id, geom_parts in components.items():
            original_id = id_mapping[safe_id]

            if original_id in edge_ids_original:
                continue

            merged_geom = unary_union(geom_parts)

            if merged_geom.is_empty:
                continue

            if not merged_geom.is_valid:
                merged_geom = merged_geom.buffer(0)

            if merged_geom.is_empty:
                continue

            geom_m = project_geometry(
                merged_geom,
                src.crs,
            )

            area_m2 = geom_m.area
            perimeter_m = geom_m.length

            if area_m2 <= 0 or perimeter_m <= 0:
                continue

            compactness = (
                4 * math.pi * area_m2
                / (perimeter_m ** 2)
            )

            shape_complexity = (
                1 / compactness
                if compactness > 0
                else np.nan
            )

            records.append(
                {
                    "chip": mask_path.name,
                    "group": get_group(mask_path.name),
                    "instance_id": original_id,
                    "n_components": len(geom_parts),
                    "area_m2": area_m2,
                    "area_ha": area_m2 / 10000,
                    "perimeter_m": perimeter_m,
                    "compactness": compactness,
                    "shape_complexity": shape_complexity,
                }
            )

    return records


def print_summary(df):
    print("\n=== SPAIN MORPHOLOGY PILOT V2 ===")
    print(
        f"Valid field instances: {len(df):,}"
    )

    print("\n=== COMPONENT QC ===")
    print(
        "Single-component:",
        (df["n_components"] == 1).sum()
    )
    print(
        "Multi-component:",
        (df["n_components"] > 1).sum()
    )
    print(
        "Multi-component rate:",
        f"{100 * (df['n_components'] > 1).mean():.2f}%"
    )
    print(
        "Max components:",
        df["n_components"].max()
    )

    metrics = [
        "area_ha",
        "perimeter_m",
        "compactness",
        "shape_complexity",
    ]

    for metric in metrics:
        s = df[metric].dropna()

        print(f"\n--- {metric} ---")
        print(f"Mean:   {s.mean():.3f}")
        print(f"Median: {s.median():.3f}")
        print(f"Q1:     {s.quantile(0.25):.3f}")
        print(f"Q3:     {s.quantile(0.75):.3f}")
        print(f"Min:    {s.min():.3f}")
        print(f"Max:    {s.max():.3f}")

    print("\n=== MEDIAN BY GROUP ===")

    print(
        df.groupby("group")[
            [
                "area_ha",
                "perimeter_m",
                "compactness",
                "shape_complexity",
            ]
        ]
        .median()
        .round(3)
    )

    contiguous = df[
        df["n_components"] == 1
    ]

    print(
        "\n=== CONTIGUOUS-ONLY MEDIAN ==="
    )

    print(
        contiguous[
            [
                "area_ha",
                "perimeter_m",
                "compactness",
                "shape_complexity",
            ]
        ]
        .median()
        .round(3)
    )


def main():
    files = sorted(
        MASK_DIR.glob("*.tif")
    )

    if not files:
        raise FileNotFoundError(
            f"No masks found in {MASK_DIR}"
        )

    sample = select_balanced_sample(files)

    print("=== SAMPLE ===")
    print(
        sample["group"]
        .value_counts()
        .sort_index()
    )
    print(
        f"Total sampled chips: {len(sample)}"
    )

    records = []

    for i, row in sample.iterrows():
        records.extend(
            process_mask(row["path"])
        )

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{len(sample)} chips..."
            )

    df = pd.DataFrame(records)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print_summary(df)

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()