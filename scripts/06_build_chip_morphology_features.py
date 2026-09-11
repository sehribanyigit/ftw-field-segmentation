from pathlib import Path
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import CRS, Transformer


INDEX_PATH = Path("data/interim/spain_dataset_index.csv")
OUTPUT_PATH = Path("data/interim/spain_chip_morphology.csv")

MIN_PIXELS = 5


def edge_instance_ids(mask):
    edges = np.concatenate(
        [
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ]
    )

    return set(np.unique(edges)) - {0}


def local_utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_transformer(src):
    bounds = src.bounds

    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.bottom + bounds.top) / 2

    target_crs = local_utm_crs(lon, lat)

    return Transformer.from_crs(
        src.crs,
        target_crs,
        always_xy=True,
    )


def remap_to_int32(mask):
    ids = np.unique(mask)
    ids = ids[ids != 0]

    safe = np.zeros(mask.shape, dtype=np.int32)
    mapping = {}

    for new_id, old_id in enumerate(ids, start=1):
        safe[mask == old_id] = new_id
        mapping[new_id] = int(old_id)

    return safe, mapping


def summarize_chip(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

        ids, counts = np.unique(
            mask,
            return_counts=True,
        )

        valid = ids != 0
        field_ids = ids[valid]
        pixel_counts = counts[valid]

        n_fields_total = len(field_ids)

        pixel_count_by_id = {
            int(field_id): int(pixel_count)
            for field_id, pixel_count
            in zip(field_ids, pixel_counts)
        }

        edge_ids = edge_instance_ids(mask)

        safe_mask, mapping = remap_to_int32(mask)

        components = {}

        for geom_json, value in shapes(
            safe_mask,
            mask=(safe_mask != 0),
            transform=src.transform,
        ):
            safe_id = int(value)

            components.setdefault(
                safe_id,
                [],
            ).append(shape(geom_json))

        transformer = make_transformer(src)

        morphology_records = []

        for safe_id, parts in components.items():
            original_id = mapping[safe_id]
            pixel_count = pixel_count_by_id[original_id]

            # Complete field geometry required
            if original_id in edge_ids:
                continue

            # Avoid unreliable micro-instance morphology
            if pixel_count < MIN_PIXELS:
                continue

            geom = unary_union(parts)

            if geom.is_empty:
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty:
                continue

            geom_m = transform(
                transformer.transform,
                geom,
            )

            area_m2 = geom_m.area
            perimeter_m = geom_m.length

            if area_m2 <= 0 or perimeter_m <= 0:
                continue

            compactness = (
                4 * math.pi * area_m2
                / (perimeter_m ** 2)
            )

            if compactness <= 0:
                continue

            morphology_records.append(
                {
                    "pixel_count": pixel_count,
                    "n_components": len(parts),
                    "area_ha": area_m2 / 10000,
                    "perimeter_m": perimeter_m,
                    "compactness": compactness,
                    "shape_complexity": 1 / compactness,
                }
            )

    morph = pd.DataFrame(morphology_records)

    result = {
        "n_fields_total": n_fields_total,
        "n_edge_fields": len(
            set(field_ids.astype(int)) & edge_ids
        ),
        "n_morphology_fields": len(morph),
    }

    if morph.empty:
        for col in [
            "area_ha",
            "perimeter_m",
            "compactness",
            "shape_complexity",
            "pixel_count",
            "n_components",
        ]:
            result[f"median_{col}"] = np.nan

        return result

    for col in [
        "area_ha",
        "perimeter_m",
        "compactness",
        "shape_complexity",
        "pixel_count",
        "n_components",
    ]:
        result[f"median_{col}"] = morph[col].median()

    result["q1_area_ha"] = morph["area_ha"].quantile(0.25)
    result["q3_area_ha"] = morph["area_ha"].quantile(0.75)

    result["q1_shape_complexity"] = (
        morph["shape_complexity"].quantile(0.25)
    )
    result["q3_shape_complexity"] = (
        morph["shape_complexity"].quantile(0.75)
    )

    return result


def main():
    index = pd.read_csv(INDEX_PATH)

    records = []

    print("=== BUILDING CHIP MORPHOLOGY FEATURES ===")
    print(f"Usable chips: {len(index)}")
    print(f"Minimum morphology size: {MIN_PIXELS} pixels")

    for i, row in index.iterrows():
        stats = summarize_chip(
            Path(row["mask_path"])
        )

        records.append(
            {
                "aoi_id": row["aoi_id"],
                "split": row["split"],
                **stats,
            }
        )

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(index)} chips..."
            )

    df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== COMPLETE ===")
    print(f"Chips: {len(df)}")

    print("\nBy split:")
    print(df["split"].value_counts())

    print("\nMedian chip characteristics:")
    print(
        df[
            [
                "n_fields_total",
                "n_morphology_fields",
                "median_area_ha",
                "median_compactness",
                "median_shape_complexity",
            ]
        ]
        .median()
        .round(3)
    )

    print(
        "\nChips with no valid morphology fields:",
        df["median_area_ha"].isna().sum(),
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()