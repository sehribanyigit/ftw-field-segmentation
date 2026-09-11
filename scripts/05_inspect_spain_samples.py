from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


INDEX = Path("data/interim/spain_dataset_index.csv")
OUTPUT_DIR = Path("outputs/qc_samples")


def normalize_rgb(arr):
    """
    arr shape: (bands, height, width)
    Uses first 3 bands as RGB.
    """
    rgb = arr[:3].astype("float32")

    out = np.zeros_like(rgb)

    for i in range(3):
        band = rgb[i]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high > low:
            band = (band - low) / (high - low)

        out[i] = np.clip(band, 0, 1)

    return np.transpose(out, (1, 2, 0))


def inspect_pair(row, label):
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    with rasterio.open(image_path) as src:
        image = src.read()

        print(f"\n=== {label.upper()} ===")
        print("AOI:", row["aoi_id"])
        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)
        print("Image CRS:", src.crs)
        print("Image resolution:", src.res)

        for b in range(image.shape[0]):
            band = image[b]
            print(
                f"Band {b+1}: "
                f"min={band.min()} "
                f"max={band.max()} "
                f"median={np.median(band):.2f}"
            )

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        ids = np.unique(mask)
        field_ids = ids[ids != 0]

        print("Mask shape:", mask.shape)
        print("Mask dtype:", mask.dtype)
        print("Mask CRS:", src.crs)
        print("Instances:", len(field_ids))

    rgb = normalize_rgb(image)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(rgb)
    axes[0].set_title(f"{label} — RGB")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title(
        f"Ground truth — {len(field_ids)} instances"
    )
    axes[1].axis("off")

    fig.suptitle(row["aoi_id"])
    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{label}_{row['aoi_id']}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)


def main():
    df = pd.read_csv(INDEX)

    samples = {}

    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]

        if subset.empty:
            continue

        samples[split] = subset.sample(
            n=1,
            random_state=42,
        ).iloc[0]

    for label, row in samples.items():
        inspect_pair(row, label)


if __name__ == "__main__":
    main()