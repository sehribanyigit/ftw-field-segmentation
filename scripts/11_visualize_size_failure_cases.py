from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


IMAGE_DIR = Path("prepared_spain/test/images")
GT_DIR = Path("data/raw/masks")
PRED_DIR = Path("outputs/test_evaluation/predictions")
OUT_DIR = Path("outputs/test_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CASES = [
    ("g3_00005_18", "Small fields"),
    ("g7_00005_6",  "Small fields"),
    ("g7_00003_7",  "Small fields"),
    ("g1_00003_12", "Large fields"),
    ("g4_00034_5",  "Large fields"),
    ("g7_00020_15", "Large fields"),
]


def read_image(path):
    with rasterio.open(path) as src:
        arr = src.read()

    # İlk 3 bandı RGB olarak kullan
    rgb = arr[:3].astype("float32")

    # Görsel amaçlı robust stretch
    out = np.zeros_like(rgb)

    for i in range(3):
        band = rgb[i]
        valid = np.isfinite(band)

        if not valid.any():
            continue

        lo, hi = np.percentile(band[valid], [2, 98])

        if hi > lo:
            out[i] = np.clip((band - lo) / (hi - lo), 0, 1)

    return np.moveaxis(out, 0, -1)


def read_mask(path):
    with rasterio.open(path) as src:
        return src.read(1)


fig, axes = plt.subplots(
    nrows=len(CASES),
    ncols=3,
    figsize=(9, 12),
)

for row, (aoi, group) in enumerate(CASES):

    image = read_image(IMAGE_DIR / f"{aoi}.tif")
    gt = read_mask(GT_DIR / f"{aoi}.tif")
    pred = read_mask(PRED_DIR / f"{aoi}_pred.tif")

    # Instance IDs -> binary field masks
    gt_binary = gt > 0
    pred_binary = pred > 0

    axes[row, 0].imshow(image)
    axes[row, 1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)

    axes[row, 0].set_ylabel(
        f"{group}\n{aoi}",
        fontsize=9,
    )

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

axes[0, 0].set_title("Satellite image")
axes[0, 1].set_title("Ground truth")
axes[0, 2].set_title("Prediction")

fig.suptitle(
    "Field Segmentation Failure Analysis: Small vs Large Fields",
    fontsize=14,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output = OUT_DIR / "small_vs_large_field_examples.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight",
)

print("Saved:", output)