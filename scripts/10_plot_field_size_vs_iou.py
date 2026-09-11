from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_PATH = Path("outputs/test_evaluation/spain_test_metrics.csv")
MORPH_PATH = Path("outputs/morphology/valid_chip_morphology.csv")
OUT_DIR = Path("outputs/test_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "field_size_quartile_iou_boxplot.png"


def main():
    metrics = pd.read_csv(METRICS_PATH)
    morph = pd.read_csv(MORPH_PATH)

    morph_test = morph[morph["split"] == "test"].copy()

    df = metrics.merge(
        morph_test[["aoi_id", "median_area_ha"]],
        on="aoi_id",
        how="inner",
    ).dropna(subset=["median_area_ha", "iou"])

    labels = ["Q1 Smallest", "Q2", "Q3", "Q4 Largest"]

    df["field_size_quartile"] = pd.qcut(
        df["median_area_ha"],
        q=4,
        labels=labels,
    )

    groups = [
        df.loc[df["field_size_quartile"] == label, "iou"]
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        groups,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_title("Held-out IoU by Median Field Size Quartile")
    ax.set_xlabel("Median field size group")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)

    summary = (
        df.groupby("field_size_quartile", observed=True)
        .agg(
            tiles=("aoi_id", "count"),
            median_field_area_ha=("median_area_ha", "median"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
        )
        .round(3)
    )

    print("=== FIELD SIZE × IoU FIGURE ===")
    print(summary)
    print()
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()