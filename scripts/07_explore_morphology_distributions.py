from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("data/interim/spain_chip_morphology.csv")
OUTPUT = Path("outputs/morphology")
OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "median_area_ha",
    "median_compactness",
    "median_shape_complexity",
    "n_morphology_fields",
]


def main():
    df = pd.read_csv(INPUT)

    print("=== MORPHOLOGY EXPLORATION ===")
    print(f"All chips: {len(df)}")

    valid = df[df["n_morphology_fields"] > 0].copy()

    print(f"Valid morphology chips: {len(valid)}")
    print(f"Excluded: {len(df) - len(valid)}")

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(
        valid[FEATURES]
        .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
        .T
        .round(3)
    )

    print("\n=== CORRELATIONS ===")
    print(
        valid[FEATURES]
        .corr(method="spearman")
        .round(3)
    )

    for feature in FEATURES:
        values = valid[feature].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(values, bins=40)

        ax.axvline(
            values.median(),
            linestyle="--",
            label=f"Median = {values.median():.3f}",
        )

        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of chips")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT / f"{feature}_distribution.png",
            dpi=180,
        )

        plt.close(fig)

    valid[FEATURES + ["aoi_id", "split"]].to_csv(
        OUTPUT / "valid_chip_morphology.csv",
        index=False,
    )

    print(f"\nSaved outputs to: {OUTPUT}")


if __name__ == "__main__":
    main()