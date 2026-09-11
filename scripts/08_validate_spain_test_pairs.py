from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INDEX = Path("data/interim/spain_dataset_index.csv")
PRED_DIR = Path("outputs/test_evaluation/predictions")


def main():
    df = pd.read_csv(INDEX)
    test = df[df["split"] == "test"].copy()

    problems = []

    for _, row in test.iterrows():
        gt_path = Path(row["mask_path"])
        pred_path = PRED_DIR / f"{gt_path.stem}_pred.tif"

        # 1. File existence
        if not gt_path.exists():
            problems.append(f"Missing GT: {gt_path}")
            continue

        if not pred_path.exists():
            problems.append(f"Missing prediction: {pred_path}")
            continue

        # 2. Raster compatibility
        with rasterio.open(gt_path) as gt, rasterio.open(pred_path) as pred:

            if gt.shape != pred.shape:
                problems.append(
                    f"Shape mismatch: {gt_path.stem} "
                    f"{gt.shape} != {pred.shape}"
                )

            if gt.crs != pred.crs:
                problems.append(
                    f"CRS mismatch: {gt_path.stem} "
                    f"{gt.crs} != {pred.crs}"
                )

            gt_transform = np.array(tuple(gt.transform))
            pred_transform = np.array(tuple(pred.transform))

            if not np.allclose(
                gt_transform,
                pred_transform,
                rtol=0,
                atol=1e-12,
            ):
                problems.append(
                    f"Transform mismatch: {gt_path.stem}"
                )

    # Only main prediction rasters
    main_preds = list(PRED_DIR.glob("*_pred.tif"))

    print("\n=== SPAIN HELD-OUT TEST PAIR VALIDATION ===")
    print(f"Test rows:        {len(test)}")
    print(f"Main predictions: {len(main_preds)}")
    print(f"Problems:         {len(problems)}")

    if problems:
        print("\nFirst problems:")
        for problem in problems[:10]:
            print("-", problem)

        print("\nSTATUS: FAIL")

    else:
        print("\nSTATUS: PASS")
        print(
            "All GT/prediction pairs are present "
            "and spatially compatible."
        )


if __name__ == "__main__":
    main()