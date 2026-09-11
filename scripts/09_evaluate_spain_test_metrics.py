from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INDEX = Path("data/interim/spain_dataset_index.csv")
PRED_DIR = Path("outputs/test_evaluation/predictions")
OUT_DIR = Path("outputs/test_evaluation")

METRICS_CSV = OUT_DIR / "spain_test_metrics.csv"
SUMMARY_TXT = OUT_DIR / "spain_test_summary.txt"


def compute_metrics(gt_mask, pred_mask):
    gt = gt_mask > 0
    pred = pred_mask > 0

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    union = np.logical_or(gt, pred).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou": float(iou),
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "f1": float(f1) if not np.isnan(f1) else np.nan,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX)
    test = df[df["split"] == "test"].copy()

    records = []

    for i, (_, row) in enumerate(test.iterrows(), start=1):
        gt_path = Path(row["mask_path"])
        pred_path = PRED_DIR / f"{gt_path.stem}_pred.tif"

        with rasterio.open(gt_path) as gt_src:
            gt = gt_src.read(1)

        with rasterio.open(pred_path) as pred_src:
            pred = pred_src.read(1)

        metrics = compute_metrics(gt, pred)

        records.append({
            "aoi_id": row["aoi_id"],
            "gt_path": str(gt_path),
            "prediction_path": str(pred_path),
            **metrics,
        })

        print(
            f"[{i:03d}/{len(test)}] "
            f"{row['aoi_id']} "
            f"IoU={metrics['iou']:.3f}"
        )

    results = pd.DataFrame(records)
    results.to_csv(METRICS_CSV, index=False)

    summary = {
        "test_tiles": len(results),
        "mean_iou": results["iou"].mean(),
        "median_iou": results["iou"].median(),
        "std_iou": results["iou"].std(),
        "mean_precision": results["precision"].mean(),
        "mean_recall": results["recall"].mean(),
        "mean_f1": results["f1"].mean(),
        "empty_gt_tiles": int((results["gt_pixels"] == 0).sum()),
        "empty_pred_tiles": int((results["pred_pixels"] == 0).sum()),
    }

    with open(SUMMARY_TXT, "w") as f:
        f.write("Spain Held-Out Test Evaluation\n")
        f.write("=" * 32 + "\n\n")
        f.write("Model: best_model.pth\n")
        f.write("Confidence threshold: 0.30\n")
        f.write(f"Test tiles: {summary['test_tiles']}\n\n")

        f.write(f"Mean IoU:       {summary['mean_iou']:.4f}\n")
        f.write(f"Median IoU:     {summary['median_iou']:.4f}\n")
        f.write(f"Std IoU:        {summary['std_iou']:.4f}\n")
        f.write(f"Mean Precision: {summary['mean_precision']:.4f}\n")
        f.write(f"Mean Recall:    {summary['mean_recall']:.4f}\n")
        f.write(f"Mean F1/Dice:   {summary['mean_f1']:.4f}\n\n")

        f.write(f"Empty GT tiles:   {summary['empty_gt_tiles']}\n")
        f.write(f"Empty pred tiles: {summary['empty_pred_tiles']}\n")

    print("\n=== SPAIN HELD-OUT TEST METRICS ===")
    print(f"Tiles:          {summary['test_tiles']}")
    print(f"Mean IoU:       {summary['mean_iou']:.4f}")
    print(f"Median IoU:     {summary['median_iou']:.4f}")
    print(f"Std IoU:        {summary['std_iou']:.4f}")
    print(f"Mean Precision: {summary['mean_precision']:.4f}")
    print(f"Mean Recall:    {summary['mean_recall']:.4f}")
    print(f"Mean F1/Dice:   {summary['mean_f1']:.4f}")
    print(f"Empty GT tiles: {summary['empty_gt_tiles']}")
    print(f"Empty pred:     {summary['empty_pred_tiles']}")

    print("\nSaved:")
    print(METRICS_CSV)
    print(SUMMARY_TXT)


if __name__ == "__main__":
    main()