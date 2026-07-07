from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from brisc_utils import DEFAULT_DATA_DIR, dice_score, ensure_reports_dir, iou_score, load_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a simple segmentation baseline on BRISC 2025.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.18)
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main() -> None:
    args = parse_args()
    reports_dir = ensure_reports_dir()
    df = load_manifest(args.data_dir)

    images = df[(df["task"] == "segmentation") & (~df["is_mask"]) & (df["split"] == args.split)].copy()
    masks = df[(df["task"] == "segmentation") & (df["is_mask"]) & (df["split"] == args.split)].copy()
    mask_by_name = {Path(row.filename).stem: row.path for row in masks.itertuples()}
    images = images.sort_values("filename")
    if args.max_samples:
        images = images.head(args.max_samples)

    rows = []
    for row in images.itertuples():
        image = load_gray(row.path)
        mask_path = mask_by_name.get(Path(row.filename).stem)
        if mask_path is None:
            continue
        target = load_gray(mask_path) > 0.5
        threshold = max(float(image.mean() + image.std()), args.threshold)
        prediction = image > threshold
        rows.append(
            {
                "filename": row.filename,
                "dice": dice_score(prediction, target),
                "iou": iou_score(prediction, target),
                "predicted_area": int(prediction.sum()),
                "target_area": int(target.sum()),
            }
        )

    dice_values = [row["dice"] for row in rows]
    iou_values = [row["iou"] for row in rows]
    metrics = {
        "split": args.split,
        "samples": len(rows),
        "threshold_floor": args.threshold,
        "mean_dice": float(np.mean(dice_values)) if rows else None,
        "median_dice": float(np.median(dice_values)) if rows else None,
        "mean_iou": float(np.mean(iou_values)) if rows else None,
        "median_iou": float(np.median(iou_values)) if rows else None,
        "examples": rows[:20],
    }

    write_json(reports_dir / "segmentation_baseline_metrics.json", metrics)
    print(f"Mean Dice: {metrics['mean_dice']:.4f}" if rows else "No paired masks found.")
    print(f"Wrote {reports_dir / 'segmentation_baseline_metrics.json'}")


if __name__ == "__main__":
    main()

