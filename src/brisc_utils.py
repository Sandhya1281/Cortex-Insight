from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parents[1] / "work" / "brisc2025"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def load_manifest(data_dir: Path) -> pd.DataFrame:
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df["path"] = df["relative_path"].map(lambda value: data_dir / str(value).replace("\\", "/"))
    return df


def crop_foreground(arr: np.ndarray) -> np.ndarray:
    mask = arr > max(10, np.percentile(arr, 15))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad = 8
    return arr[max(0, y0 - pad) : min(arr.shape[0], y1 + pad), max(0, x0 - pad) : min(arr.shape[1], x1 + pad)]


def image_features(path: Path, size: int = 48) -> np.ndarray:
    image = Image.open(path).convert("L")
    cropped = crop_foreground(np.asarray(image))
    image = Image.fromarray(cropped).resize((size, size), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gx = np.diff(arr, axis=1, prepend=arr[:, :1])
    gy = np.diff(arr, axis=0, prepend=arr[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)

    blocks = []
    for grid in (4, 6):
        patch_size = size // grid
        for yy in range(grid):
            for xx in range(grid):
                patch = arr[yy * patch_size : (yy + 1) * patch_size, xx * patch_size : (xx + 1) * patch_size]
                gpatch = grad[yy * patch_size : (yy + 1) * patch_size, xx * patch_size : (xx + 1) * patch_size]
                blocks.extend((patch.mean(), patch.std(), np.percentile(patch, 85), gpatch.mean()))

    hist, _ = np.histogram(arr, bins=48, range=(0.0, 1.0), density=True)
    profile = np.concatenate((arr.mean(axis=0), arr.mean(axis=1), grad.mean(axis=0), grad.mean(axis=1)))
    return np.concatenate((arr.ravel(), hist.astype(np.float32), profile, np.asarray(blocks, dtype=np.float32)))


def build_feature_matrix(paths: Iterable[Path], size: int = 48) -> np.ndarray:
    return np.vstack([image_features(path, size=size) for path in paths])


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection) / denominator)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
