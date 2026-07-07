from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from brisc_utils import DEFAULT_DATA_DIR, load_manifest
from train_classifier import split_classification_rows


def crop_foreground(arr: np.ndarray) -> np.ndarray:
    mask = arr > max(8, np.percentile(arr, 12))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad_y = max(2, int((y1 - y0) * 0.04))
    pad_x = max(2, int((x1 - x0) * 0.04))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(arr.shape[0], y1 + pad_y)
    x1 = min(arr.shape[1], x1 + pad_x)
    return arr[y0:y1, x0:x1]


def features(path: Path, size: int = 64) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    cropped = crop_foreground(arr)
    image = Image.fromarray(cropped).resize((size, size), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image)
    x = np.asarray(image, dtype=np.float32) / 255.0

    flat = x.flatten()
    hist, _ = np.histogram(x, bins=48, range=(0.0, 1.0), density=True)
    gx = np.diff(x, axis=1, prepend=x[:, :1])
    gy = np.diff(x, axis=0, prepend=x[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    blocks = []
    for grid in (4, 8):
        h = size // grid
        for yy in range(grid):
            for xx in range(grid):
                block = x[yy * h : (yy + 1) * h, xx * h : (xx + 1) * h]
                gblock = grad[yy * h : (yy + 1) * h, xx * h : (xx + 1) * h]
                blocks.extend([block.mean(), block.std(), np.percentile(block, 80), gblock.mean()])
    moments = [x.mean(), x.std(), grad.mean(), grad.std(), np.percentile(x, 10), np.percentile(x, 90)]
    return np.concatenate([flat, hist.astype(np.float32), np.asarray(blocks, dtype=np.float32), np.asarray(moments, dtype=np.float32)])


def matrix(paths, size: int) -> np.ndarray:
    return np.vstack([features(path, size) for path in paths])


def main() -> None:
    df = load_manifest(DEFAULT_DATA_DIR)
    train_df = split_classification_rows(df, "train", None)
    test_df = split_classification_rows(df, "test", None)
    y_train = train_df["tumor_label"].to_numpy()
    y_test = test_df["tumor_label"].to_numpy()

    for size in (48, 64):
        print(f"features {size}")
        x_train = matrix(train_df["path"], size)
        x_test = matrix(test_df["path"], size)
        models = {
            "linear_svc": make_pipeline(StandardScaler(), LinearSVC(C=0.08, class_weight="balanced", dual="auto", random_state=42, max_iter=5000)),
            "rbf_svc": make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", class_weight="balanced")),
            "extra_trees": ExtraTreesClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
            "random_forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        }
        fitted = []
        for name, model in models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            acc = accuracy_score(y_test, pred)
            print(size, name, f"{acc:.4f}")
            fitted.append((name, model))
        vote = VotingClassifier(fitted, voting="hard")
        vote.fit(x_train, y_train)
        pred = vote.predict(x_test)
        print(size, "vote", f"{accuracy_score(y_test, pred):.4f}")


if __name__ == "__main__":
    main()
