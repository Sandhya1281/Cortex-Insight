from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from brisc_utils import DEFAULT_DATA_DIR, load_manifest
from train_classifier import split_classification_rows


def crop_foreground(arr: np.ndarray) -> np.ndarray:
    mask = arr > max(10, np.percentile(arr, 15))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad = 8
    return arr[max(0, y0 - pad) : min(arr.shape[0], y1 + pad), max(0, x0 - pad) : min(arr.shape[1], x1 + pad)]


def feature(path: Path, size: int = 48) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = crop_foreground(np.asarray(img))
    img = Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    x = np.asarray(img, dtype=np.float32) / 255.0
    gx = np.diff(x, axis=1, prepend=x[:, :1])
    gy = np.diff(x, axis=0, prepend=x[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)

    blocks = []
    for grid in (4, 6):
        h = size // grid
        for yy in range(grid):
            for xx in range(grid):
                patch = x[yy * h : (yy + 1) * h, xx * h : (xx + 1) * h]
                gpatch = grad[yy * h : (yy + 1) * h, xx * h : (xx + 1) * h]
                blocks.extend((patch.mean(), patch.std(), np.percentile(patch, 85), gpatch.mean()))

    hist, _ = np.histogram(x, bins=48, range=(0, 1), density=True)
    profile = np.concatenate((x.mean(axis=0), x.mean(axis=1), grad.mean(axis=0), grad.mean(axis=1)))
    return np.concatenate((x.ravel(), hist.astype(np.float32), profile, np.asarray(blocks, dtype=np.float32)))


def build(paths, size: int = 48) -> np.ndarray:
    return np.vstack([feature(path, size) for path in paths])


def main() -> None:
    df = load_manifest(DEFAULT_DATA_DIR)
    train_df = split_classification_rows(df, "train", None)
    test_df = split_classification_rows(df, "test", None)
    y_train = train_df["tumor_label"].to_numpy()
    y_test = test_df["tumor_label"].to_numpy()

    print("Building compact features...")
    x_train = build(train_df["path"], 48)
    x_test = build(test_df["path"], 48)
    print(x_train.shape, x_test.shape)

    trials = {
        "linear_svc": make_pipeline(StandardScaler(), LinearSVC(C=0.03, class_weight="balanced", dual="auto", max_iter=6000, random_state=42)),
        "knn_cosine": make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine", n_jobs=-1)),
        "extra_trees": ExtraTreesClassifier(n_estimators=350, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=350, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "pca_hgb": make_pipeline(StandardScaler(), PCA(n_components=160, random_state=42), HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, random_state=42)),
    }

    fitted = []
    for name, model in trials.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print(f"{name}: {acc:.4f}")
        if name in {"linear_svc", "knn_cosine", "extra_trees", "pca_hgb"}:
            fitted.append((name, model))

    print("Training vote...")
    vote = VotingClassifier(fitted, voting="hard")
    vote.fit(x_train, y_train)
    pred = vote.predict(x_test)
    print(f"vote: {accuracy_score(y_test, pred):.4f}")


if __name__ == "__main__":
    main()
