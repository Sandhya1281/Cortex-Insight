from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from brisc_utils import DEFAULT_DATA_DIR, build_feature_matrix, ensure_reports_dir, load_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BRISC 2025 image-classification baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def split_classification_rows(df: pd.DataFrame, split: str, limit: int | None) -> pd.DataFrame:
    rows = df[(df["task"] == "classification") & (~df["is_mask"]) & (df["split"] == split)].copy()
    rows = rows.sort_values(["tumor_label", "filename"]).reset_index(drop=True)
    if limit and limit < len(rows):
        rows = rows.groupby("tumor_label", group_keys=False).head(max(1, limit // rows["tumor_label"].nunique()))
    return rows.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    reports_dir = ensure_reports_dir()
    df = load_manifest(args.data_dir)

    train_df = split_classification_rows(df, "train", args.max_train)
    test_df = split_classification_rows(df, "test", args.max_test)

    print(f"Extracting features: {len(train_df)} train, {len(test_df)} test")
    x_train = build_feature_matrix(train_df["path"], size=args.image_size)
    x_test = build_feature_matrix(test_df["path"], size=args.image_size)
    y_train = train_df["tumor_label"].to_numpy()
    y_test = test_df["tumor_label"].to_numpy()

    model = VotingClassifier(
        estimators=[
            (
                "linear_svc",
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=0.03, class_weight="balanced", dual="auto", max_iter=6000, random_state=42),
                ),
            ),
            (
                "knn_cosine",
                make_pipeline(
                    Normalizer(),
                    KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine", n_jobs=-1),
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "pca_hgb",
                make_pipeline(
                    StandardScaler(),
                    PCA(n_components=160, random_state=42),
                    HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, random_state=42),
                ),
            ),
        ],
        voting="hard",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    labels = sorted(train_df["tumor_label"].unique())
    metrics = {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "image_size": args.image_size,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "model": "hard-voting ensemble: LinearSVC + cosine KNN + ExtraTrees + PCA HistGradientBoosting",
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
    }

    write_json(reports_dir / "classification_metrics.json", metrics)
    joblib.dump(model, reports_dir / "classification_baseline.joblib")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Wrote {reports_dir / 'classification_metrics.json'}")
    print(f"Wrote {reports_dir / 'classification_baseline.joblib'}")


if __name__ == "__main__":
    main()
