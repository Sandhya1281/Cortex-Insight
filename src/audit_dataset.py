from __future__ import annotations

import argparse
from pathlib import Path

from brisc_utils import DEFAULT_DATA_DIR, ensure_reports_dir, load_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the BRISC 2025 dataset layout.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manifest(args.data_dir)
    reports_dir = ensure_reports_dir()

    missing = [str(path) for path in df["path"] if not path.exists()]
    summary = {
        "data_dir": str(args.data_dir),
        "rows": int(len(df)),
        "missing_files": len(missing),
        "tasks": df["task"].value_counts().to_dict(),
        "splits": df["split"].value_counts().to_dict(),
        "classification_by_split_and_label": (
            df[(df["task"] == "classification") & (~df["is_mask"])]
            .groupby(["split", "tumor_label"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        ),
        "segmentation_images": int(((df["task"] == "segmentation") & (~df["is_mask"])).sum()),
        "segmentation_masks": int(((df["task"] == "segmentation") & (df["is_mask"])).sum()),
        "widths": sorted(map(int, df["width"].dropna().unique())),
        "heights": sorted(map(int, df["height"].dropna().unique())),
    }

    write_json(reports_dir / "dataset_audit.json", summary)
    print(f"Wrote {reports_dir / 'dataset_audit.json'}")
    if missing:
        print(f"Missing files detected: {len(missing)}")
    else:
        print("Dataset audit passed with no missing files.")


if __name__ == "__main__":
    main()

