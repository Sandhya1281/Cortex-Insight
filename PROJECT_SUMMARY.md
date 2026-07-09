# Project Summary

## Completed work

- Extracted and validated the BRISC 2025 dataset.
- Added a reproducible Python project structure.
- Added dataset audit, classification training, and segmentation evaluation scripts.
- Ran the project and generated reports in `reports/`.
- Saved a trained classification baseline model.

## Results from the completed run

- Dataset audit: passed, 0 missing files.
- Manifest rows: 15,586.
- Classification images: 6,000.
- Segmentation image-mask pairs: 4,793.
- Classification ensemble: 5,000 train / 1,000 test images.
- Classification accuracy: 92.10%.
- Segmentation baseline: 860 paired test samples.
- Segmentation mean Dice: 0.1410.
- Segmentation mean IoU: 0.0834.

## Generated artifacts

- `reports/dataset_audit.json`
- `reports/classification_metrics.json`
- `reports/classification_baseline.joblib`
- `reports/segmentation_baseline_metrics.json`

