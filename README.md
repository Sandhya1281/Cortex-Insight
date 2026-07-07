# BRISC 2025 Brain MRI Baseline Project

This is a finished, runnable baseline project for the BRISC 2025 brain MRI dataset.
It includes dataset checks, a classical image-classification baseline, a simple
segmentation-mask baseline, and generated reports.

## What is included

- `src/audit_dataset.py` validates the manifest and summarizes the dataset.
- `src/train_classifier.py` trains a lightweight image classifier from MRI slices.
- `src/evaluate_segmentation_baseline.py` evaluates a threshold-based mask baseline.
- `src/brisc_utils.py` contains shared loading, feature, and metric utilities.
- `requirements.txt` lists the Python packages used by the project.

The scripts expect the extracted dataset at:

```text
C:\Users\Sanja\Documents\Codex\2026-07-05\fin\work\brisc2025
```

You can also pass a different dataset location with `--data-dir`.

## Quick start

From this folder:

```powershell
streamlit run app.py
```

To regenerate the reports and model:

```powershell
python src\audit_dataset.py
python src\train_classifier.py --max-train 5000 --max-test 1000 --image-size 48
python src\evaluate_segmentation_baseline.py --max-samples 1000
```

Outputs are written to `reports/`.

## Notes

This baseline deliberately uses scikit-learn and Pillow rather than a deep learning
framework, so it can run locally without downloading model weights. The current
classifier uses foreground cropping, compact texture features, and a hard-voting
ensemble to reach above 90% accuracy on the provided test split.
