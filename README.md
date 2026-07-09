# Cortex Insight

**AI-powered brain tumor detection with LLM-generated insights**

Cortex Insight is a web app that analyzes brain MRI scans using a deep learning model to detect and classify tumors, then uses an LLM (Grok) to translate the raw prediction into a clear, human-readable explanation and suggested next steps.

**Live App:** [cortex-insight.streamlit.app](https://cortex-insight.streamlit.app/)

---

## Objective

Early detection of brain tumors from MRI scans is time-critical but often bottlenecked by specialist availability. Cortex Insight aims to:
- Provide fast, preliminary AI-assisted screening of MRI scans.
- Make model predictions understandable to non-experts through natural language explanations.
- Serve as a lightweight, accessible proof-of-concept for AI-assisted diagnostics — not a replacement for professional medical diagnosis.

---

## How It Works

1. **Upload** — User uploads a brain MRI scan (JPG/PNG) through the Streamlit interface.
2. **Preprocess** — Image is resized, normalized, and reshaped to match the model's expected input.
3. **Predict** — A trained CNN classifies the scan (e.g., glioma, meningioma, pituitary tumor, or no tumor) and returns a confidence score.
4. **Explain** — The prediction is passed to the Grok LLM API, which generates a plain-English explanation and general next-step suggestions.
5. **Display** — The uploaded image, prediction, confidence score, and LLM explanation are shown together on one page.

> The architecture deliberately separates *prediction* (CNN, deterministic) from *explanation* (LLM, generative) — the LLM never makes the diagnosis itself, it only explains a prediction the model already made. This reduces hallucination risk.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML / Deep Learning | Python, CNN (TensorFlow/Keras or PyTorch) |
| Image Processing | OpenCV / PIL, NumPy |
| Natural Language Insights | Grok API (LLM) |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## Project Structure

```
├── app.py                  # Main Streamlit app
├── model/                  # Trained CNN model file (.h5 / .pt)
├── utils/
│   ├── preprocess.py       # Image resizing & normalization
│   └── llm.py               # Grok API call wrapper
├── requirements.txt        # Python dependencies
└── README.md
```

---


## Features

- Fast MRI scan classification with confidence score.
- LLM-generated, easy-to-understand explanations of results.
- No login or setup required — instant browser access.
- Clean, single-page interface.

---

## Disclaimer

This tool is a **proof-of-concept for educational/demonstration purposes only**. It is **not a certified medical device** and should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.

---

