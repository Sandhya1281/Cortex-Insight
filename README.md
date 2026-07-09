**Cortex Insight**
AI-powered brain tumor detection with LLM-generated insights
Cortex Insight is a web app that analyzes brain MRI scans using a deep learning model to detect and classify tumors, then uses an LLM (Grok) to translate the raw prediction into a clear, human-readable explanation and suggested next steps.
Live App: cortex-insight.streamlit.app


**Objective**
Early detection of brain tumors from MRI scans is time-critical but often bottlenecked by specialist availability. Cortex Insight aims to:
Provide fast, preliminary AI-assisted screening of MRI scans.
Make model predictions understandable to non-experts through natural language explanations.
Serve as a lightweight, accessible proof-of-concept for AI-assisted diagnostics — not a replacement for professional medical diagnosis.



**How It Works**
Upload — User uploads a brain MRI scan (JPG/PNG) through the Streamlit interface.
Preprocess — Image is resized, normalized, and reshaped to match the model's expected input.
Predict — A trained CNN classifies the scan (e.g., glioma, meningioma, pituitary tumor, or no tumor) and returns a confidence score.
Explain — The prediction is passed to the Grok LLM API, which generates a plain-English explanation and general next-step suggestions.
Display — The uploaded image, prediction, confidence score, and LLM explanation are shown together on one page.



The architecture deliberately separates prediction (CNN, deterministic) from explanation (LLM, generative) — the LLM never makes the diagnosis itself, it only explains a prediction the model already made. This reduces hallucination risk.




**Tech Stack**
LayerTechnologyFrontend / UIStreamlitML / Deep LearningPython, CNN (TensorFlow/Keras or PyTorch)Image ProcessingOpenCV / PIL, NumPyNatural Language InsightsGrok API (LLM)DeploymentStreamlit Community CloudVersion ControlGitHub


**Project Structure**

├── app.py                  # Main Streamlit app
├── model/                  # Trained CNN model file (.h5 / .pt)
├── utils/
│   ├── preprocess.py       # Image resizing & normalization
│   └── llm.py               # Grok API call wrapper
├── requirements.txt        # Python dependencies
└── README.md


**Features**
Fast MRI scan classification with confidence score.
LLM-generated, easy-to-understand explanations of results.
No login or setup required — instant browser access.
Clean, single-page interface.



**Disclaimer**
This tool is a proof-of-concept for educational/demonstration purposes only. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.


**Future Enhancements**
Train on larger, more diverse MRI datasets for better generalization.
Add tumor segmentation (e.g., U-Net) to highlight the tumor region visually.
Add Grad-CAM visual explainability to show which regions influenced the CNN's decision.
Ensemble models for improved accuracy.
User accounts with scan history tracking.
Offline/low-connectivity mode for clinics with limited internet access.
Clinical validation with radiologists.
