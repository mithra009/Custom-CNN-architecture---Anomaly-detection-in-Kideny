# Kidney-AI: Deep Learning for Renal Anomaly Detection

Hackathon submission — a compact, end-to-end deep learning solution for rapid detection and interpretation of common kidney anomalies from medical images.

This project demonstrates a production-minded prototype: a custom CNN trained with Focal Loss, an explainability module (Grad-CAM) and a lightweight interactive demo for fast evaluation. It was built to provide a reliable "second opinion" for clinicians and to showcase interpretability in a short-timespan hackathon setting.

## Why this matters
Timely and accurate detection of renal anomalies improves clinical decision-making and patient outcomes. Image-based screening requires expert radiology review; Kidney-AI aims to accelerate preliminary triage by providing quick, explainable predictions and visual attention maps that highlight model-focus regions.

## Problem statement
Manual review of medical scans is time-consuming and resource-intensive. Small datasets and class imbalance (e.g., fewer tumor examples) make training robust models difficult. We focused on building a compact, interpretable model that performs well on imbalanced data and that judges / clinicians can inspect quickly.

## Our solution
- A custom convolutional neural network (no heavy pre-trained backbone) designed for speed and interpretability.
- Focal Loss to effectively handle class imbalance and improve detection of rare but critical classes (e.g., tumors).
- Grad-CAM visual explanations to surface image regions that influenced the model's decision.
- A simple interactive demo (Streamlit) for rapid validation, visualization and exportable reports.

## Key features & innovations
- Custom CNN architecture built with TensorFlow/Keras (see `model.py:get_model`).
- Focal Loss implemented end-to-end (source in `model.py`) to emphasize hard examples.
- Grad-CAM implementation and overlay utilities in `app.py` for visual explainability.
- Small, reproducible demo pipeline and downloadable report output for judges and reviewers.

## Tech stack
- Python 3.8+
- TensorFlow / Keras
- Streamlit (demo)
- OpenCV, Pillow, NumPy, Matplotlib

Minimum dependencies are listed in `requirements.txt`.

## Repository contents
- `app.py` — demo/inference UI, Grad-CAM visualization and report generation.
- `model.py` — network architecture, `FocalLoss`, load & predict helpers.
- `model.h5` — example saved model (replace with your own weights if desired).
- `dl-cnn.ipynb` — notebook with training/experiments used during development.
- `requirements.txt` — pinning of the main Python packages.

## Quick demo — run locally
1. Clone repo and change directory:

```powershell
cd C:\Users\mithr\Desktop\analytica\final
```

2. (Recommended) Create and activate a virtual environment:

PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Bash (macOS / Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Quick inference from the command line (example):

```powershell
python -c "from model import load_robust_model, predict_image; m=load_robust_model('model.h5'); print(predict_image(m,'example.jpg'))"
```

5. Run the visual demo (Streamlit):

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (typically http://localhost:8501).

## Example output
- Prediction object contains: predicted class (Cyst/Normal/Stone/Tumor), confidence score, and per-class probabilities.
- Grad-CAM overlays are produced and can be downloaded as PNGs from the demo.

## Reproducibility & training notes
- Training experiments and augmentation details are in `dl-cnn.ipynb`.
- The model defined by `model.py:get_model(img_size=128)` was trained using class weighting and Focal Loss. To retrain, instantiate the model, compile with the FocalLoss class and call `model.fit(...)` with your dataset.

## Accomplishments (hackathon highlights)
- Built and trained a custom CNN for a clinically relevant detection task within a tight deadline.
- Implemented Focal Loss to tackle class imbalance from literature and applied it successfully in training.
- Integrated Grad-CAM for transparency and trust — a major plus in medical AI demos.
- Delivered an end-to-end prototype that includes a demo UI and downloadable clinician-style reports.

## Limitations & next steps
- Not a certified medical device — intended for research/demo only.
- Next steps if extended beyond the hackathon:
	- More extensive cross-validation and external validation on independent datasets.
	- Confidence calibration and uncertainty estimation for safer clinical use.
	- PDF report export, automated evaluation scripts, and CI tests for inference.
	- Lightweight model quantization for on-device/inference-speed improvements.

## Medical disclaimer
This project is a proof-of-concept created for a hackathon. It is provided for research and educational purposes only and is NOT a medical diagnostic device. Do NOT use this tool to make clinical decisions. Always consult qualified healthcare professionals.



