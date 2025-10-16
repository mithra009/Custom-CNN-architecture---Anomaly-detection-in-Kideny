
# Kidney Image Classifier — Streamlit App

This repository contains a small Streamlit web app that performs image classification of kidney images into one of the four classes: `cyst`, `stone`, `normal`, `tumor`.

The app attempts to load a trained Keras/TensorFlow model from `model.h5` (preferred). If not found, it will try to import `model.py` and call a `build_model()` function to construct the model architecture (optionally followed by loading `model_weights.h5`). The app resizes uploaded images to the model's expected input size, runs inference, and produces a small static text report you can download.

This README explains how to set up and run the app, where to put model files, how to handle common problems (custom losses, input-shape mismatches), and a few extension points.

Table of contents
- Requirements
- Quick start (PowerShell commands)
- Model files and expected layout
- Custom losses / custom_objects
- Input shape handling and tips
- Troubleshooting
- Development tips and extending the app
- License & disclaimer

Requirements
- Python 3.8+ (3.10/3.11 recommended)
- A working TensorFlow installation compatible with your model (TensorFlow 2.x)
- The packages in `requirements.txt` (Streamlit, TensorFlow, Pillow, numpy)

Quick start (Windows PowerShell)
1) Open PowerShell and change to the project folder:

```powershell
cd C:\Users\mithr\Desktop\analytica
```

2) (Optional, recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Place your model files (see the next section), then start the app:

```powershell
streamlit run app.py
```

The Streamlit server prints a local URL (typically http://localhost:8501). Open that in your browser.

Model files and layout
- Place these files alongside `app.py` in the same folder (the app searches the current working directory):
	- `model.h5` — recommended: full Keras model saved with `model.save('model.h5')`.
	- `model.py` — optional: Python module exposing `def build_model():` which returns a compiled Keras model (used when `model.h5` is not present).
	- `model_weights.h5` — optional: weights file which `model.py` can load after building the architecture.
	- `custom_objects.py` — optional: if your model used custom layers/losses/metrics, put definitions here and the app can be configured to load them.

Notes about common model formats
- If your `model.h5` was saved with custom losses/metrics (for example `focal_loss`) the app will attempt to load it normally; if Keras cannot resolve the loss, the app will retry with `load_model(..., compile=False)` so inference still works. If you need to re-enable compilation or train further, provide the custom loss function via `custom_objects`.

Handling custom losses / custom_objects
- If your model requires custom objects (custom layers, losses, or metrics), create a `custom_objects.py` in the same folder with the required definitions. Example:

```python
# custom_objects.py
import tensorflow as tf

def focal_loss(y_true, y_pred):
		# simple placeholder example — replace with your implementation
		gamma = 2.0
		alpha = 0.25
		bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
		p_t = tf.exp(-bce)
		loss = alpha * (1 - p_t) ** gamma * bce
		return loss

CUSTOM_OBJECTS = {"focal_loss": focal_loss}
```

You can then modify `app.py` to import `CUSTOM_OBJECTS` and pass it to `load_model(..., custom_objects=CUSTOM_OBJECTS)` when loading the model. If you want, I can add automatic `custom_objects` loading to the app.

Input shape handling
- The app inspects `model.input_shape` and resizes uploaded images to match the model's expected height and width before prediction. Common formats handled:
	- `(None, H, W, C)` — typical TensorFlow/Keras format
	- `(None, C, H, W)` — less common; handled heuristically
- If your model expects a non-RGB input (e.g., single-channel) or a different preprocessing (mean/std normalization, channel swap, scaling to a different range), update `preprocess_image()` inside `app.py` to match the original preprocessing used at training time.

Troubleshooting
- Error: "Could not interpret loss identifier: focal_loss"
	- The app will attempt to load with `compile=False` to allow inference without resolving the loss. To enable compiling, provide `custom_objects.py` (see above) and modify `app.py` to pass `custom_objects` into `load_model()`.

- Error: "Input ... is incompatible with the layer: expected shape=(None, 128, 128, 3), found shape=(1, 224, 224, 3)"
	- The app resizes images automatically using the model's `input_shape`. If your model expects 128×128, either:
		- Ensure `model.h5` correctly reports `input_shape`, or
		- Set the desired input size manually in `app.py` by changing the fallback `target_size` or adding a configuration variable.

- Streamlit deprecation warnings
	- The app uses `use_container_width=True` for `st.image()` (recent Streamlit versions). If you see warnings related to Streamlit API changes, upgrade Streamlit or let me update the app to match the exact version you're running.

- Failed imports or missing packages
	- Ensure you're running in the same virtual environment where you installed packages. `pip show streamlit` and `pip show tensorflow` help verify installation.

Development tips & extensions
- Automatically load custom_objects: I can add logic to detect and import `custom_objects.py` and pass its `CUSTOM_OBJECTS` dict to `load_model()`.
- Add a config section at top of `app.py` to override input shape, labels, or model path.
- Create a small test script to validate model loading and a single inference (useful to debug model/TF version mismatches).

Example: test_model.py (quick local check)

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model.h5', compile=False)
print('Loaded OK, input shape =', model.input_shape)
arr = np.zeros((1,)+tuple([d for d in model.input_shape if d is not None][1:]), dtype='float32')
pred = model.predict(arr)
print('Pred shape:', pred.shape)
```

Checklist before sharing or deploying
- Verify the model loads locally (run the example above).
- Confirm the model's expected input size and preprocessing steps. Update `preprocess_image()` if needed.
- Ensure any custom objects are provided via `custom_objects.py` or added to `app.py`.

License & disclaimer
- This code is provided as-is for demonstration and research. The generated reports and predictions are NOT medical diagnoses. Always consult a qualified clinician for medical interpretation.

If you want, I can:
- Add automatic `custom_objects` loading from `custom_objects.py`.
- Add a UI control that shows the detected `model.input_shape` and lets you override it manually.
- Generate PDF reports including the uploaded image.

If you'd like any of the above, tell me which feature and I'll implement it.

