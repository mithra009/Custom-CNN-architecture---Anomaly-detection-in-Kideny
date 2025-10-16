import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64

# Try tensorflow.keras first
try:
    from tensorflow.keras.models import load_model
except Exception:
    # fallback for older TensorFlow installations
    from keras.models import load_model

st.set_page_config(page_title="Kidney Image Classifier", layout="centered")

MODEL_H5 = "model.h5"
MODEL_PY = "model.py"

@st.cache_resource
def load_trained_model():
    # Attempt to load model.h5 in current directory
    cwd = os.getcwd()
    paths_tried = []
    h5_path = os.path.join(cwd, MODEL_H5)
    if os.path.exists(h5_path):
        try:
            model = load_model(h5_path)
            return model, f"Loaded {h5_path}"
        except Exception as e:
            # Common problem: custom loss (e.g. 'focal_loss') cannot be resolved
            err = str(e)
            # Try loading without compiling (skips resolving loss/optimizer)
            try:
                model = load_model(h5_path, compile=False)
                return model, f"Loaded {h5_path} (loaded with compile=False due to: {err})"
            except Exception as e2:
                paths_tried.append((h5_path, err + " | " + str(e2)))

    # If model.py exists and defines a build_model function, try importing it
    py_path = os.path.join(cwd, MODEL_PY)
    if os.path.exists(py_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_def", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_model"):
                model = mod.build_model()
                # try to load weights if weights file exists
                try:
                    weights_path = os.path.join(cwd, "model_weights.h5")
                    if os.path.exists(weights_path):
                        model.load_weights(weights_path)
                except Exception:
                    pass
                return model, f"Built model from {py_path}"
        except Exception as e:
            paths_tried.append((py_path, str(e)))

    return None, f"Model not found. Tried: {paths_tried}"

def preprocess_image(image: Image.Image, target_size=(224,224)) -> np.ndarray:
    # Convert to RGB, resize, normalize to [0,1]
    if image.mode != "RGB":
        image = image.convert("RGB")
    # target_size is (height, width)
    image = image.resize((target_size[1], target_size[0]))
    arr = np.array(image).astype("float32") / 255.0
    # add batch dim
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr

LABELS = ["cyst", "stone", "normal", "tumor"]

def make_report(pred_label: str, prob: float):
    # Static report content based on prediction
    lines = []
    lines.append("Kidney Image Classification Report")
    lines.append("=================================")
    lines.append(f"Prediction: {pred_label}")
    lines.append(f"Confidence: {prob*100:.2f}%")
    lines.append("")
    lines.append("Interpretation:")
    if pred_label == "cyst":
        lines.append("- The model predicts a cyst. Cysts are usually fluid-filled sacs; consider clinical correlation and ultrasound follow-up.")
    elif pred_label == "stone":
        lines.append("- The model predicts a stone. Stones are calcifications that may cause obstruction; correlate with symptoms and consider CT or ultrasound.")
    elif pred_label == "normal":
        lines.append("- The model predicts a normal kidney. No obvious abnormality detected by the model; correlate clinically.")
    elif pred_label == "tumor":
        lines.append("- The model predicts a tumor. This is a higher-risk finding; recommend urgent clinical follow-up and imaging (contrast CT/MRI) and specialist referral.")
    else:
        lines.append("- Unknown label.")

    lines.append("")
    lines.append("Disclaimer: This is an automated model output for educational/demo purposes only. Not a medical diagnosis.")
    return "\n".join(lines)

def get_download_link(text: str, filename: str = "report.txt"):
    b = text.encode("utf-8")
    b64 = base64.b64encode(b).decode()
    href = f"data:text/plain;base64,{b64}"
    return href

def main():
    st.title("Kidney Image Classifier")
    st.markdown("Upload a kidney image (JPG/PNG). The model will predict one of: cyst, stone, normal, tumor.")

    model, status = load_trained_model()
    if model is None:
        st.warning(status)
    else:
        st.success(status)

    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"] )
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)

            # Determine target size from model if available
            target_size = (224, 224)
            try:
                if hasattr(model, 'input_shape') and model.input_shape is not None:
                    # model.input_shape can be (None, H, W, C) or (None, C, H, W)
                    ishape = model.input_shape
                    # if tuple of tuples (for multi-input models) pick first
                    if isinstance(ishape, list) or isinstance(ishape, tuple) and isinstance(ishape[0], (list, tuple)):
                        ishape = ishape[0]
                    # convert to list to inspect
                    ish = list(ishape)
                    # remove None dims
                    ish = [d for d in ish if d is not None]
                    if len(ish) >= 3:
                        # try to find H and W
                        if ish[0] in (1,3) and ish[1] > 0 and ish[2] > 0:
                            # format (C,H,W)
                            target_size = (ish[1], ish[2])
                        else:
                            # assume (H,W,C)
                            target_size = (ish[0], ish[1])
            except Exception:
                target_size = (224,224)

            X = preprocess_image(image, target_size=target_size)
            if model is None:
                st.error("No model available to predict. Place model.h5 in the app folder or provide model.py with build_model.")
            else:
                preds = model.predict(X)
                # handle outputs that are logits or multi-dim
                preds = np.asarray(preds).reshape(-1)
                # if len matches labels
                if preds.size == len(LABELS):
                    probs = np.exp(preds) / np.sum(np.exp(preds)) if (preds.min() < 0 or preds.max() > 1) else preds
                    idx = int(np.argmax(probs))
                    label = LABELS[idx]
                    prob = float(probs[idx])
                else:
                    # fallback: if single output, map thresholds
                    if preds.size == 1:
                        v = float(preds[0])
                        # map to normal vs tumor: naive
                        if v < 0.25:
                            label = "normal"
                            prob = 1 - v
                        elif v < 0.5:
                            label = "cyst"
                            prob = 0.7
                        elif v < 0.75:
                            label = "stone"
                            prob = 0.8
                        else:
                            label = "tumor"
                            prob = v
                    else:
                        # unknown shape
                        label = "unknown"
                        prob = 0.0

                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {prob*100:.2f}%")

                report = make_report(label, prob)
                st.text_area("Static report", value=report, height=220)
                href = get_download_link(report, "report.txt")
                st.markdown(f"[Download report]({href})")

        except Exception as e:
            st.error(f"Error processing image: {e}")

    st.sidebar.header("Model info")
    st.sidebar.write(status)
    st.sidebar.write("Labels: ")
    st.sidebar.write(LABELS)

if __name__ == '__main__':
    main()
