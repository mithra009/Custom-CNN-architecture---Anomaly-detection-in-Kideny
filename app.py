import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*parameter has been deprecated.*')
warnings.filterwarnings('ignore', message='.*use_column_width.*')

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import json
from datetime import datetime


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Kidney Health Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .urgent-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ff6b6b;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """Focal Loss for multi-class classification"""
    
    def __init__(self, alpha=None, gamma=2.0, name='focal_loss', reduction='sum_over_batch_size', **kwargs):
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = tf.constant(alpha, dtype=tf.float32)
            else:
                self.alpha = tf.cast(alpha, tf.float32)
        else:
            self.alpha = None
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        ce = -tf.math.log(p_t)
        focal_loss = modulating_factor * ce
        
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1, keepdims=True)
            focal_loss = alpha_t * focal_loss
        
        return tf.squeeze(focal_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha.numpy().tolist() if self.alpha is not None else None
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================================
# IMPROVED GRAD-CAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """
    Fixed Grad-CAM implementation with improved tensor handling
    """
    def __init__(self, model, layer_name=None):
        self.model = model
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        self.layer_name = layer_name
        
        try:
            # Get the target layer
            target_layer = model.get_layer(self.layer_name)
            
            # Build grad model that outputs both the conv layer and softmax outputs
            self.grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[
                    target_layer.output,
                    model.output
                ]
            )
            
            # Store the model's output shape for validation
            self.num_classes = model.output_shape[-1]
            
        except Exception as e:
            st.error(f"Error building Grad-CAM model: {e}")
            self.grad_model = None

    def _find_last_conv_layer(self):
        """Find the last convolutional layer"""
        conv_layers = []
        
        for layer in self.model.layers:
            if isinstance(layer, (keras.layers.Conv2D, 
                                keras.layers.SeparableConv2D,
                                keras.layers.DepthwiseConv2D)):
                conv_layers.append(layer.name)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in model!")
        
        return conv_layers[-1]

    def generate_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Generate Grad-CAM heatmap with improved tensor handling and bounds checking
        """
        if self.grad_model is None:
            return None

        try:
            # Input validation and preprocessing
            if isinstance(img_array, np.ndarray):
                if len(img_array.shape) == 3:
                    img_array = np.expand_dims(img_array, axis=0)
                elif len(img_array.shape) != 4:
                    raise ValueError(f"Expected 3D or 4D input array, got shape {img_array.shape}")
            
            img_tensor = tf.cast(img_array, tf.float32)
            if img_tensor.shape[0] != 1:
                raise ValueError("Expected batch size of 1")
            
            # First forward pass: Get predictions to validate class index
            _, preds = self.grad_model(img_tensor)
            
            # Handle different prediction formats
            if isinstance(preds, list):
                preds = tf.convert_to_tensor(preds)
            preds = tf.cast(preds, tf.float32)
            
            # Ensure we have the correct prediction shape
            if len(preds.shape) == 1:
                preds = tf.expand_dims(preds, 0)
            
            # Validate and get class index
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            class_idx = int(class_idx)  # Convert to Python int
            
            # Validate class index
            if not 0 <= class_idx < self.num_classes:
                raise ValueError(f"Class index {class_idx} is out of bounds [0, {self.num_classes})")
            
            # Second forward pass with gradient tape
            with tf.GradientTape() as tape:
                conv_output, predictions = self.grad_model(img_tensor)
                
                # Convert predictions safely
                if isinstance(predictions, list):
                    predictions = predictions[-1] if len(predictions) > 1 else predictions[0]
                
                predictions = tf.convert_to_tensor(predictions)
                predictions = tf.cast(predictions, tf.float32)
                
                if len(predictions.shape) == 1:
                    predictions = tf.expand_dims(predictions, 0)
                
                # Safe indexing with bounds check
                if class_idx >= predictions.shape[-1]:
                    st.error(f"Class index {class_idx} exceeds model output size {predictions.shape[-1]}")
                    return None
                
                class_score = predictions[:, class_idx]
                
            # Compute gradients of the class score with respect to conv_output
            grads = tape.gradient(class_score, conv_output)
            
            # Convert to numpy with proper shape handling
            conv_output = conv_output.numpy()[0]  # Remove batch dimension
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2)).numpy()[0]
            
            # Generate weighted feature map
            heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)
            for i, weight in enumerate(pooled_grads):
                heatmap += weight * conv_output[:, :, i]
            
            # Apply ReLU and normalize
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > eps:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            return heatmap
            
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
            return None

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on original image with improved image handling"""
        if heatmap is None:
            return np.array(original_img)

        try:
            # Convert PIL to numpy if needed
            if isinstance(original_img, Image.Image):
                img = np.array(original_img)
            else:
                img = original_img.copy()
            
            # Ensure uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Ensure RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Resize heatmap to image size
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superimpose with error checking
            try:
                superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
            except cv2.error:
                st.error("Error in overlay: Shape mismatch between image and heatmap")
                return img
            
            return superimposed
            
        except Exception as e:
            st.error(f"Error overlaying heatmap: {str(e)}")
            return img


# ============================================================================
# CLINICAL REPORT GENERATOR
# ============================================================================

class ClinicalReportGenerator:
    """
    Automated report generator for kidney anomaly detection
    Creates clinician-friendly text reports
    """
    def __init__(self):
        self.CLASS_DESCRIPTIONS = {
            'Cyst': {
                'description': 'Fluid-filled sacs in the kidney.',
                'clinical_significance': 'Most simple cysts are benign and asymptomatic.',
                'recommendation': 'Follow-up imaging in 6–12 months if clinically indicated.'
            },
            'Normal': {
                'description': 'No apparent abnormalities detected.',
                'clinical_significance': 'Kidneys appear structurally within normal limits.',
                'recommendation': 'Routine health monitoring.'
            },
            'Stone': {
                'description': 'Findings suggest nephrolithiasis (kidney stone).',
                'clinical_significance': 'May cause pain, obstruction, or infection.',
                'recommendation': 'Urology consult; consider metabolic workup and imaging follow-up.'
            },
            'Tumor': {
                'description': 'Mass lesion detected in kidney.',
                'clinical_significance': 'Requires urgent evaluation to rule out malignancy.',
                'recommendation': 'URGENT: Oncology/urology consultation; consider contrast-enhanced imaging and biopsy.'
            }
        }

    def generate_text_report(
        self,
        prediction_class,
        confidence,
        all_probabilities,
        patient_id=None,
        model_name='KidneyAnomalyNet',
        notes=None
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""{'='*70}
                    KIDNEY HEALTH DIAGNOSTIC REPORT
                        AI-Assisted Analysis
{'='*70}

REPORT DETAILS
{'-'*70}
Report Generated:     {timestamp}
Patient ID:           {patient_id or 'Not Provided'}
Analysis Model:       {model_name}

DIAGNOSTIC FINDINGS
{'-'*70}
PRIMARY DIAGNOSIS:    {prediction_class.upper()}
Confidence Level:     {confidence:.2f}%

CLASSIFICATION PROBABILITIES
{'-'*70}
"""
        class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
        for class_name in class_names:
            prob = float(all_probabilities.get(class_name, 0.0))
            report += f"  {class_name:10s}: {prob:6.2f}%  " + ("█" * int(prob / 2)) + "\n"

        desc = self.CLASS_DESCRIPTIONS[prediction_class]['description']
        clin = self.CLASS_DESCRIPTIONS[prediction_class]['clinical_significance']
        rec  = self.CLASS_DESCRIPTIONS[prediction_class]['recommendation']

        report += f"""
CLINICAL INTERPRETATION
{'-'*70}
Condition:            {desc}

Clinical Significance:
  {clin}

RECOMMENDATIONS
{'-'*70}
  {rec}
"""

        # Urgency flag
        if prediction_class == 'Tumor':
            report += f"""
{'='*70}
⚠️  URGENT ATTENTION REQUIRED
{'='*70}
Potential mass lesion requiring prompt specialist evaluation.
"""

        elif prediction_class == 'Stone':
            report += f"""
CLINICAL NOTE
{'-'*70}
Kidney stones may need intervention depending on size, location, and symptoms.
Assess for obstruction, hydronephrosis, infection, and renal function if indicated.
"""

        if notes:
            report += f"""
ADDITIONAL NOTES
{'-'*70}
{notes}
"""

        report += f"""
DISCLAIMER
{'-'*70}
This AI-assisted report supports clinical decision-making and must be
correlated with clinical history, physical examination, and radiologist review.

Model Summary:
  - Trained with supervised multi-class classification (Cyst/Normal/Stone/Tumor)
  - Uses CNN with class weighting and augmentation
  - Evaluate local validation metrics for deployment context

{'='*70}
                    END OF REPORT
{'='*70}
"""
        return report


# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

@st.cache_resource
def load_model(model_path='models/model_78.h5'):
    """Load model with caching"""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info(f"**Setup Instructions:**\n\n1. Create directory: `mkdir -p {os.path.dirname(model_path)}`\n2. Place your trained model at: `{model_path}`")
        return None
    
    try:
        custom_objects = {
            'FocalLoss': FocalLoss,
            'focal_loss': FocalLoss()
        }
        
        with st.spinner('🔄 Loading model...'):
            model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
        
        st.success('✅ Model loaded successfully!')
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image, img_size=128):
    """Preprocess image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_img = image.copy()
        img_resized = image.resize((img_size, img_size), Image.LANCZOS)
        
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_img, img_resized
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None, None


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_image(model, img_array, class_names):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        results = {
            'predicted_class': predicted_class,
            'predicted_index': int(predicted_idx),
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(probabilities[i])
                for i in range(len(class_names))
            }
        }
        
        sorted_probs = sorted(
            results['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        results['top_predictions'] = sorted_probs
        
        return results
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_probabilities(probabilities, class_names):
    """Plot probability distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.barh(class_names, probabilities, color=colors)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.2f}%',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


# ============================================================================
# CLASS INFORMATION
# ============================================================================

def get_class_info(class_name):
    """Get information about each kidney condition"""
    info = {
        'Cyst': {
            'description': 'Kidney cysts are fluid-filled sacs that form in or on the kidneys.',
            'severity': 'Low to Moderate',
            'color': '#FF6B6B',
            'icon': '🔴'
        },
        'Normal': {
            'description': 'Healthy kidney with no detected abnormalities.',
            'severity': 'None',
            'color': '#4ECDC4',
            'icon': '✅'
        },
        'Stone': {
            'description': 'Kidney stones are hard deposits of minerals and salts in the kidneys.',
            'severity': 'Moderate to High',
            'color': '#45B7D1',
            'icon': '💎'
        },
        'Tumor': {
            'description': 'Abnormal tissue growth in the kidney that may be benign or malignant.',
            'severity': 'High',
            'color': '#FFA07A',
            'icon': '⚠️'
        }
    }
    return info.get(class_name, {})


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Kidney Health Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Kidney Disease Classification with Visual Explanations</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Model configuration
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/model_78.h5",
        help="Path to the trained model file"
    )
    
    img_size = st.sidebar.slider(
        "Image Size",
        64, 256, 128, 32,
        help="Input image size (must match training)"
    )
    
    st.sidebar.markdown("---")
    
    # Grad-CAM settings
    st.sidebar.subheader("📊 Grad-CAM Settings")
    use_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True)
    gradcam_alpha = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.4, 0.05)
    colormap_options = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'PLASMA': cv2.COLORMAP_PLASMA
    }
    colormap_name = st.sidebar.selectbox("Colormap", list(colormap_options.keys()))
    colormap = colormap_options[colormap_name]
    
    st.sidebar.markdown("---")
    
    # Report Generation Settings
    st.sidebar.subheader("📝 Report Settings")
    enable_reports = st.sidebar.checkbox("Enable Clinical Reports", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About:** This app uses a deep learning model trained with Focal Loss "
        "to classify kidney conditions. Grad-CAM visualization shows which regions "
        "of the image were most important for the prediction."
    )
    
    # Class names
    class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.error("⚠️ Please ensure the model file exists at the specified path and try again.")
        st.stop()
    
    # Main content area - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a kidney CT/ultrasound image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a kidney image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
            
            if analyze_button:
                with st.spinner('🔄 Processing image...'):
                    # Preprocess image
                    img_array, original_img, img_resized = preprocess_image(image, img_size)
                    
                    if img_array is not None:
                        # Make prediction
                        results = predict_image(model, img_array, class_names)
                        
                        if results is not None:
                            # Store results in session state
                            st.session_state['results'] = results
                            st.session_state['original_img'] = original_img
                            st.session_state['img_resized'] = img_resized
                            st.session_state['img_array'] = img_array
                            st.session_state['model'] = model
                            st.session_state['use_gradcam'] = use_gradcam
                            st.session_state['gradcam_alpha'] = gradcam_alpha
                            st.session_state['colormap'] = colormap
                            st.session_state['enable_reports'] = enable_reports
                            st.rerun()
        else:
            st.info("👆 Please upload a kidney image to begin analysis")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            
            # Display prediction with colored box
            class_info = get_class_info(predicted_class)
            
            st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: {class_info.get('color', '#ccc')}22; border-left: 5px solid {class_info.get('color', '#ccc')}; margin-bottom: 20px;'>
                    <h2 style='margin: 0;'>{class_info.get('icon', '🏥')} {predicted_class}</h2>
                    <h3 style='margin: 10px 0;'>Confidence: {confidence*100:.2f}%</h3>
                    <p style='margin: 5px 0;'><strong>Severity:</strong> {class_info.get('severity', 'Unknown')}</p>
                    <p style='margin: 5px 0;'>{class_info.get('description', '')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            st.markdown("### 📈 Detailed Probabilities")
            
            # Create metrics row
            metric_cols = st.columns(4)
            for i, (cls, prob) in enumerate(results['top_predictions']):
                with metric_cols[i]:
                    st.metric(
                        label=cls,
                        value=f"{prob*100:.1f}%"
                    )
            
            # Plot probabilities
            st.markdown("### 📊 Probability Distribution")
            probs = [results['probabilities'][cls] for cls in class_names]
            fig_probs = plot_probabilities(probs, class_names)
            st.pyplot(fig_probs)
            plt.close()
            
            # Store figure for download
            st.session_state['fig_probs'] = fig_probs
        else:
            st.info("👈 Upload an image and click 'Analyze Image' to see results")
    
    # Grad-CAM visualization (full width)
    if 'results' in st.session_state and st.session_state.get('use_gradcam', True):
        st.markdown("---")
        st.subheader("🔥 Grad-CAM Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Prediction")
            with st.spinner('Generating primary visualization...'):
                try:
                    # Initialize Grad-CAM
                    model = st.session_state['model']
                    gradcam = GradCAM(model)
                    
                    # Generate heatmap for predicted class
                    img_array = st.session_state['img_array']
                    pred_index = st.session_state['results']['predicted_index']
                    heatmap = gradcam.generate_heatmap(img_array, pred_index)
                    
                    if heatmap is not None:
                        # Create overlay
                        img_resized = st.session_state['img_resized']
                        overlay = gradcam.overlay_heatmap(
                            heatmap,
                            img_resized,
                            alpha=st.session_state.get('gradcam_alpha', 0.4),
                            colormap=st.session_state.get('colormap', cv2.COLORMAP_JET)
                        )
                        
                        st.image(overlay, caption=f"Attention Map for {st.session_state['results']['predicted_class']}", use_column_width=True)
                        
                        # Store overlay for report
                        st.session_state['gradcam_overlay'] = overlay
                    else:
                        st.warning("Could not generate Grad-CAM visualization")
                        
                except Exception as e:
                    st.error(f"Error in primary visualization: {e}")

        with col2:
            st.markdown("### Alternative Analysis")
            class_idx = st.selectbox(
                "View Grad-CAM for other classes:",
                range(len(class_names)),
                format_func=lambda x: class_names[x]
            )
            
            if class_idx is not None:
                with st.spinner('Generating alternative visualization...'):
                    try:
                        model = st.session_state['model']
                        gradcam = GradCAM(model)
                        img_array = st.session_state['img_array']
                        img_resized = st.session_state['img_resized']
                        
                        heatmap = gradcam.generate_heatmap(img_array, class_idx)
                        if heatmap is not None:
                            overlay = gradcam.overlay_heatmap(
                                heatmap,
                                img_resized,
                                alpha=st.session_state.get('gradcam_alpha', 0.4),
                                colormap=st.session_state.get('colormap', cv2.COLORMAP_JET)
                            )
                            st.image(overlay, caption=f"Attention Map for {class_names[class_idx]}", use_column_width=True)
                        else:
                            st.warning("Could not generate alternative visualization")
                            
                    except Exception as e:
                        st.error(f"Error in alternative visualization: {e}")

        # Add explanation
        st.info(
            """
            **💡 Understanding Grad-CAM:**
            - Red/warm colors show regions most important for the prediction
            - Blue/cool colors indicate less important areas
            - Compare attention maps across classes to understand model focus
            - This helps validate if the model uses meaningful features for classification
            """
        )
    
    # Clinical Report Section (NEW)
    if 'results' in st.session_state and st.session_state.get('enable_reports', True):
        st.markdown("---")
        st.subheader("📋 Clinical Report Generation")
        
        # Report input fields
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input(
                "Patient ID (Optional)",
                placeholder="e.g., PATIENT-12345",
                help="Enter patient identifier for the report"
            )
        
        with col2:
            model_name = st.text_input(
                "Model Name",
                value="KidneyAnomalyNet",
                help="Name of the AI model used for analysis"
            )
        
        clinical_notes = st.text_area(
            "Additional Clinical Notes (Optional)",
            placeholder="Enter any additional observations, patient history, or clinical context...",
            height=100,
            help="Add context or observations to include in the report"
        )
        
        # Generate report button
        if st.button("📝 Generate Clinical Report", use_container_width=True):
            with st.spinner('Generating comprehensive clinical report...'):
                try:
                    # Initialize report generator
                    report_gen = ClinicalReportGenerator()
                    
                    # Get results
                    results = st.session_state['results']
                    predicted_class = results['predicted_class']
                    confidence = results['confidence'] * 100  # Convert to percentage
                    
                    # Convert probabilities to percentage
                    all_probs = {k: v * 100 for k, v in results['probabilities'].items()}
                    
                    # Generate report
                    report_text = report_gen.generate_text_report(
                        prediction_class=predicted_class,
                        confidence=confidence,
                        all_probabilities=all_probs,
                        patient_id=patient_id if patient_id else None,
                        model_name=model_name,
                        notes=clinical_notes if clinical_notes else None
                    )
                    
                    # Store report in session state
                    st.session_state['clinical_report'] = report_text
                    st.success("✅ Clinical report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
        
        # Display generated report
        if 'clinical_report' in st.session_state:
            st.markdown("---")
            st.markdown("### 📄 Generated Clinical Report")
            
            # Show urgency warning if applicable
            results = st.session_state['results']
            if results['predicted_class'] == 'Tumor':
                st.markdown("""
                    <div class="urgent-warning">
                        <h3 style='margin: 0; color: #ff6b6b;'>⚠️ URGENT ATTENTION REQUIRED</h3>
                        <p style='margin: 10px 0 0 0;'><strong>Potential mass lesion detected requiring prompt specialist evaluation.</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            elif results['predicted_class'] == 'Stone':
                st.info("ℹ️ **Clinical Note:** Kidney stones may require intervention depending on size, location, and symptoms.")
            
            # Display report in expandable section
            with st.expander("📖 View Full Report", expanded=True):
                st.code(st.session_state['clinical_report'], language=None)
            
            # Clinical interpretation section
            st.markdown("---")
            st.markdown("### 🔬 Clinical Interpretation Summary")
            
            report_gen = ClinicalReportGenerator()
            class_desc = report_gen.CLASS_DESCRIPTIONS[results['predicted_class']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class="report-section">
                        <h4>📝 Condition Description</h4>
                        <p>{class_desc['description']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="report-section">
                        <h4>🏥 Clinical Significance</h4>
                        <p>{class_desc['clinical_significance']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="report-section">
                        <h4>💊 Recommendations</h4>
                        <p>{class_desc['recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                confidence_pct = results['confidence'] * 100
                confidence_color = "#4ECDC4" if confidence_pct >= 80 else "#FFA07A" if confidence_pct >= 60 else "#FF6B6B"
                
                st.markdown(f"""
                    <div class="report-section" style="border-left-color: {confidence_color};">
                        <h4>📊 Confidence Assessment</h4>
                        <p><strong>Model Confidence:</strong> {confidence_pct:.2f}%</p>
                        <p>{'High confidence - Results are reliable' if confidence_pct >= 80 else 
                           'Moderate confidence - Consider additional testing' if confidence_pct >= 60 else
                           'Low confidence - Further evaluation recommended'}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Download results section
    if 'results' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        download_cols = st.columns(4)
        
        with download_cols[0]:
            # Download predictions as JSON
            results_json = st.session_state['results'].copy()
            results_json['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json_str = json.dumps(results_json, indent=4)
            st.download_button(
                label="📥 Results (JSON)",
                data=json_str,
                file_name=f"kidney_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with download_cols[1]:
            # Download probability plot
            if 'fig_probs' in st.session_state:
                buf = io.BytesIO()
                st.session_state['fig_probs'].savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 Probability Chart",
                    data=buf,
                    file_name=f"probabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with download_cols[2]:
            # Download clinical report
            if 'clinical_report' in st.session_state:
                st.download_button(
                    label="📥 Clinical Report",
                    data=st.session_state['clinical_report'],
                    file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with download_cols[3]:
            # Download Grad-CAM overlay
            if 'gradcam_overlay' in st.session_state:
                # Convert overlay to PIL Image and save to bytes
                overlay_img = Image.fromarray(st.session_state['gradcam_overlay'].astype('uint8'))
                buf = io.BytesIO()
                overlay_img.save(buf, format='PNG')
                buf.seek(0)
                st.download_button(
                    label="📥 Grad-CAM Image",
                    data=buf,
                    file_name=f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            Always consult with qualified healthcare professionals for medical diagnosis and treatment.</p>
            <p>Powered by TensorFlow & Streamlit | Model: Custom CNN with Focal Loss</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()