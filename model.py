import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for multi-class classification (Lin et al., 2017)
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = tf.constant(alpha, dtype=tf.float32)
            else:
                self.alpha = tf.cast(alpha, tf.float32)
        else:
            self.alpha = None
    
    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate p_t: probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        
        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Cross entropy: -log(p_t)
        ce = -tf.math.log(p_t)
        
        # Focal loss
        focal_loss = modulating_factor * ce
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1, keepdims=True)
            focal_loss = alpha_t * focal_loss
        
        return tf.squeeze(focal_loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha.numpy().tolist() if isinstance(self.alpha, tf.Tensor) and self.alpha.shape.rank > 0 else self.alpha,
            'gamma': self.gamma,
        })
        return config

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def get_model(num_classes=4, img_size=128, dropout_rate=0.5):
    """
    Enhanced CNN for kidney classification
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name='input_image')
    
    # Data augmentation (these will be inactive during inference)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.15)(x)
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return keras.models.Model(inputs=inputs, outputs=outputs, name='KidneyNet_FocalLoss')

# ============================================================================
# LOAD MODEL FUNCTION
# ============================================================================

def load_robust_model(model_path='model_focal_latest.h5'):
    """
    Load the saved model with custom objects.
    Handles potential issues with custom loss.
    """
    try:
        # Load with custom objects
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'FocalLoss': FocalLoss
            },
            compile=True
        )
        print("✅ Model loaded successfully with compilation.")
    except Exception as e:
        print(f"Warning: Error during loading with compile: {str(e)}")
        print("Attempting to load without compiling...")
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'FocalLoss': FocalLoss
            },
            compile=False
        )
        print("✅ Model loaded without compilation (suitable for inference).")
    
    return model

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
IMG_SIZE = 128

def predict_image(model, image_path):
    """
    Preprocess and predict on a single image.
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
    
    Returns:
        dict with class name and confidence
    """
    # Load and preprocess image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img_array)
    pred_class_idx = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    return {
        'class': pred_class,
        'confidence': float(confidence),
        'all_probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(predictions[0])}
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load the model
    model = load_robust_model('model_focal_latest.h5')  # Adjust path if needed
    
    # Example inference
    # result = predict_image(model, 'path/to/your/image.jpg')
    # print(result)