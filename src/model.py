"""
TensorFlow model loading and inference for Brain Tumor MRI classification.
Predicts: glioma, meningioma, pituitary, no_tumor
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

# Tumor class labels (matches common brain tumor dataset conventions)
CLASS_LABELS = ["pituitary", "no_tumor", "meningioma", "glioma"]

# Alternative mapping for models using "notumor" internally
CLASS_INDEX_TO_LABEL = {
    0: "pituitary",
    1: "no_tumor",   # notumor -> no_tumor for display
    2: "meningioma",
    3: "glioma",
}


def _find_model_path() -> Optional[str]:
    """Find the model file - check data directory for best_cnn_model.h5."""
    base_dir = Path(__file__).resolve().parent.parent
    paths = [
        base_dir / "data" / "best_cnn_model.h5",
        base_dir / "data" / "tumor_model.h5",
        base_dir / "best_cnn_model.h5",  # Fallback
    ]
    
    # First try to find existing model
    for p in paths:
        if p.exists():
            return str(p)
    
    # If not found, try to restore from backup
    backup_path = base_dir / "data" / "best_cnn_model.h5.backup"
    if backup_path.exists():
        try:
            import shutil
            target_path = base_dir / "data" / "best_cnn_model.h5"
            shutil.copy2(backup_path, target_path)
            return str(target_path)
        except Exception:
            pass  # Failed to copy, continue to error
    
    return None


def load_model(model_path: Optional[str] = None) -> tf.keras.Model:
    """
    Load the TensorFlow tumor classification model.
    
    Args:
        model_path: Optional path to model. If None, auto-detect.
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If no model file found
    """
    path = model_path or _find_model_path()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "No model file found. Place 'tumor_model.h5' in this directory "
            "or 'best_cnn_model.h5' in the parent directory."
        )
    return tf.keras.models.load_model(path)


def predict_tumor(
    image: np.ndarray,
    model: tf.keras.Model
) -> tuple:
    """
    Run tumor classification on a preprocessed MRI image.
    
    Pipeline:
    1. Image is expected preprocessed (224x224, normalized, batched)
    2. Run model prediction
    3. Extract class index and confidence
    4. Map to class label
    
    Args:
        image: Preprocessed image (1, 224, 224, 3)
        model: Loaded Keras model
        
    Returns:
        tuple: (predicted_class, confidence_score)
        Also accessible as dict via .predicted_class and .confidence_score
    """
    predictions = model.predict(image, verbose=0)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    # Map index to display label
    predicted_class = CLASS_INDEX_TO_LABEL.get(
        class_index,
        CLASS_LABELS[class_index] if class_index < len(CLASS_LABELS) else f"class_{class_index}"
    )
    
    return predicted_class, confidence


def predict_tumor_dict(
    image: np.ndarray,
    model: tf.keras.Model
) -> dict:
    """Convenience wrapper that returns dict with predicted_class and confidence_score."""
    pred_class, confidence = predict_tumor(image, model)
    return {
        "predicted_class": pred_class,
        "confidence_score": confidence,
    }
